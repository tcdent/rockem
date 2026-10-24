# The API

rockem's own HTTP surface, for our own clients: codey first. The service is `api/`, the
surface is `api/rockem/adapters/internal.py` and `api/rockem/views.py`, and the Rust client
that speaks it is `api/client/`. This document, that code and that crate say the same thing;
if they ever disagree, the code is wrong.

> "The idea is that we grow it as we need to, and we include features that directly
> support our use cases rather than trying to bloat it with all of the industry standard
> conventions."

## Overview

### The system

```
client (codey, the chat page)
   │  HTTP: JSON in, a stream of events out
   ▼
the API service (api/)  ── the edge: one process
   │  the wire (docs/wire.md): token ids in, token ids out, over TCP
   ▼
nodes ── the models' weights; one connection at a time each
```

The API service is where text stops. Everything that knows characters lives here: the
tokenizer, the model's chat template, reading tool calls and reasoning back out of
generated text, stop sequences, and audio preprocessing. The node never sees a character;
it reads ids and writes ids (`docs/wire.md`).

The service reads one bundle (`docs/bundle.md`). A bundle can carry several models, each
with its constants and its edge files under a directory named for it, and one node serves
every model in it. The service is told which model each surface speaks:

| setting | what |
|---|---|
| `ROCKEM_BUNDLE` | the bundle directory |
| `ROCKEM_NODES` | the nodes, comma-separated `host:port` |
| `ROCKEM_MODEL` | the model generation speaks (required; the wire has no default model) |
| `ROCKEM_TRANSCRIBE_MODEL` | the model transcription speaks; unset, transcription answers 503 |
| `ROCKEM_IDENTITY` | `bundle` (default): refuse a node built from another bundle; `any`: accept any |

The service is also the nodes' scheduler. A node serves one connection at a time, across
all its models, and parks a second connector without a word, so the service never opens a
connection to a node it is using: every request, generation or transcription, takes a node
from one line, in arrival order, and gives it back when its stream ends.

The service holds no state between requests. A request carries everything it needs: a
generation carries the whole conversation, and the client keeps it. Repeating the
conversation costs nothing on the node: its store holds the attention rows of every
transcript it has computed and matches each new prompt against them, so only the ids past
the longest match are computed. That holds as long as the front of the conversation stays
the same from one request to the next; a system prompt that changes recomputes everything
after it.

> "I do like that the API server is stateless, so I'm okay with pushing state out to
> client. Although I think it is worth us thinking about this more in the future."

There are no sampling fields anywhere. Temperature, top-k, top-p and the seed are constants
of each model, and a transcript always continues the same way.

### The endpoints

| endpoint | what |
|---|---|
| `POST /rockem/v1/generate` | a conversation (or raw text) in; the model's reply streamed: reasoning, text, tool calls |
| `POST /rockem/v1/transcribe` | a recording in; its text streamed, for the user to review and send as a turn |
| `GET /health` | the model, the bundle's identity, and the line: `{model, identity, nodes, free, waiting}` |
| `GET /` | a chat page for talking to the service by hand |

Beside these, `POST /v1/chat/completions` and `POST /v1/messages` emulate OpenAI's and
Anthropic's formats for tools that already speak them. They stream text only, and are not
described here.

Both of rockem's endpoints share one response form, described next: a stream of events,
the same refusals before it, and the same abort.

## Streams

### Framing

Every response is a stream; there is no non-streaming form. The body is a chunked HTTP/1.1
response of events, each one line `data: {json}` followed by a blank line, the framing
every streaming client already reads. Every event has a `type`.

Text and thinking are sent 450 milliseconds at a time: the tokens generated in that time
arrive as one event, and a call, the end, or a switch between thinking and text sends what
is held at once.

```
data: {"type": "started", "prompt_tokens": 294}

data: {"type": "thinking", "text": "The user wants the file read.", "ids": [...]}

data: {"type": "text", "text": " Let me look.", "ids": [...]}

data: {"type": "tool_call", "id": "call_9ad679bd99d2", "name": "read_file", "arguments": {"path": "src/main.rs"}, "ids": [...]}

data: {"type": "done", "reason": "tool_use", "stop_sequence": null, "usage": {"prompt_tokens": 294, "completion_tokens": 54, "positions": 1048574}, "model": "glm-5.3-flash", "identity": "a50b91c72472b516"}
```

### Events

| type | fields | meaning |
|---|---|---|
| `started` | `prompt_tokens` | a node took the request and has its prompt. First, whenever a node is asked (a transcription with no speech never asks one): before it the request waited in line, after it the node computes the prompt and then generates |
| `thinking` | `text`, `ids` | the model's reasoning, before its reply, for a model that reasons (generate only). Never parsed for calls and never matched against `stop`. Send it back as the assistant message's `thinking` |
| `text` | `text`, `ids` | generated text, in order. `ids` are the token ids behind it; text that could still turn out to be a stop sequence or a call is held until it resolves, so `ids` can arrive with empty `text` |
| `tool_call` | `id`, `name`, `arguments`, `ids` | one complete call, arguments an object (generate only). The id is the service's; send it back as the result's `tool_call_id` |
| `done` | `reason`, `stop_sequence`, `usage`, `model`, `identity` | the end. Always the last event of a stream that did not break |
| `error` | `message` | the generation broke after the stream began (the node went away). Last event; no `done` follows |

A stream ends with exactly one `done` or one `error`. A body that ends with neither was cut
by the network.

`reason`:

| reason | meaning |
|---|---|
| `end_turn` | the model ended its turn: it sampled an end id, or its text opened the next turn |
| `tool_use` | the turn ended having made one or more calls; run them and send the results. A turn's calls come last: text the model writes after them, other than another call, ends the turn and is not sent |
| `stop_sequence` | one of the request's `stop` strings appeared; `stop_sequence` names it, and the text stops before it |
| `max_tokens` | the request's `max_tokens` was reached |
| `context_full` | the node's budget was reached: the conversation is as long as this node can hold |
| `no_speech` | transcription only: the recording has no voice in it, so the service answered without the model. The stream is this `done` alone |

`usage` counts ids: `prompt_tokens` fed after the model's start ids, `completion_tokens`
generated (the end id included), and `positions`, the node's budget for a whole sequence.
`prompt_tokens + completion_tokens` against `positions` is how full the context is. `model`
is the model's name in the bundle and `identity` the bundle's 64-bit identity in hex
(`docs/bundle.md`).

### Before the stream

A request refused before any generation is an ordinary HTTP error with a JSON body
`{"detail": …}`, and no stream:

| status | when |
|---|---|
| 422 | the request does not match its endpoint's shape |
| 400 | the content is refused: the chat template refused the messages, `tools` were given for a model with no tool-call format, or the audio is not audio |
| 413 | the prompt does not fit in the node's budget |
| 503 | no node could be reached, or the endpoint's model is not configured |

### Waiting and aborting

A request waits in line while the nodes are busy; there is no busy error, and `started`
marks the moment it leaves the line. Closing the connection while the node computes the
prompt or generates aborts the request and frees the node at once (the node checks before
every step). A request whose client left while it was still waiting in line is dropped when
it reaches the node, within one step.

## POST /rockem/v1/generate

A conversation in, the model's reply out.

### Request

```json
{
  "messages": [
    {"role": "system", "content": "You are Codey, an AI coding assistant."},
    {"role": "user", "content": "What is in src/main.rs?"},
    {"role": "assistant", "content": "", "thinking": "The user wants the file read.", "tool_calls": [
      {"id": "call_1a2b", "name": "read_file", "arguments": {"path": "src/main.rs"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1a2b", "content": "fn main() {}\n"}
  ],
  "tools": [
    {"name": "read_file", "description": "Read a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
  ],
  "max_tokens": 1024,
  "stop": [],
  "reasoning_effort": "low"
}
```

| field | meaning |
|---|---|
| `messages` | the conversation, oldest first. Exactly one of `messages` or `prompt` |
| `prompt` | raw text, fed as it is after the model's start ids: no template, no turns, no tools |
| `tools` | functions the model may call; needs `messages`. Omitted or empty: no tools |
| `max_tokens` | ids to generate at most. Omitted: until the turn ends or the node's budget is full |
| `stop` | up to 8 strings; the text ends before the first one that appears |
| `reasoning_effort` | `low`, `high` or `max`: how much a reasoning model thinks before its reply, passed to its template's variable of that name (GLM-5.3-Flash writes it as the prompt's first line; there is no token budget). Omitted: `low`. A model whose template does not read it ignores it. Keep it fixed for a conversation: changing it changes the prompt's front, and the node recomputes everything after |

A message:

| field | meaning |
|---|---|
| `role` | `system`, `user`, `assistant` or `tool` |
| `content` | its text; may be empty for an assistant message that only made calls |
| `thinking` | assistant only: its reasoning, exactly as the `thinking` events gave it. Send it back: the model's template renders past reasoning, so a turn sent with its reasoning renders as it was generated and the node's cache keeps matching past it |
| `tool_calls` | assistant only: the calls it made, `{id, name, arguments}`, arguments an object |
| `tool_call_id` | tool only, and required there: the id of the call this is the result of |

A tool is `{name, description, parameters}`, `parameters` a JSON Schema object.

The reply arrives as `started`, then `thinking` events for a model that reasons, then
`text` and `tool_call` events, then `done`.

### Tool calls

Everything the model reads about tools comes from its chat template: the list of tools,
how a past call and its result are written, any instruction that calls exist. The service
passes `tools` and `messages` to the template and adds nothing of its own. Reading calls
back out of the generated text is a per-model parser in the service (`api/rockem/calls.py`),
checked against the same template by round trip. A client never sees a model's call
syntax: it sends and receives the flat `{id, name, arguments}` above, whatever the model.

A model trained for tool use brings its format in its checkpoint (GLM-5.3-Flash does).
DeepSeek-V2-Lite was not, and its bundle carries a format we wrote; it produces calls that
parse but uses them poorly.

### Reasoning

A model that reasons writes it between markers its template defines (`<think>` and
`</think>` for GLM-5.3-Flash), usually because the template's generation prompt opens them.
The service splits the reply at those markers: what is inside arrives as `thinking`, the
markers themselves are dropped, and only what follows is `text`, parsed for calls and
matched against stop sequences. The markers are per model, beside its call format
(`api/rockem/calls.py`); a model without them sends no `thinking` events.

## POST /rockem/v1/transcribe

A recording in, its text out, for the user to review before sending it as an ordinary turn
of a conversation. The audio itself never enters a conversation.

> tcdent (2026-10-23): "thinking of it as a separate transcribe step so codey just sees
> parsed text which gives me an opportunity to review."

### Request

```json
{"audio": "<base64 of the recording>", "language": "en", "max_tokens": 4096}
```

| field | meaning |
|---|---|
| `audio` | the recording, base64: any format libsndfile reads (WAV, FLAC, OGG, MP3), any rate, any channels |
| `language` | optional, a code like `en`. Granite detects the language itself and ignores this |
| `max_tokens` | optional: ids to generate at most |

The text arrives as `started`, then `text` events, then `done` with `end_turn`.

A recording with no voice in it ends at once with `done` and `no_speech`, and nothing else: no
node is asked. Sent to the model, silence and noise come back as invented text (one second of
silence became "The 2019–20 season was the 117th season of competitive football by Rangers.").
The service counts the clip's voiced frames, those whose spectrum over 120 Hz–4 kHz is peaked
rather than flat, and needs 0.2 s of them; a single short word measures about 0.34 s. Loudness
only drops near-digital silence (−70 dBFS), so quiet speech still goes through.

### What the service does

Every step before the model's encoder happens in the service: decode, average the
channels, resample to the model's rate, pad to whole windows, compute the spectrogram over
the whole clip and cut it into windows. That pipeline is the reference's (for Granite, torchaudio's
mel spectrogram with a 512-point FFT over a 400-sample window, matched to its float32 rounding),
with the model's mel filterbank read from the bundle. The service renders the model's
transcription prompt and sends the windows to the node with the wire's AUDIO verb
(`docs/queue/094-audio-verb.md`).

The transcription model is Granite Speech 4.1 2B (`granite-speech-4.1-2b`, since 2026-10-24;
tcdent: "our transcription API will only serve the new model"). It hears 3-second windows, 30
placeholders each (10 a second of audio), so a recording's length is bounded by the node's
context. The prompt is the model card's raw transcription request: the text comes back
lowercase and unpunctuated, for word accuracy. The model detects the language, so `language` is
accepted and ignored. It shares the nodes, and their one line, with the generation model.

## Versioning

The version is in the path, `/rockem/v1/`. The Rust client is published to the house registry
(crates.a10k.co) as `rockem-client` 0.1.x until rockem's first public release, only the last
number moving (tcdent, 2026-10-23). A change a client must know about is a new version beside the old
one. Fields may be added to events and to a request's optional fields within a version; a
client ignores fields and event types it does not know. The Rust client's `Event` and
`Reason` are non-exhaustive for this reason: a match on them keeps a `_` arm.

v1 is still a development version: additions land in it without a bump.

> tcdent (2026-10-21): "this is still a dev version of the Rockem protocol, so I don't
> think that we need a protocol bump yet."

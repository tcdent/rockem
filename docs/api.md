# The API

rockem's own HTTP surface, for our own clients: codey first. The service is `api/`, this
surface is `api/rockem/adapters/internal.py`, and the Rust client that speaks it is
`api/client/`. This document, that module and that crate say the same thing; if they ever
disagree, the code is wrong. The OpenAI- and Anthropic-shaped surfaces beside it emulate
other people's formats and are not described here.

> "The idea is that we grow it as we need to, and we include features that directly
> support our use cases rather than trying to bloat it with all of the industry standard
> conventions."

## Shape

One endpoint, `POST /rockem/v1/generate`, a JSON request in, a stream out. There is no
non-streaming form: every response is a chunked HTTP/1.1 body of events. Text and thinking
are sent 450 milliseconds at a time: the tokens generated in that time arrive as one event,
and a call, the end, or a switch between thinking and text sends what is held at once.

The service holds no state between requests. A request carries the whole conversation,
and the client keeps it. Repeating the conversation costs nothing on the node: its store
holds the attention rows of every transcript it has computed and matches each new prompt
against them, so only the ids past the longest match are computed (`docs/wire.md`). That
holds as long as the front of the conversation stays the same from one request to the
next; a system prompt that changes recomputes everything after it.

> "I do like that the API server is stateless, so I'm okay with pushing state out to
> client. Although I think it is worth us thinking about this more in the future."

## Request

```json
{
  "messages": [
    {"role": "system", "content": "You are Codey, an AI coding assistant."},
    {"role": "user", "content": "What is in src/main.rs?"},
    {"role": "assistant", "content": "", "tool_calls": [
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
| `prompt` | raw text, fed as it is with BOS before it: no template, no turns, no tools |
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

There are no sampling fields. Temperature, top-k, top-p and the seed are constants of the
model, and a transcript always continues the same way.

## Events

The body is a sequence of events, each one line `data: {json}` followed by a blank line,
the framing every streaming client already reads. Every event has a `type`.

```
data: {"type": "started", "prompt_tokens": 294}

data: {"type": "thinking", "text": "The user wants the file read.", "ids": [...]}

data: {"type": "text", "text": " Let me look.", "ids": [6000, 1144, 1068, 13]}

data: {"type": "tool_call", "id": "call_9ad679bd99d2", "name": "read_file", "arguments": {"path": "src/main.rs"}, "ids": [...]}

data: {"type": "done", "reason": "tool_use", "stop_sequence": null, "usage": {"prompt_tokens": 294, "completion_tokens": 54, "positions": 99999}, "model": "deepseek-v2-lite", "identity": "bfbdd806f6549722"}
```

| type | fields | meaning |
|---|---|---|
| `started` | `prompt_tokens` | a node took the request and has its prompt. Always first: before it the request waited in line, after it the node computes the prompt and then generates |
| `thinking` | `text`, `ids` | the model's reasoning, before its reply, for a model that reasons. Never parsed for calls and never matched against `stop`. Send it back as the assistant message's `thinking` |
| `text` | `text`, `ids` | generated text, in order. `ids` are the token ids behind it; text that could still turn out to be a stop sequence or a call is held until it resolves, so `ids` can arrive with empty `text` |
| `tool_call` | `id`, `name`, `arguments`, `ids` | one complete call, arguments an object. The id is the service's; send it back as the result's `tool_call_id` |
| `done` | `reason`, `stop_sequence`, `usage`, `model`, `identity` | the end. Always the last event of a stream that did not break |
| `error` | `message` | the generation broke after the stream began (the node went away). Last event; no `done` follows |

A stream ends with exactly one `done` or one `error`. A body that ends with neither was cut
by the network.

`reason`:

| reason | meaning |
|---|---|
| `end_turn` | the model ended its turn: it sampled EOS, or its text opened the next turn |
| `tool_use` | the turn ended having made one or more calls; run them and send the results. A turn's calls come last: text the model writes after them, other than another call, ends the turn and is not sent |
| `stop_sequence` | one of the request's `stop` strings appeared; `stop_sequence` names it, and the text stops before it |
| `max_tokens` | the request's `max_tokens` was reached |
| `context_full` | the node's budget was reached: the conversation is as long as this node can hold |

`usage` counts ids: `prompt_tokens` fed after BOS, `completion_tokens` generated (EOS
included), and `positions`, the node's budget for a whole sequence. `prompt_tokens +
completion_tokens` against `positions` is how full the context is. `model` is the bundle's
model name and `identity` its 64-bit identity in hex (`docs/bundle.md`).

## Before the stream

A request refused before any generation is an ordinary HTTP error with a JSON body
`{"detail": …}`, and no stream:

| status | when |
|---|---|
| 422 | the request does not match the shape above |
| 400 | the chat template refused the messages, or `tools` were given for a model with no tool-call format |
| 413 | the rendered prompt does not fit in the node's budget |
| 503 | no node could be reached |

A request waits in line while the nodes are busy; there is no busy error. Closing the
connection at any point aborts the generation and frees the node.

## Tool calls

Everything the model reads about tools comes from its chat template: the list of tools,
how a past call and its result are written, any instruction that calls exist. The service
passes `tools` and `messages` to the template and adds nothing of its own. Reading calls
back out of the generated text is a per-model parser in the service (`api/rockem/calls.py`),
checked against the same template by round trip. A client never sees a model's call
syntax: it sends and receives the flat `{id, name, arguments}` above, whatever the model.

A model trained for tool use brings its format in its checkpoint. DeepSeek-V2-Lite was not,
and its bundle carries a format we wrote; it produces calls that parse but uses them poorly.

## Reasoning

A model that reasons writes it between markers its template defines (`<think>` and
`</think>` for GLM-5.3-Flash), usually because the template's generation prompt opens them.
The service splits the reply at those markers: what is inside arrives as `thinking`, the
markers themselves are dropped, and only what follows is `text`, parsed for calls and
matched against stop sequences. The markers are per model, beside its call format
(`api/rockem/calls.py`); a model without them sends no `thinking` events.

## Transcription

`POST /rockem/v1/transcribe` turns a recording into text for the user to review, then send as
an ordinary turn. The audio never enters a conversation, and the stream is the same as
generate's: `started`, `text` events, then `done` (or `error`).

```json
{"audio": "<base64 of the recording>", "language": "en", "max_tokens": 4096}
```

| field | meaning |
|---|---|
| `audio` | the recording, base64: any format libsndfile reads (WAV, FLAC, OGG, MP3), any rate, any channels |
| `language` | optional, a code like `en`; omitted, the model detects the language |
| `max_tokens` | optional, as for generate |

The service does every step before the model's encoder: decode, average the channels, resample
to the model's rate, pad to whole windows, compute the spectrogram over the whole clip and cut it
into windows. That pipeline is the reference's (mistral-common's loader and transformers'
Whisper features, matched to within one f32 rounding step). It renders the model's transcription
prompt and sends the windows to the node with the wire's AUDIO verb (`docs/queue/094-audio-verb.md`).
Errors before the stream: 400 for bytes that are not audio or a malformed language, 413 for a
clip longer than the node can hold, 503 when the service has no transcription model
(`ROCKEM_TRANSCRIBE_MODEL`). The transcription model shares the nodes, and their one line, with
the chat model.

## Versioning

The version is in the path. A change a client must know about is a new version beside the
old one, and the Rust client's version names the API version it speaks. Fields may be added
to events and to the request's optional fields within a version; a client ignores fields it
does not know. The Rust client's `Event` and `Reason` are non-exhaustive for this reason: a match on them keeps a `_` arm.

# rockem-client

The Rust client for rockem's own HTTP API. rockem is our inference stack: an API service on
the LAN in front of nodes that run the models. This crate speaks the service's two endpoints:

- **transcription**: a recording in, its text streamed back;
- **generation**: a conversation in, the model's reply streamed back (reasoning, text, tool
  calls).

Every response is a stream of events. You iterate it, and dropping it cancels the request.

## Using it

The crate is on the house registry, `crates.a10k.co`. Cargo needs the registry named once, at
the machine level (`~/.cargo/config.toml`, or your environment image), never in your
project:

```toml
[registries.a10k]
index = "sparse+https://crates.a10k.co/api/v1/crates/"
```

Then depend on it by naming the registry:

```toml
[dependencies]
rockem-client = { version = "0.1", registry = "a10k" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

The service is at `https://rockem.a10k.co`, on the LAN. TLS is built in (rustls, with
its bundled public roots), so the address works with no setup. If your build has no use for
it, depend with `default-features = false`.

## Transcription

Send the recording's bytes in any format libsndfile reads (WAV, FLAC, OGG, MP3), at any
sample rate and any number of channels. The service does all the audio processing. The text
streams back as `Text` events, and the stream ends with `Done`.

```rust,no_run
use rockem_client::{Client, Event, StreamExt, TranscribeRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("https://rockem.a10k.co");
    let audio = std::fs::read("note.wav")?;
    // The language is optional; the model detects the language itself.
    let mut events = client.transcribe(&TranscribeRequest::new(audio).language("en")).await?;
    let mut text = String::new();
    while let Some(event) = events.next().await {
        match event? {
            Event::Text { text: piece, .. } => text.push_str(&piece),
            Event::Done(done) => println!("{text}\n[{:?}, {} ids]", done.reason, done.usage.completion_tokens),
            _ => {}
        }
    }
    Ok(())
}
```

A recording with no voice in it, such as silence, hiss or noise, ends at once with
`Reason::NoSpeech` and no text: the service checks before sending it to the model, which would
otherwise invent a transcript. A single short word is enough to pass.

The model is Granite Speech 4.1 2B. The transcript comes back lowercase and unpunctuated, for
word accuracy, and the model detects the language itself, so `language` is ignored. A
recording's length is bounded by the service's context. The transcript is text for a person to
review: the audio never enters a conversation.

**Status:** while the service has no transcription model deployed, `transcribe` returns
`Error::Status { status: 503, .. }`. Build and test against the API now; the endpoint goes
live without any change on your side.

## Generation

A request carries the whole conversation, oldest message first. The service keeps no state
between requests, so you keep the conversation and send all of it each time. Repeating it is
cheap: the node reuses the computation for whatever stays the same at the front.

```rust,no_run
use rockem_client::{Client, Event, Message, Request, StreamExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("https://rockem.a10k.co");
    let req = Request::chat(vec![
        Message::system("You are a terse assistant."),
        Message::user("Name three primary colors."),
    ])
    .max_tokens(512);
    let mut events = client.generate(&req).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Thinking { .. } => {} // the model's reasoning; send it back with the turn
            Event::Text { text, .. } => print!("{text}"),
            Event::Done(done) => println!("\n[{:?}]", done.reason),
            _ => {}
        }
    }
    Ok(())
}
```

To continue a conversation, append the assistant's turn as it arrived, with
`Message::assistant(text, calls).with_thinking(reasoning)`, then the next message. Send the
reasoning back exactly as the `Thinking` events gave it: the node then reuses the turn's
computation.

Tools are `Tool { name, description, parameters }`, with `parameters` a JSON Schema. A call
arrives as `Event::ToolCall`. Run it and send the result back with
`Message::tool_result(call.id, output)`. `Request::reasoning_effort(Effort::Low | High | Max)`
sets how much the model reasons. It defaults to low; keep it fixed for a conversation.

## The events

| event | meaning |
|---|---|
| `Started { prompt_tokens }` | a node took the request. First, whenever a node is asked: before it, the request waited in line |
| `Thinking { text, ids }` | the model's reasoning, before its reply (generation only) |
| `Text { text, ids }` | text, in order; `ids` are the token ids behind it |
| `ToolCall(call)` | one complete tool call (generation only) |
| `Done(done)` | the end: `done.reason`, `done.usage`, the model's name. Always last |

`Reason` is `EndTurn`, `ToolUse`, `StopSequence`, `MaxTokens`, `ContextFull` or (transcription) `NoSpeech`. `Usage`
counts token ids: `prompt_tokens`, `completion_tokens`, and `positions`, the context budget.

`Event` and `Reason` are non-exhaustive. The service may add event types within this
version, so keep a `_` arm in every match.

## Errors and aborting

| error | when |
|---|---|
| `Error::Status { status, detail }` | refused before the stream: 400 the content (not audio, a template refusal), 413 too long for the context, 422 a malformed request, 503 no node or no model for the endpoint |
| `Error::Generation(message)` | the stream broke after it began (the node went away) |
| `Error::Truncated` | the body ended with neither `Done` nor an error: cut on the way |
| `Error::Decode { line, error }` | a line the service wrote is not an event this version reads |
| `Error::Http(e)` | the request did not reach the service |

A request waits in line while the nodes are busy. There is no busy error. Dropping the
event stream closes the connection, which aborts the request and frees the node.

## Versions

Until rockem's first public release, the crate is `0.1.x`, and only the last number moves.
Depend on `"0.1"` and you get every release. It speaks `/rockem/v1`, which is still a
development version: events and optional request fields may be added, and this crate ignores
what it does not know. Every published version stays available. The API's full description is
`docs/api.md` in the rockem repository.

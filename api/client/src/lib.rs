#![doc = include_str!("../README.md")]
// In the rockem repository this crate, docs/api.md and the service's
// api/rockem/adapters/internal.py say the same thing; if they disagree, the code is wrong.
// The README above is the crate's documentation for the teams that build against it.

use std::pin::Pin;

use futures_util::Stream;
pub use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The path this crate speaks; the version in it is the crate's major version.
pub const PATH: &str = "/rockem/v1/generate";
/// Transcription: audio in, the same events out, the text for the user to review.
pub const TRANSCRIBE_PATH: &str = "/rockem/v1/transcribe";

// --- the request ---

/// What to generate from: `messages` through the model's chat template, or a raw `prompt`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Request {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// How much the model reasons before replying, for a model whose template reads it
    /// (GLM-5.3-Flash). Omitted: `Low`. Keep it fixed for a conversation: it is part of the
    /// prompt's first line, so changing it recomputes everything after.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Effort {
    Low,
    High,
    Max,
}

impl Request {
    pub fn chat(messages: Vec<Message>) -> Self {
        Self { messages: Some(messages), ..Self::default() }
    }
    pub fn raw(prompt: impl Into<String>) -> Self {
        Self { prompt: Some(prompt.into()), ..Self::default() }
    }
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }
    pub fn reasoning_effort(mut self, effort: Effort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    /// Assistant only: its reasoning, as the `Thinking` events gave it. Send it back as it
    /// came: the template renders it, and the node's cache matches the turn as generated.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thinking: String,
    /// Assistant only: the calls it made.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Tool only: the call this is the result of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into(), thinking: String::new(), tool_calls: Vec::new(), tool_call_id: None }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
    /// An assistant turn as it was generated: its text and the calls it made.
    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self { tool_calls, ..Self::new(Role::Assistant, content) }
    }
    /// The reasoning that came before an assistant turn's text.
    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = thinking.into();
        self
    }
    /// A call's result, sent back with the call's id.
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { tool_call_id: Some(call_id.into()), ..Self::new(Role::Tool, content) }
    }
}

/// A function the model may call; `parameters` is a JSON Schema object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub parameters: Value,
}

/// One call: in an assistant message going out, and in a `tool_call` event coming back.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: Map<String, Value>,
    /// The token ids the call was generated as (events only; never sent).
    #[serde(default, skip_serializing)]
    pub ids: Vec<u32>,
}

/// A recording to transcribe: any format the service reads (WAV, FLAC, OGG, MP3), sent as
/// base64. The text comes back as `Text` events, ending in `Done`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscribeRequest {
    pub audio: Vec<u8>,
    /// A code like "en"; `None`, the model detects the language.
    pub language: Option<String>,
    pub max_tokens: Option<u32>,
}

impl TranscribeRequest {
    pub fn new(audio: Vec<u8>) -> Self {
        Self { audio, ..Self::default() }
    }
    pub fn language(mut self, code: impl Into<String>) -> Self {
        self.language = Some(code.into());
        self
    }
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
}

impl Serialize for TranscribeRequest {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use base64::Engine;
        use serde::ser::SerializeMap;
        let mut m = s.serialize_map(None)?;
        m.serialize_entry("audio", &base64::engine::general_purpose::STANDARD.encode(&self.audio))?;
        if let Some(l) = &self.language {
            m.serialize_entry("language", l)?;
        }
        if let Some(n) = self.max_tokens {
            m.serialize_entry("max_tokens", &n)?;
        }
        m.end()
    }
}

// --- the events ---

/// Non-exhaustive: the service may add event types within a version, so a match keeps a
/// `_` arm (and `Unknown` stands for a type this crate does not name yet).
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Event {
    /// A node took the request and has its prompt: what follows is the node's work. First.
    Started { prompt_tokens: u32 },
    /// The model's reasoning before its reply, in order; send it back as the message's `thinking`.
    Thinking { text: String, ids: Vec<u32> },
    /// Generated text, in order; `ids` can arrive with empty text while text is held.
    Text { text: String, ids: Vec<u32> },
    /// One complete call. Run it and send the result with its id.
    ToolCall(ToolCall),
    /// The end: always the last event of a stream that did not break.
    Done(Done),
    /// An event type this version does not know: a newer service. Skip it.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Done {
    pub reason: Reason,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
    pub model: String,
    pub identity: String,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Reason {
    /// The model ended its turn.
    EndTurn,
    /// The turn ended having made calls: run them and send the results.
    ToolUse,
    /// One of the request's `stop` strings appeared.
    StopSequence,
    /// The request's `max_tokens` was reached.
    MaxTokens,
    /// The node's budget was reached.
    ContextFull,
    #[serde(other)]
    Unknown,
}

/// Counts in token ids. `prompt_tokens + completion_tokens` against `positions` is how
/// full the context is.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub positions: u32,
}

// --- errors ---

#[derive(Debug)]
pub enum Error {
    /// The request did not reach the service, or the body broke in transit.
    Http(reqwest::Error),
    /// Refused before the stream began: 400 template, 413 too long, 422 shape, 503 no node.
    Status { status: u16, detail: String },
    /// The generation broke after the stream began (the service's `error` event).
    Generation(String),
    /// A line the service wrote is not an event.
    Decode { line: String, error: serde_json::Error },
    /// The body ended with neither `done` nor `error`: cut on the way.
    Truncated,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Http(e) => write!(f, "rockem: {e}"),
            Error::Status { status, detail } => write!(f, "rockem: {status}: {detail}"),
            Error::Generation(m) => write!(f, "rockem: generation broke: {m}"),
            Error::Decode { line, error } => write!(f, "rockem: not an event ({error}): {line}"),
            Error::Truncated => write!(f, "rockem: stream ended without done"),
        }
    }
}

impl std::error::Error for Error {}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Http(e)
    }
}

// --- the stream, without a socket ---

/// Bytes of a response body in, events out. Events are `data: {json}` lines separated by
/// a blank line; a chunk may end anywhere, so an incomplete event waits for the next one.
#[derive(Default)]
pub struct Decoder {
    buf: Vec<u8>,
    done: bool,
}

impl Decoder {
    /// The complete events in what has arrived so far. An `error` event comes back as
    /// `Err(Error::Generation)`; nothing is returned after it or after `done`.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Result<Event, Error>> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(i) = self.buf.windows(2).position(|w| w == b"\n\n") {
            let chunk: Vec<u8> = self.buf.drain(..i + 2).collect();
            if self.done {
                continue;
            }
            for line in String::from_utf8_lossy(&chunk[..i]).lines() {
                let Some(json) = line.strip_prefix("data: ") else { continue };
                out.push(Self::event(json));
                if !matches!(out.last(), Some(Ok(Event::Started { .. } | Event::Thinking { .. } | Event::Text { .. } | Event::ToolCall(_) | Event::Unknown))) {
                    self.done = true;
                    break;
                }
            }
        }
        out
    }

    fn event(json: &str) -> Result<Event, Error> {
        #[derive(Deserialize)]
        struct Failed {
            r#type: String,
            message: String,
        }
        if let Ok(f) = serde_json::from_str::<Failed>(json) {
            if f.r#type == "error" {
                return Err(Error::Generation(f.message));
            }
        }
        serde_json::from_str(json).map_err(|error| Error::Decode { line: json.to_string(), error })
    }

    /// Whether the stream reached its end, `done` or `error`.
    pub fn finished(&self) -> bool {
        self.done
    }
}

// --- the client ---

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
}

/// The events of one generation, in order; ends after `done`, or with an error.
pub type Events = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;

impl Client {
    /// `base` is the service's root, e.g. `https://rockem.a10k.co`.
    pub fn new(base: &str) -> Self {
        Self::with_http(reqwest::Client::new(), base)
    }

    /// With the caller's own `reqwest::Client` (its TLS, timeouts, proxies).
    pub fn with_http(http: reqwest::Client, base: &str) -> Self {
        Self { http, base: base.trim_end_matches('/').to_string() }
    }

    /// Send the request and stream its events. Refusals before the stream are
    /// `Error::Status`. Dropping the stream closes the connection, which aborts the
    /// generation and frees the node.
    pub async fn generate(&self, req: &Request) -> Result<Events, Error> {
        self.post(PATH, serde_json::to_vec(req).expect("a Request always serializes")).await
    }

    /// Transcribe a recording and stream its text, as `Text` events ending in `Done`.
    /// Refusals before the stream are `Error::Status` (400 not audio, 503 no transcription
    /// model). Dropping the stream aborts it, as for `generate`.
    pub async fn transcribe(&self, req: &TranscribeRequest) -> Result<Events, Error> {
        self.post(TRANSCRIBE_PATH, serde_json::to_vec(req).expect("a TranscribeRequest always serializes")).await
    }

    async fn post(&self, path: &str, body: Vec<u8>) -> Result<Events, Error> {
        let resp = self.http.post(format!("{}{path}", self.base)).header("content-type", "application/json").body(body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let detail = serde_json::from_str::<Value>(&text).ok().and_then(|v| v.get("detail").map(|d| d.as_str().map(String::from).unwrap_or_else(|| d.to_string()))).unwrap_or(text);
            return Err(Error::Status { status: status.as_u16(), detail });
        }
        let bytes = resp.bytes_stream();
        let state = (bytes, Decoder::default(), std::collections::VecDeque::new(), false);
        Ok(Box::pin(futures_util::stream::unfold(state, |(mut bytes, mut dec, mut queue, mut ended)| async move {
            loop {
                if let Some(item) = queue.pop_front() {
                    return Some((item, (bytes, dec, queue, ended)));
                }
                if ended || dec.finished() {
                    return None;
                }
                match bytes.next().await {
                    Some(Ok(chunk)) => queue.extend(dec.feed(&chunk)),
                    Some(Err(e)) => {
                        ended = true;
                        queue.push_back(Err(Error::Http(e)));
                    }
                    None => {
                        ended = true;
                        if !dec.finished() {
                            queue.push_back(Err(Error::Truncated));
                        }
                    }
                }
            }
        })))
    }
}

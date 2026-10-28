//! The contract with the service, pinned by files the Python suite writes and checks
//! (`api/tests/test_contract.py`): the request Rust builds is `request.json` exactly, and
//! every `*.stream` decodes the same however its bytes are split.

use rockem_client::{Decoder, Done, Effort, Error, Event, Message, Reason, Request, Tool, ToolCall, TranscribeRequest, Usage};
use serde_json::{json, Value};

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// Decode `body` fed in pieces of `step` bytes, and again split at every single position.
fn decode_all_ways(body: &str) -> Vec<Result<Event, String>> {
    let simplify = |v: Vec<Result<Event, Error>>| v.into_iter().map(|r| r.map_err(|e| e.to_string())).collect::<Vec<_>>();
    let whole = simplify(Decoder::default().feed(body.as_bytes()));
    for step in 1..=7 {
        let mut d = Decoder::default();
        let mut got = Vec::new();
        for piece in body.as_bytes().chunks(step) {
            got.extend(d.feed(piece));
        }
        assert_eq!(simplify(got), whole, "pieces of {step} bytes");
    }
    for cut in 0..=body.len() {
        let mut d = Decoder::default();
        let mut got = d.feed(&body.as_bytes()[..cut]);
        got.extend(d.feed(&body.as_bytes()[cut..]));
        assert_eq!(simplify(got), whole, "split at {cut}");
    }
    whole
}

fn done(reason: Reason, stop: Option<&str>, usage: (u32, u32, u32)) -> Event {
    Event::Done(Done {
        reason,
        stop_sequence: stop.map(String::from),
        usage: Usage { prompt_tokens: usage.0, completion_tokens: usage.1, positions: usage.2 },
        model: "tiny".into(),
        identity: "0000000000000000".into(),
    })
}

#[test]
fn the_request_rust_builds_is_the_fixture() {
    let call = ToolCall { id: "call_0".into(), name: "read".into(), arguments: json!({"path": "b"}).as_object().unwrap().clone(), ids: vec![] };
    let req = Request::chat(vec![
        Message::system("You are Codey, an AI coding assistant."),
        Message::user("hello"),
        Message::assistant("", vec![call]).with_thinking("Read b first."),
        Message::tool_result("call_0", "world"),
    ])
    .tools(vec![Tool {
        name: "read".into(),
        description: "Read a file.".into(),
        parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    }])
    .max_tokens(64)
    .stop(vec!["stop".into()])
    .reasoning_effort(Effort::High);
    let ours: Value = serde_json::to_value(&req).unwrap();
    let theirs: Value = serde_json::from_str(&fixture("request.json")).unwrap();
    assert_eq!(ours, theirs);
    assert_eq!(serde_json::from_value::<Request>(theirs).unwrap(), req);
}

#[test]
fn a_tool_call_stream() {
    let events = decode_all_ways(&fixture("tool_call.stream"));
    let call = ToolCall { id: "call_0".into(), name: "read".into(), arguments: json!({"path": "a"}).as_object().unwrap().clone(), ids: vec![8, 9, 10] };
    assert_eq!(
        events,
        vec![
            Ok(Event::Started { prompt_tokens: 11 }),
            Ok(Event::Text { text: " paris".into(), ids: vec![2] }),
            Ok(Event::ToolCall(call)),
            Ok(done(Reason::ToolUse, None, (11, 5, 63))),
        ]
    );
}

#[test]
fn a_stop_sequence_stream() {
    let events = decode_all_ways(&fixture("stop_sequence.stream"));
    assert_eq!(events.last().unwrap(), &Ok(done(Reason::StopSequence, Some("stop"), (1, 2, 11))));
    let text: String = events.iter().filter_map(|e| match e {
        Ok(Event::Text { text, .. }) => Some(text.as_str()),
        _ => None,
    }).collect();
    assert_eq!(text, " paris ");
}

#[test]
fn an_error_stream_ends_in_an_error() {
    let events = decode_all_ways(&fixture("error.stream"));
    assert_eq!(events.len(), 3);
    assert_eq!(events[2], Err("rockem: generation broke: node: the node closed with 9 positions left".into()));
}

#[test]
fn newer_services_are_tolerated_and_the_stream_stops_at_done() {
    let body = "data: {\"type\":\"progress\",\"text\":\"hm\"}\n\n\
                data: {\"type\":\"text\",\"text\":\"a\",\"ids\":[1],\"new_field\":true}\n\n\
                data: {\"type\":\"done\",\"reason\":\"some_new_reason\",\"stop_sequence\":null,\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"positions\":9},\"model\":\"m\",\"identity\":\"x\"}\n\n\
                data: {\"type\":\"text\",\"text\":\"after\",\"ids\":[]}\n\n";
    let events = decode_all_ways(body);
    assert_eq!(events.len(), 3);
    assert_eq!(events[0], Ok(Event::Unknown));
    assert!(matches!(&events[2], Ok(Event::Done(d)) if d.reason == Reason::Unknown));
}

#[test]
fn a_thinking_stream() {
    let events = decode_all_ways(&fixture("thinking.stream"));
    assert_eq!(
        events[..3],
        [
            Ok(Event::Started { prompt_tokens: 1 }),
            Ok(Event::Thinking { text: " hello".into(), ids: vec![12, 0] }),
            Ok(Event::Text { text: " paris".into(), ids: vec![13, 2] }),
        ]
    );
    assert_eq!(events[3], Ok(done(Reason::EndTurn, None, (1, 5, 11))));
}

#[test]
fn the_transcribe_request_rust_builds_is_the_fixture() {
    let req = TranscribeRequest::new(b"RIFF".to_vec()).language("en").max_tokens(512).keywords(["rockem", "GLM-5.3-Flash"]);
    let ours: Value = serde_json::to_value(&req).unwrap();
    let theirs: Value = serde_json::from_str(&fixture("transcribe.json")).unwrap();
    assert_eq!(ours, theirs);
    assert_eq!(serde_json::to_value(TranscribeRequest::new(vec![])).unwrap(), json!({"audio": ""}));
}

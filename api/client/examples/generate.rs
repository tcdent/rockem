//! `cargo run --example generate -- <base url> <question>`: one chat turn with two tools,
//! printed as it streams.
use rockem_client::{Client, Event, Message, Request, StreamExt, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let base = args.next().unwrap_or_else(|| "http://localhost:8765".into());
    let question = args.next().unwrap_or_else(|| "List the files in the current directory.".into());
    let tools = vec![
        Tool { name: "read_file".into(), description: "Read a file's contents.".into(), parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}) },
        Tool { name: "shell".into(), description: "Run a bash command and return its output.".into(), parameters: json!({"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}) },
    ];
    let req = Request::chat(vec![Message::system("You are Codey, an AI coding assistant running in a terminal interface."), Message::user(question)]).tools(tools).max_tokens(200);
    let mut events = Client::new(&base).generate(&req).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Started { prompt_tokens } => println!("[started, {prompt_tokens} prompt ids]"),
            Event::Thinking { text, .. } => print!("\x1b[2m{text}\x1b[0m"),
            Event::Text { text, .. } => print!("{text}"),
            Event::ToolCall(call) => println!("\n[tool_call {} {} {}]", call.id, call.name, serde_json::Value::Object(call.arguments)),
            Event::Done(d) => println!("\n[done {:?}, {} + {} of {} positions]", d.reason, d.usage.prompt_tokens, d.usage.completion_tokens, d.usage.positions),
            _ => {}
        }
    }
    Ok(())
}

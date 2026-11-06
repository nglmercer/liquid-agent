//! Phase 0 probe: load the installed LFM3 GGUF with mistral.rs and measure
//! load time, time-to-first-token, decode speed, and stream cancellation.
//!
//! Usage: mistralrs-probe <models_dir> <gguf_file> [mmproj_file]
//! Example: mistralrs-probe ~/.local/share/liquid-agent/models LFM3-1.7B-Q4_K_M.gguf

use std::time::Instant;

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use mistralrs::{GgufModelBuilder, ModelDType, Response, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: mistralrs-probe <models_dir> <gguf_file> [mmproj_file]");
    }
    let dir = args[1].clone();
    let file = args[2].clone();

    let started = Instant::now();
    let mut builder = GgufModelBuilder::new(dir, vec![file]).with_logging();
    // PROBE_DTYPE=f32|f16|bf16 selects the compute type (default: the SDK's choice).
    builder = match std::env::var("PROBE_DTYPE").as_deref() {
        Ok("f16") => builder.with_dtype(ModelDType::F16),
        Ok("bf16") => builder.with_dtype(ModelDType::BF16),
        Ok("f32") => builder.with_dtype(ModelDType::F32),
        _ => builder,
    };
    if let Some(mmproj) = args.get(3) {
        builder = builder.with_mmproj_files(vec![mmproj.clone()]);
    }
    let model = builder.build().await.context("load failed")?;
    println!("LOAD_MS {}", started.elapsed().as_millis());

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "In two sentences, what is a liquid neural network?",
    );

    // Full generation: time to first token and decode rate.
    let started = Instant::now();
    let mut stream = model
        .stream_chat_request(messages.clone())
        .await
        .context("stream request failed")?;
    let mut first: Option<u128> = None;
    let mut chunks = 0usize;
    let mut text = String::new();
    while let Some(response) = stream.next().await {
        match response {
            Response::Chunk(chunk) => {
                if first.is_none() {
                    first = Some(started.elapsed().as_millis());
                }
                chunks += 1;
                if let Some(choice) = chunk.choices.first() {
                    if let Some(delta) = &choice.delta.content {
                        text.push_str(delta);
                    }
                }
                // Streamed chat ends with usage on the last chunk; no Done.
                if let Some(usage) = &chunk.usage {
                    let total = started.elapsed().as_secs_f64();
                    println!("TTFT_MS {}", first.unwrap_or(0));
                    println!("CHUNKS {chunks} TOTAL_S {total:.2}");
                    println!(
                        "PROMPT_TOK {} COMPL_TOK {} PROMPT_TOK_S {:.2} COMPL_TOK_S {:.2}",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.avg_prompt_tok_per_sec,
                        usage.avg_compl_tok_per_sec
                    );
                    break;
                }
            }
            Response::Done(done) => {
                let total = started.elapsed().as_secs_f64();
                println!("TTFT_MS {}", first.unwrap_or(0));
                println!("CHUNKS {chunks} TOTAL_S {total:.2}");
                println!(
                    "PROMPT_TOK_S {:.2} COMPL_TOK_S {:.2}",
                    done.usage.avg_prompt_tok_per_sec, done.usage.avg_compl_tok_per_sec
                );
                break;
            }
            other => {
                println!("UNEXPECTED {:?}", std::mem::discriminant(&other));
            }
        }
    }
    println!("TEXT {}", text.trim());

    // Cancellation: drop the stream after a few chunks and measure how long a
    // new request takes to start. A slow start means the engine kept generating.
    let mut stream = model
        .stream_chat_request(messages.clone())
        .await
        .context("second stream request failed")?;
    let mut seen = 0;
    while let Some(response) = stream.next().await {
        if matches!(response, Response::Chunk(_)) {
            seen += 1;
            if seen == 5 {
                break;
            }
        }
    }
    drop(stream);
    let resumed = Instant::now();
    let mut again = model
        .stream_chat_request(messages)
        .await
        .context("third stream request failed")?;
    let _ = again.next().await;
    println!("AFTER_CANCEL_FIRST_RESPONSE_MS {}", resumed.elapsed().as_millis());

    Ok(())
}

//! Phase 0 probe for LFM3 Omni (GGUF + multimodal projector): image QA, ASR,
//! and chat audio output. Also reports peak resident memory.
//!
//! Usage: omni <models_dir> <gguf_file> <mmproj_file> <wav_for_asr> <out_dir> <mode...>
//! Modes: image | asr | tts | all

use std::{path::PathBuf, time::Instant};

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{DynamicImage, Rgb, RgbImage};
use mistralrs::{
    AudioInput, ChatAudioFormat, ChatAudioMode, GgufModelBuilder, RequestBuilder, TextMessageRole,
};

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        bail!("usage: omni <models_dir> <gguf> <mmproj> <wav> <out_dir> <image|asr|tts|all>...");
    }
    let (dir, gguf, mmproj, wav, out) = (&args[1], &args[2], &args[3], &args[4], &args[5]);
    let modes: Vec<&str> = args[6..].iter().map(String::as_str).collect();
    let run = |m: &str| modes.contains(&"all") || modes.contains(&m);

    let started = Instant::now();
    let model = GgufModelBuilder::new(dir.clone(), vec![gguf.clone()])
        .with_mmproj_files(vec![mmproj.clone()])
        .build()
        .await
        .context("omni load failed")?;
    println!("LOAD_MS {}", started.elapsed().as_millis());

    if run("image") {
        // A solid red 64x64 square. The right answer is "red".
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([255, 0, 0])));
        let req = RequestBuilder::new().add_image_message(
            TextMessageRole::User,
            "What single color is this image? Answer with one word.",
            vec![img],
        );
        let started = Instant::now();
        let resp = model
            .send_chat_request(req)
            .await
            .context("image request failed")?;
        println!("IMAGE_MS {}", started.elapsed().as_millis());
        println!(
            "IMAGE_ANSWER {:?}",
            resp.choices[0].message.content.as_deref().unwrap_or("")
        );
    }

    if run("asr") {
        let audio = AudioInput::read_wav(wav).context("read wav failed")?;
        let req = RequestBuilder::new().add_audio_message(
            TextMessageRole::User,
            "Transcribe this audio exactly. Output only the transcript.",
            vec![audio],
        );
        let started = Instant::now();
        let resp = model
            .send_chat_request(req)
            .await
            .context("asr request failed")?;
        println!("ASR_MS {}", started.elapsed().as_millis());
        println!(
            "ASR_TEXT {:?}",
            resp.choices[0].message.content.as_deref().unwrap_or("")
        );
    }

    if run("tts") {
        let req = RequestBuilder::new()
            .add_message(
                TextMessageRole::User,
                "Say the following sentence out loud: The weather is clear today.",
            )
            .with_audio_output(ChatAudioMode::TextAudio, ChatAudioFormat::Wav);
        let started = Instant::now();
        let resp = model
            .send_chat_request(req)
            .await
            .context("tts request failed")?;
        println!("TTS_MS {}", started.elapsed().as_millis());
        let msg = &resp.choices[0].message;
        println!(
            "TTS_TEXT {:?}",
            msg.content.as_deref().unwrap_or("")
        );
        match msg.audio.as_ref().and_then(|a| a.data.as_ref()) {
            Some(data) => {
                let bytes = STANDARD.decode(data).context("bad base64")?;
                let path = PathBuf::from(out).join("tts-output.wav");
                std::fs::write(&path, &bytes)?;
                println!("TTS_AUDIO_BYTES {} FILE {}", bytes.len(), path.display());
            }
            None => println!("TTS_AUDIO_BYTES 0 (no audio in response)"),
        }
    }

    if run("stream") {
        // Streamed TTS with PCM16 output and thinking disabled, as the server will use it.
        let req = RequestBuilder::new()
            .add_message(
                TextMessageRole::User,
                "Say the following sentence out loud: The weather is clear today.",
            )
            .enable_thinking(false)
            .set_sampler_temperature(0.0)
            .set_sampler_max_len(256)
            .with_audio_output(ChatAudioMode::TextAudio, ChatAudioFormat::Pcm16);
        let started = Instant::now();
        let mut stream = model
            .stream_chat_request(req)
            .await
            .context("stream request failed")?;
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut audio = Vec::new();
        let mut first_audio_ms: Option<u128> = None;
        let mut usage = None;
        while let Some(response) = stream.next().await {
            if let mistralrs::Response::Chunk(chunk) = response {
                for choice in &chunk.choices {
                    if let Some(t) = &choice.delta.content {
                        text.push_str(t);
                    }
                    if let Some(r) = &choice.delta.reasoning_content {
                        reasoning.push_str(r);
                    }
                    if let Some(a) = choice.delta.audio.as_ref().and_then(|a| a.data.as_ref()) {
                        if first_audio_ms.is_none() {
                            first_audio_ms = Some(started.elapsed().as_millis());
                        }
                        audio.extend(STANDARD.decode(a).context("bad base64 chunk")?);
                    }
                }
                if chunk.usage.is_some() {
                    usage = chunk.usage.clone();
                }
            }
        }
        println!("STREAM_TOTAL_MS {}", started.elapsed().as_millis());
        println!("STREAM_TEXT {:?}", text);
        println!("STREAM_REASONING_LEN {}", reasoning.len());
        println!("STREAM_FIRST_AUDIO_MS {:?} AUDIO_BYTES {}", first_audio_ms, audio.len());
        if let Some(u) = usage {
            println!("STREAM_TOKENS prompt {} completion {}", u.prompt_tokens, u.completion_tokens);
        }
        let path = PathBuf::from(out).join("stream-output.pcm16");
        std::fs::write(&path, &audio)?;
        println!("STREAM_AUDIO_FILE {}", path.display());
    }

    println!("PEAK_RSS_KIB {}", peak_rss_kib().unwrap_or(0));
    Ok(())
}

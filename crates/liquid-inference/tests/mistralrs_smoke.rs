//! Opt-in smoke tests: real GGUF models through the mistral.rs adapter and the
//! manager. They are ignored by default because they need model weights.
//!
//! Text (LFM3-1.7B):
//!   LIQUID_INFERENCE_GGUF=/path/LFM3-1.7B-Q4_K_M.gguf \
//!   cargo test --release -p liquid-inference --features mistralrs --test mistralrs_smoke text -- --ignored --nocapture
//!
//! Omni (image, ASR, speech); also needs the projector and a WAV clip:
//!   LIQUID_INFERENCE_OMNI_GGUF=/path/LFM3-Omni-2B-Q4_K_M.gguf \
//!   LIQUID_INFERENCE_MMPROJ=/path/mmproj-LFM3-Omni-2B-Q4_K_M.gguf \
//!   LIQUID_INFERENCE_WAV=/path/clip.wav \
//!   cargo test --release -p liquid-inference --features mistralrs --test mistralrs_smoke omni -- --ignored --nocapture

#![cfg(feature = "mistralrs")]

use std::{path::PathBuf, sync::Arc};

use futures_util::StreamExt;
use image::{DynamicImage, Rgb, RgbImage};
use liquid_inference::{
    ChatMessage, ChatRequest, DevicePreference, EventStream, InferenceError, InferenceEvent,
    InferenceManager, LoadSpec, MistralEngine, ModelRole, ModelState,
};
use tokio_util::sync::CancellationToken;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} to run this test")))
}

fn spec(role: ModelRole, weights: PathBuf, projector: Option<PathBuf>) -> LoadSpec {
    // LIQUID_INFERENCE_CONTEXT overrides the planned context, for memory experiments.
    let context_tokens = std::env::var("LIQUID_INFERENCE_CONTEXT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    LoadSpec {
        role,
        weights,
        projector,
        context_tokens,
        max_concurrent: 1,
        device: DevicePreference::Cpu,
    }
}

/// Drains a stream, returning (text, reasoning, audio bytes, saw usage, finished).
async fn collect(mut stream: EventStream) -> (String, String, Vec<u8>, bool, bool) {
    let (mut text, mut reasoning, mut audio) = (String::new(), String::new(), Vec::new());
    let (mut usage, mut finished) = (false, false);
    while let Some(event) = stream.next().await {
        match event.expect("no error during generation") {
            InferenceEvent::TextDelta(d) => text.push_str(&d),
            InferenceEvent::ReasoningDelta(d) => reasoning.push_str(&d),
            InferenceEvent::AudioChunk { pcm16, sample_rate } => {
                assert_eq!(sample_rate, 24_000);
                audio.extend(pcm16);
            }
            InferenceEvent::Usage { .. } => usage = true,
            InferenceEvent::Finished => finished = true,
            InferenceEvent::ToolCallDelta(_) => {}
        }
    }
    (text, reasoning, audio, usage, finished)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LIQUID_INFERENCE_GGUF pointing at a GGUF text model"]
async fn text_model_streams_cancels_and_unloads() {
    let manager = InferenceManager::new(Arc::new(MistralEngine));
    manager
        .load(spec(
            ModelRole::Small,
            env_path("LIQUID_INFERENCE_GGUF"),
            None,
        ))
        .await
        .expect("load should succeed");

    let caps = manager.capabilities(ModelRole::Small).expect("loaded");
    assert!(caps.text);
    assert!(!caps.vision && !caps.asr && !caps.tts && !caps.voice_selection);

    let request = |max| {
        ChatRequest::new(
            vec![
                ChatMessage::text("system", "You are a concise assistant."),
                ChatMessage::text(
                    "user",
                    "Explain in two sentences what a liquid neural network is.",
                ),
            ],
            max,
        )
    };

    // Full generation with thinking disabled: answer text, usage, Finished, no reasoning.
    let stream = manager
        .generate(ModelRole::Small, request(256), CancellationToken::new())
        .expect("generate");
    let (text, reasoning, _, usage, finished) = collect(stream).await;
    assert!(!text.trim().is_empty(), "model produced no answer text");
    assert!(
        reasoning.is_empty(),
        "thinking was disabled but reasoning arrived"
    );
    assert!(usage && finished, "stream must end with usage and Finished");

    // Cancellation mid-stream surfaces as Cancelled.
    let cancel = CancellationToken::new();
    let mut stream = manager
        .generate(ModelRole::Small, request(512), cancel.clone())
        .expect("generate");
    loop {
        match stream.next().await {
            Some(Ok(InferenceEvent::TextDelta(_))) => break,
            Some(Ok(_)) => continue,
            other => panic!("stream ended before any text: {other:?}"),
        }
    }
    cancel.cancel();
    let mut saw_cancelled = false;
    while let Some(item) = stream.next().await {
        if matches!(item, Err(InferenceError::Cancelled)) {
            saw_cancelled = true;
        }
    }
    assert!(saw_cancelled, "cancel must surface as Cancelled");

    // The slot is free again, and unload completes.
    let next = manager
        .generate(ModelRole::Small, request(8), CancellationToken::new())
        .expect("slot should be free after cancellation");
    drop(next);
    manager.unload(ModelRole::Small).await.expect("unload");
    assert_eq!(manager.status(ModelRole::Small), ModelState::Unloaded);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the LIQUID_INFERENCE_OMNI_* variables; see the file header"]
async fn omni_answers_images_transcribes_audio_and_speaks() {
    let gguf = env_path("LIQUID_INFERENCE_OMNI_GGUF");
    let projector = env_path("LIQUID_INFERENCE_MMPROJ");
    let wav = std::fs::read(env_path("LIQUID_INFERENCE_WAV")).expect("read wav");
    let manager = InferenceManager::new(Arc::new(MistralEngine));
    manager
        .load(spec(ModelRole::Omni, gguf, Some(projector)))
        .await
        .expect("omni load");
    let caps = manager.capabilities(ModelRole::Omni).unwrap();
    assert!(caps.vision && caps.asr && caps.tts && caps.voice_streaming);

    // Image question: a solid red square.
    let mut image_message = ChatMessage::text(
        "user",
        "What single color is this image? Answer with one word.",
    );
    let mut png = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(RgbImage::from_pixel(64, 64, Rgb([255, 0, 0])))
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    image_message.images.push(png.into_inner());
    let stream = manager
        .generate(
            ModelRole::Omni,
            ChatRequest::new(vec![image_message], 16),
            CancellationToken::new(),
        )
        .expect("image generate");
    let (answer, _, _, _, _) = collect(stream).await;
    assert!(
        answer.to_lowercase().contains("red"),
        "image answer was {answer:?}"
    );

    // ASR: transcribe the clip.
    let mut audio_message = ChatMessage::text(
        "user",
        "Transcribe this audio exactly. Output only the transcript.",
    );
    audio_message.audio_wav.push(wav);
    let stream = manager
        .generate(
            ModelRole::Omni,
            ChatRequest::new(vec![audio_message], 256),
            CancellationToken::new(),
        )
        .expect("asr generate");
    let (transcript, _, _, _, _) = collect(stream).await;
    assert!(!transcript.trim().is_empty(), "asr produced no text");
    eprintln!("ASR_TEXT {transcript:?}");

    // Speech: streamed PCM16 at 24 kHz with a text answer.
    let mut speak = ChatRequest::new(
        vec![ChatMessage::text(
            "user",
            "Say the following sentence out loud: The weather is clear today.",
        )],
        256,
    );
    speak.speech = true;
    let stream = manager
        .generate(ModelRole::Omni, speak, CancellationToken::new())
        .expect("speech generate");
    let (text, _, audio, _, _) = collect(stream).await;
    assert!(!audio.is_empty(), "no speech audio was produced");
    assert!(audio.len() % 2 == 0, "PCM16 must have an even byte count");
    eprintln!("SPEECH_TEXT {text:?} AUDIO_BYTES {}", audio.len());

    manager.unload(ModelRole::Omni).await.expect("unload");
}

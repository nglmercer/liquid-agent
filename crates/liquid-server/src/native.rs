//! The in-process inference runtime (mistral.rs): loads the models and runs every model request.
//!
//! The server keeps its prompts, tools, approvals and storage. Only the transport changes: chat and media
//! requests become engine requests here, and loaded models are owned by `InferenceManager`. No server
//! URL, port, or child process is involved.

use std::{
    path::Path,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context};
use axum::{http::StatusCode, response::Response};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::StreamExt;
use liquid_inference::{
    fixture::{FixtureConfig, FixtureEngine},
    Capabilities, ChatMessage, ChatRequest, DevicePreference, Engine, EventStream, InferenceError,
    InferenceEvent, InferenceManager, LoadSpec, MistralEngine, ModelRole, ModelState,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::{
    backends::{projector_name, safe_model},
    json_response, App,
};

/// The one in-process runtime for the app. Models stay loaded here between requests.
static MANAGER: OnceLock<InferenceManager> = OnceLock::new();

/// Scripted replies for the fixture engine, from the JSON file named by `LIQUID_AGENT_NATIVE_SCRIPT`:
/// `[{"when": "text", "reply": "text", "delayMs": 0}]`. Empty when the variable is not set.
fn fixture_rules() -> Vec<liquid_inference::fixture::ScriptRule> {
    let Some(path) = std::env::var_os("LIQUID_AGENT_NATIVE_SCRIPT") else {
        return Vec::new();
    };
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let items: Vec<Value> = serde_json::from_str(&text).unwrap_or_default();
    items
        .iter()
        .map(|item| liquid_inference::fixture::ScriptRule {
            when: item["when"].as_str().unwrap_or("").to_owned(),
            reply: item["reply"].as_str().unwrap_or("").to_owned(),
            delay: Duration::from_millis(item["delayMs"].as_u64().unwrap_or(0)),
        })
        .collect()
}

pub(crate) fn manager() -> &'static InferenceManager {
    MANAGER.get_or_init(|| {
        // Test seam: `LIQUID_AGENT_NATIVE_ENGINE=fixture` replaces the SDK with a scripted engine, so the server's
        // native request paths run without model weights. Production never sets it.
        let engine: Arc<dyn Engine> =
            if std::env::var("LIQUID_AGENT_NATIVE_ENGINE").as_deref() == Ok("fixture") {
                Arc::new(FixtureEngine::new(FixtureConfig {
                    capabilities: Capabilities {
                        text: true,
                        vision: true,
                        asr: true,
                        tts: true,
                        ..Capabilities::default()
                    },
                    tokens: vec!["Hello".into(), " world".into()],
                    scripted: true,
                    rules: fixture_rules(),
                    ..FixtureConfig::default()
                }))
            } else {
                Arc::new(MistralEngine)
            };
        InferenceManager::new(engine)
    })
}

pub(crate) fn role_of(id: &str) -> Option<ModelRole> {
    match id {
        "small" => Some(ModelRole::Small),
        "omni" => Some(ModelRole::Omni),
        _ => None,
    }
}

/// Quantizations the built-in runtime has been run with on the real files. Others are refused with a clear
/// message instead of being assumed to work. Projectors: Q4_K_M and Q8_0 were both run with Omni Q4_K_M.
pub(crate) const VERIFIED_QUANTS: &[&str] = &["Q4_K_M"];

/// Settings mapped to a load request: the model files, the context, and how many generations may run at once
/// (`omniSlots: 2` allows two). The device is chosen automatically (a GPU the build supports, else the CPU) unless
/// `LIQUID_AGENT_DEVICE` names one.
pub(crate) fn load_spec(data: &Path, settings: &Value, id: &str) -> Result<LoadSpec, String> {
    let role = role_of(id).ok_or_else(|| "Not a model.".to_owned())?;
    let (weights, custom, context, _template) =
        safe_model(data, settings, id).ok_or_else(|| "The model is not installed.".to_owned())?;
    if !custom {
        let key = if role == ModelRole::Small {
            "smallQuant"
        } else {
            "omniQuant"
        };
        let quant = settings[key].as_str().unwrap_or("");
        if !VERIFIED_QUANTS.contains(&quant) {
            return Err(format!(
                "The built-in runtime has been verified with {} files only. {quant} is not verified yet; choose {} in Settings.",
                VERIFIED_QUANTS.join(" and "),
                VERIFIED_QUANTS[0]
            ));
        }
    }
    let projector = match role {
        ModelRole::Omni => {
            let name =
                projector_name(settings).ok_or_else(|| "Invalid projector size.".to_owned())?;
            Some(data.join("models").join(name))
        }
        ModelRole::Small => None,
    };
    let max_concurrent = if role == ModelRole::Omni && settings["omniSlots"] == 2 {
        2
    } else {
        1
    };
    let device = device_preference(std::env::var("LIQUID_AGENT_DEVICE").ok().as_deref())?;
    Ok(LoadSpec {
        role,
        weights,
        projector,
        context_tokens: context,
        max_concurrent,
        device,
    })
}

/// The device setting: `auto` (the default), `cpu`, `cuda` or `metal`.
pub(crate) fn device_preference(value: Option<&str>) -> Result<DevicePreference, String> {
    match value {
        None => Ok(DevicePreference::Auto),
        Some(value) => DevicePreference::parse(value).ok_or_else(|| {
            format!("LIQUID_AGENT_DEVICE is \"{value}\"; use auto, cpu, cuda or metal.")
        }),
    }
}

/// Memory a load should have free before it starts: about three times the size of its files, plus a gigabyte
/// for activations. Measured: Omni's 1.9 GB of files peaked near 5.8 GB resident on CPU.
pub(crate) fn memory_needed(spec: &LoadSpec) -> u64 {
    let files: u64 = [Some(&spec.weights), spec.projector.as_ref()]
        .into_iter()
        .flatten()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|meta| meta.len())
        .sum();
    files.saturating_mul(3).saturating_add(1 << 30)
}

/// Refuses a load that would not fit in the memory that is free now, with the numbers, instead of loading into
/// swap or crashing. Unknown free memory does not block a load.
pub(crate) fn check_memory(spec: &LoadSpec) -> Result<(), String> {
    let Some(available) = crate::backends::system_memory()["availableBytes"].as_u64() else {
        return Ok(());
    };
    let needed = memory_needed(spec);
    if available >= needed {
        return Ok(());
    }
    let gb = |bytes: u64| format!("{:.1}", bytes as f64 / (1u64 << 30) as f64);
    Err(format!(
        "Not enough free memory to load this model: it needs about {} GB free and {} GB is free. Close other apps, or choose a smaller model.",
        gb(needed),
        gb(available)
    ))
}

pub(crate) fn caps_json(caps: Capabilities) -> Value {
    json!({
        "text": caps.text,
        "vision": caps.vision,
        "asr": caps.asr,
        "tts": caps.tts,
        "voiceStreaming": caps.voice_streaming,
        "realtimeFullDuplex": caps.realtime_full_duplex,
        "voiceSelection": caps.voice_selection,
        "speakerReference": caps.speaker_reference,
    })
}

/// The file's quantization, from its name: the part after the last dash, as the settings use it.
fn quant_of(spec: &LoadSpec) -> String {
    let stem = spec
        .weights
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    stem.rsplit_once('-')
        .map(|(_, q)| q.to_owned())
        .unwrap_or_default()
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Starts a model in the background. The UI follows the state the way it did for the child process.
pub(crate) async fn start(app: &App, id: &str) -> Response {
    let Some(role) = role_of(id) else {
        return json_response(StatusCode::NOT_FOUND, json!({"error":"Not found."}));
    };
    let settings = app.settings.lock().unwrap().clone();
    let spec = match load_spec(&app.data, &settings, id) {
        Ok(spec) => spec,
        Err(message) => {
            app.backends.set(
                app,
                id,
                json!({"status":"failed","error":message,"runtime":"native"}),
            );
            return json_response(StatusCode::CONFLICT, json!({"error": message}));
        }
    };
    if matches!(
        manager().status(role),
        ModelState::Ready | ModelState::Busy | ModelState::Loading
    ) {
        return json_response(StatusCode::OK, json!({"ok":true}));
    }
    if let Err(message) = check_memory(&spec) {
        app.backends.set(
            app,
            id,
            json!({"status":"failed","error":message,"runtime":"native"}),
        );
        return json_response(StatusCode::CONFLICT, json!({"error": message}));
    }
    app.backends.set(
        app,
        id,
        json!({"status":"starting","text":"Waking up…","runtime":"native"}),
    );
    let app = app.clone();
    let id = id.to_owned();
    tokio::spawn(async move {
        match manager().load(spec.clone()).await {
            Ok(()) => publish_running(&app, &id, role, &spec, false),
            Err(err) => app.backends.set(&app, &id, failed(&err)),
        }
    });
    json_response(StatusCode::OK, json!({"ok":true}))
}

fn failed(err: &InferenceError) -> Value {
    json!({"status":"failed","error": format!("The model could not be loaded: {err}"),"runtime":"native"})
}

fn publish_running(app: &App, id: &str, role: ModelRole, spec: &LoadSpec, sleeping: bool) {
    let caps = manager().capabilities(role).unwrap_or_default();
    app.backends
        .contexts
        .lock()
        .unwrap()
        .insert(id.to_owned(), spec.context_tokens);
    app.backends
        .slots
        .lock()
        .unwrap()
        .insert(id.to_owned(), spec.max_concurrent);
    app.backends.set(
        app,
        id,
        json!({
            "status":"running",
            "runtime":"native",
            "sleeping": sleeping,
            "capabilities": caps_json(caps),
            "context": spec.context_tokens,
            "slots": spec.max_concurrent,
            "quant": quant_of(spec),
            "projector": spec.projector.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()),
            "loadedAt": now_millis(),
        }),
    );
}

/// Unloads a model the user stopped. Waits for a load in progress, then for the active generations.
pub(crate) async fn stop(app: &App, id: &str) -> Response {
    let Some(role) = role_of(id) else {
        return json_response(StatusCode::NOT_FOUND, json!({"error":"Not found."}));
    };
    wait_while(role, ModelState::Loading).await;
    if let Err(err) = manager().unload(role).await {
        return json_response(StatusCode::CONFLICT, json!({"error": err.to_string()}));
    }
    app.backends.contexts.lock().unwrap().remove(id);
    app.backends.slots.lock().unwrap().remove(id);
    let settings = app.settings.lock().unwrap().clone();
    let state = if crate::backends::installed(&app.data, &settings, id) {
        "installed"
    } else {
        "missing"
    };
    app.backends
        .set(app, id, json!({"status": state, "runtime":"native"}));
    json_response(StatusCode::OK, json!({"ok":true}))
}

/// Frees memory after the idle period. The state stays `running`, so the next request wakes the model.
async fn sleep_idle(app: &App, id: &str, role: ModelRole) {
    // Read the capabilities first: they are not available once the model is unloaded.
    let caps = manager().capabilities(role).unwrap_or_default();
    if manager().unload(role).await.is_err() {
        return;
    }
    let settings = app.settings.lock().unwrap().clone();
    if let Ok(spec) = load_spec(&app.data, &settings, id) {
        app.backends.set(
            app,
            id,
            json!({
                "status":"running",
                "runtime":"native",
                "sleeping": true,
                "capabilities": caps_json(caps),
                "context": spec.context_tokens,
                "slots": spec.max_concurrent,
            }),
        );
    }
}

/// Makes sure a model is loaded before a request uses it. Waking a sleeping model takes a few seconds.
pub(crate) async fn ensure_loaded(app: &App, id: &str) -> anyhow::Result<()> {
    let role = role_of(id).ok_or_else(|| anyhow!("Not a model."))?;
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        match manager().status(role) {
            ModelState::Ready | ModelState::Busy => return Ok(()),
            ModelState::Loading | ModelState::Unloading => {}
            ModelState::Unloaded | ModelState::Failed(_) => {
                let settings = app.settings.lock().unwrap().clone();
                let spec = load_spec(&app.data, &settings, id).map_err(|m| anyhow!(m))?;
                if let Err(message) = check_memory(&spec) {
                    app.backends.set(
                        app,
                        id,
                        json!({"status":"failed","error":message,"runtime":"native"}),
                    );
                    bail!("{message}");
                }
                match manager().load(spec.clone()).await {
                    Ok(()) => {
                        publish_running(app, id, role, &spec, false);
                        return Ok(());
                    }
                    Err(InferenceError::InvalidTransition { .. }) => {}
                    Err(err) => {
                        app.backends.set(app, id, failed(&err));
                        bail!("The model could not be loaded: {err}");
                    }
                }
            }
        }
        if Instant::now() > deadline {
            bail!("The model is still loading. Try again shortly.");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn wait_while(role: ModelRole, state: ModelState) {
    let deadline = Instant::now() + Duration::from_secs(180);
    while manager().status(role) == state && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Starts a generation. The caller holds the stream until it ends; dropping it or cancelling `cancel`
/// stops the engine request.
pub(crate) fn generate(
    role: ModelRole,
    request: ChatRequest,
    cancel: CancellationToken,
) -> anyhow::Result<EventStream> {
    manager()
        .generate(role, request, cancel)
        .map_err(|err| anyhow!(err.to_string()))
}

/// Converts the chat wire format (OpenAI-style messages) into an engine request. Images arrive as data URLs
/// and audio as base64 WAV; both are decoded here, so the engine never sees JSON.
pub(crate) fn request_from_wire(
    messages: &[Value],
    max_output_tokens: u32,
    temperature: f32,
    stop: &[&str],
    enable_thinking: bool,
    speech: bool,
) -> anyhow::Result<ChatRequest> {
    log_request(
        messages,
        max_output_tokens,
        temperature,
        stop,
        enable_thinking,
        speech,
    );
    let mut converted = Vec::with_capacity(messages.len());
    for message in messages {
        let role = message["role"]
            .as_str()
            .ok_or_else(|| anyhow!("A message has no role."))?;
        let mut item = ChatMessage::text(role, "");
        match &message["content"] {
            Value::String(text) => item.content = text.clone(),
            Value::Array(parts) => {
                let mut texts = Vec::new();
                for part in parts {
                    match part["type"].as_str() {
                        Some("text") => {
                            if let Some(text) = part["text"].as_str() {
                                texts.push(text.to_owned());
                            }
                        }
                        Some("image_url") => {
                            let url = part["image_url"]["url"].as_str().unwrap_or("");
                            item.images.push(decode_data_url(url)?);
                        }
                        Some("input_audio") => {
                            let data = part["input_audio"]["data"].as_str().unwrap_or("");
                            let format = part["input_audio"]["format"].as_str().unwrap_or("wav");
                            if format != "wav" {
                                bail!("Audio must be WAV.");
                            }
                            item.audio_wav
                                .push(BASE64.decode(data).context("Bad audio data.")?);
                        }
                        _ => {}
                    }
                }
                item.content = texts.join("\n");
            }
            _ => {}
        }
        converted.push(item);
    }
    let mut request = ChatRequest::new(converted, max_output_tokens);
    request.temperature = temperature;
    request.stop = stop.iter().map(|s| (*s).to_owned()).collect();
    request.enable_thinking = enable_thinking;
    request.speech = speech;
    Ok(request)
}

/// Test seam: with `LIQUID_AGENT_NATIVE_REQUEST_LOG` set, every request is appended to that file as one JSON line,
/// in the shape the model received it. Production never sets it.
fn log_request(
    messages: &[Value],
    max_output_tokens: u32,
    temperature: f32,
    stop: &[&str],
    enable_thinking: bool,
    speech: bool,
) {
    let Some(path) = std::env::var_os("LIQUID_AGENT_NATIVE_REQUEST_LOG") else {
        return;
    };
    let line = json!({
        "stream": true,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "stop": stop,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "speech": speech,
    });
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{line}");
    }
}

fn decode_data_url(url: &str) -> anyhow::Result<Vec<u8>> {
    let (header, payload) = url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(','))
        .ok_or_else(|| anyhow!("An image is not a data URL."))?;
    if !header.ends_with(";base64") {
        bail!("An image must be base64 encoded.");
    }
    BASE64.decode(payload).context("Bad image data.")
}

/// Speaks `text` with Omni and returns a 24 kHz mono 16-bit WAV. Returns the PCM and the spoken text.
pub(crate) async fn speak(
    text: &str,
    cancel: CancellationToken,
) -> anyhow::Result<(Vec<u8>, String)> {
    let mut request = ChatRequest::new(
        vec![ChatMessage::text(
            "user",
            format!("Say the following sentence out loud: {text}"),
        )],
        1536,
    );
    request.speech = true;
    request.temperature = 0.0;
    let mut stream = generate(ModelRole::Omni, request, cancel)?;
    let mut pcm = Vec::new();
    let mut spoken = String::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(InferenceEvent::AudioChunk { pcm16, .. }) => pcm.extend(pcm16),
            Ok(InferenceEvent::TextDelta(delta)) => spoken.push_str(&delta),
            Ok(_) => {}
            Err(InferenceError::Cancelled) => bail!("Stopped."),
            Err(err) => bail!("{err}"),
        }
    }
    if pcm.is_empty() {
        bail!("The model produced no speech.");
    }
    Ok((pcm, spoken))
}

/// Collects a non-streaming response: the text, the usage, and any error.
pub(crate) async fn collect_text(mut stream: EventStream) -> anyhow::Result<(String, u64)> {
    let mut text = String::new();
    let mut tokens = 0;
    while let Some(event) = stream.next().await {
        match event {
            Ok(InferenceEvent::TextDelta(delta)) => text.push_str(&delta),
            Ok(InferenceEvent::Usage { output_tokens, .. }) => tokens = output_tokens,
            Ok(_) => {}
            Err(InferenceError::Cancelled) => bail!("Stopped."),
            Err(err) => bail!("{err}"),
        }
    }
    Ok((text, tokens))
}

/// Unloads models that have been idle longer than the sleep setting. Checked every 30 seconds.
pub(crate) fn spawn_idle_watch(app: App) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            let settings = app.settings.lock().unwrap().clone();
            let minutes = match settings["idleSleepMinutes"].as_u64() {
                Some(m @ (5 | 15 | 30)) => m,
                _ => continue,
            };
            for (id, role) in [("small", ModelRole::Small), ("omni", ModelRole::Omni)] {
                if manager().status(role) != ModelState::Ready {
                    continue;
                }
                let idle = manager()
                    .last_used(role)
                    .is_some_and(|at| at.elapsed() > Duration::from_secs(minutes * 60));
                if idle {
                    sleep_idle(&app, id, role).await;
                }
            }
        }
    });
}

/// Per-model engine facts for diagnostics. Unknown values are null, never zero.
pub(crate) fn diagnostics() -> Value {
    let mut out = serde_json::Map::new();
    for (id, role) in [("small", ModelRole::Small), ("omni", ModelRole::Omni)] {
        let state = match manager().status(role) {
            ModelState::Ready => "ready",
            ModelState::Busy => "busy",
            ModelState::Loading => "loading",
            ModelState::Unloading => "unloading",
            ModelState::Unloaded => "unloaded",
            ModelState::Failed(_) => "failed",
        };
        let caps = manager().capabilities(role).ok().map(caps_json);
        out.insert(
            id.to_owned(),
            json!({
                "state": state,
                "capabilities": caps,
                "lastUsedSecondsAgo": manager().last_used(role).map(|at| at.elapsed().as_secs()),
                "device": manager().device(role).ok().map(|device| device.as_str()),
                "gpuMemoryBytes": Value::Null,
            }),
        );
    }
    Value::Object(out)
}

/// Devices automatic selection can use on this machine, best first; the CPU is always last.
pub(crate) fn available_devices() -> Value {
    json!(liquid_inference::available_devices()
        .into_iter()
        .map(|device| device.as_str())
        .collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_messages_become_engine_messages() {
        let messages = vec![
            json!({"role":"system","content":"Be brief."}),
            json!({"role":"user","content":[
                {"type":"text","text":"What is this?"},
                {"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}", BASE64.encode([1u8,2,3]))}},
                {"type":"input_audio","input_audio":{"data":BASE64.encode([4u8,5]),"format":"wav"}},
            ]}),
        ];
        let request =
            request_from_wire(&messages, 64, 0.3, &["</tool_call>"], false, false).unwrap();
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].content, "Be brief.");
        assert_eq!(request.messages[1].content, "What is this?");
        assert_eq!(request.messages[1].images, vec![vec![1, 2, 3]]);
        assert_eq!(request.messages[1].audio_wav, vec![vec![4, 5]]);
        assert_eq!(request.stop, vec!["</tool_call>".to_owned()]);
        assert!((request.temperature - 0.3).abs() < f32::EPSILON);
        assert!(!request.enable_thinking && !request.speech);
    }

    #[test]
    fn non_wav_audio_and_plain_image_urls_are_refused() {
        let audio = json!([{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA==","format":"mp3"}}]}]);
        assert!(request_from_wire(audio.as_array().unwrap(), 8, 0.0, &[], false, false).is_err());
        assert!(decode_data_url("https://example.com/a.png").is_err());
        assert!(decode_data_url("data:image/png,raw").is_err());
        assert_eq!(
            decode_data_url("data:image/png;base64,AQID").unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn memory_rule_is_three_times_the_files_plus_a_gigabyte() {
        let dir = std::env::temp_dir().join(format!("liquid-native-mem-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights = dir.join("w.gguf");
        std::fs::write(&weights, vec![0u8; 1000]).unwrap();
        let spec = LoadSpec {
            role: ModelRole::Small,
            weights: weights.clone(),
            projector: None,
            context_tokens: 4096,
            max_concurrent: 1,
            device: DevicePreference::Auto,
        };
        assert_eq!(memory_needed(&spec), 3000 + (1 << 30));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn device_setting_defaults_to_auto_and_refuses_unknown_names() {
        assert_eq!(device_preference(None), Ok(DevicePreference::Auto));
        assert_eq!(
            device_preference(Some(" CUDA ")),
            Ok(DevicePreference::Cuda)
        );
        assert_eq!(device_preference(Some("cpu")), Ok(DevicePreference::Cpu));
        assert_eq!(
            device_preference(Some("metal")),
            Ok(DevicePreference::Metal)
        );
        let err = device_preference(Some("vulkan")).unwrap_err();
        assert!(err.contains("auto, cpu, cuda or metal"), "{err}");
    }

    #[test]
    fn unverified_quantizations_are_refused_with_the_choice_to_make() {
        let data = std::env::temp_dir().join(format!("liquid-native-quant-{}", std::process::id()));
        let models = data.join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("LFM3-1.7B-Q8_0.gguf"), b"w").unwrap();
        let err = load_spec(&data, &json!({"smallQuant":"Q8_0"}), "small").unwrap_err();
        assert!(err.contains("Q8_0 is not verified yet"), "{err}");
        let _ = std::fs::remove_dir_all(&data);
    }

    #[test]
    fn load_spec_maps_the_model_settings() {
        let data = std::env::temp_dir().join(format!("liquid-native-spec-{}", std::process::id()));
        let models = data.join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("LFM3-Omni-2B-Q4_K_M.gguf"), b"w").unwrap();
        std::fs::write(models.join("mmproj-LFM3-Omni-2B-Q4_K_M.gguf"), b"p").unwrap();
        let settings = json!({"omniQuant":"Q4_K_M","omniProjector":"Q4_K_M","omniSlots":2,"modelContext":8192});
        let spec = load_spec(&data, &settings, "omni").unwrap();
        assert_eq!(spec.role, ModelRole::Omni);
        assert_eq!(
            spec.projector,
            Some(models.join("mmproj-LFM3-Omni-2B-Q4_K_M.gguf"))
        );
        assert_eq!(spec.device, DevicePreference::Auto);
        assert_eq!(spec.max_concurrent, 2);
        assert_eq!(spec.context_tokens, 8192);
        let auto = load_spec(
            &data,
            &json!({"omniQuant":"Q4_K_M","omniProjector":"Q4_K_M"}),
            "omni",
        )
        .unwrap();
        assert_eq!(auto.device, DevicePreference::Auto);
        assert_eq!(auto.max_concurrent, 1);
        let _ = std::fs::remove_dir_all(&data);
    }
}

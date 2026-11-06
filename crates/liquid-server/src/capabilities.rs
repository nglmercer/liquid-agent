//! Explicit, limited checks against the in-process multimodal model.
use crate::{json_response, media, App};
use axum::{http::StatusCode, response::Response};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde_json::json;
use std::time::Duration;

pub(crate) async fn probe(app: &App) -> Response {
    if app.omni_slots_busy() {
        return json_response(
            StatusCode::CONFLICT,
            json!({"error":"Omni is busy with a voice call. End the call or configure another Omni slot."}),
        );
    }
    if !app.backends.running("omni") {
        return json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"Eyes and ears must be running for this check."}),
        );
    }
    let Some(_model_use) = app.backends.begin_model_use() else {
        return json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"The model is changing. Try again shortly."}),
        );
    };
    if !app.backends.running("omni") {
        return json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"error":"Eyes and ears is not running."}),
        );
    }
    native_probe().await
}

/// A 1x1 PNG. The vision probe only checks that the model answers about an image.
const PROBE_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

/// Native probes: each capability is checked by running it once. Voice selection, speaker cloning and full-duplex
/// audio are reported as unsupported, because the engine has no such control.
async fn native_probe() -> Response {
    let speech = tokio::time::timeout(Duration::from_secs(120), media::synthesize("Hola.")).await;
    let (speech_ok, heard) = match speech {
        Ok(Ok(wav)) => {
            let ok = media::wav_format(&wav).is_some_and(|format| {
                format.encoding == 1
                    && format.channels == 1
                    && format.bits == 16
                    && format.rate > 0
                    && !format.data.is_empty()
                    && format.data.len() % 2 == 0
            });
            let heard = if ok {
                tokio::time::timeout(Duration::from_secs(120), media::completion(&wav, "es"))
                    .await
                    .ok()
                    .and_then(Result::ok)
                    .unwrap_or_default()
            } else {
                String::new()
            };
            (ok, heard)
        }
        _ => (false, String::new()),
    };
    let image = BASE64.decode(PROBE_PNG_B64).unwrap_or_default();
    let vision_ok = tokio::time::timeout(
        Duration::from_secs(120),
        media::vision(&image, "image/png", "What do you see? One word."),
    )
    .await
    .is_ok_and(|answer| answer.is_ok_and(|text| !text.trim().is_empty()));
    json_response(
        StatusCode::OK,
        json!({
            "runtime": "native",
            "spanishWav": speech_ok,
            "asr": !heard.trim().is_empty(),
            "vision": vision_ok,
            "speech": speech_ok,
            "voiceSelection": false,
            "speakerReference": false,
            "realtimeFullDuplex": false,
        }),
    )
}

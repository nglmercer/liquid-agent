//! Voice calls on the in-process model: each turn is heard, answered, and spoken sentence by sentence.
//! Turns are sent by the browser, which detects the end of each utterance; this module transcribes, replies
//! through the normal chat, and streams the speech back as PCM16 chunks.
use crate::{allowed, json_response, media, store, App};
use anyhow::Context;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{header, HeaderMap, StatusCode, Uri},
    response::Response,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde_json::{json, Value};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub async fn websocket(
    State(app): State<App>,
    headers: HeaderMap,
    uri: Uri,
    ws: WebSocketUpgrade,
) -> Response {
    if !headers.contains_key(header::ORIGIN) || !allowed(&headers, &uri, &app, true) {
        return json_response(StatusCode::UNAUTHORIZED, json!({"error":"Not allowed."}));
    }
    ws.max_message_size(4 * 1024 * 1024)
        .on_upgrade(move |socket| run(app, socket))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn make_chat(app: &App, requested: Option<&str>) -> anyhow::Result<String> {
    let mut chats = app.chats.lock().unwrap();
    if let Some(chat) = requested.and_then(|id| chats.iter().find(|c| c["id"] == id)) {
        return Ok(chat["id"].as_str().unwrap().to_owned());
    }
    let id = store::id();
    let stamp = now();
    let chat = json!({"id":id,"title":"Voice chat","source":"voice","named":true,"createdAt":stamp,"updatedAt":stamp,"pinned":false,"messages":[]});
    chats.insert(0, chat.clone());
    if let Err(e) = store::save_array(&app.data, "chats.json", &chats) {
        chats.remove(0);
        return Err(e.into());
    }
    drop(chats);
    app.emit(json!({"type":"chat","chat":chat}));
    Ok(id)
}

async fn send(socket: &mut WebSocket, value: Value) -> bool {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .is_ok()
}

#[cfg(test)]
mod sentence_tests {
    use super::sentence_cut;

    #[test]
    fn cuts_after_the_first_sentence_that_is_long_enough() {
        let text =
            "This first sentence is long enough to be spoken on its own. And a second one follows";
        let cut = sentence_cut(text).unwrap();
        assert_eq!(
            &text[..cut],
            "This first sentence is long enough to be spoken on its own."
        );
    }

    #[test]
    fn waits_for_a_complete_sentence() {
        assert_eq!(sentence_cut("Yes. Still going without a stop"), None);
        assert_eq!(sentence_cut("no terminator at all, just words"), None);
    }

    #[test]
    fn handles_multibyte_punctuation() {
        let sentence = "这是一个足够长的句子。";
        let text = format!("{}{}", sentence.repeat(4), "然后是下一句");
        let cut = sentence_cut(&text).unwrap();
        assert_eq!(&text[..cut], sentence.repeat(4));
    }
}

#[cfg(test)]
mod tests {
    use super::is_jpeg_base64;

    #[test]
    fn recognises_a_jpeg_from_its_first_base64_group() {
        assert!(is_jpeg_base64("/9j/4AAQSkZJRg=="));
        assert!(!is_jpeg_base64("iVBORw0KGgo="));
    }
}

/// A JPEG starts with FF D8 FF. The first base64 group is enough to tell, so a whole frame isn't decoded to check it.
fn is_jpeg_base64(image: &str) -> bool {
    image
        .get(..4)
        .and_then(|head| BASE64.decode(head).ok())
        .is_some_and(|bytes| bytes.starts_with(&[0xff, 0xd8]))
}

async fn describe(image: String) -> anyhow::Result<String> {
    let bytes = BASE64
        .decode(image.as_bytes())
        .context("Bad camera frame.")?;
    media::vision(
        &bytes,
        "image/jpeg",
        "This is a frame from the user's camera during a voice call. In one or two short sentences, describe what it shows.",
    )
    .await
}

/// Counts a live voice call, so an update waits until nobody is talking to the agent.
struct CallGuard(App);

impl Drop for CallGuard {
    fn drop(&mut self) {
        self.0
            .voice_calls
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

async fn run(app: App, mut socket: WebSocket) {
    app.voice_calls
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let _call = CallGuard(app.clone());
    let first = tokio::time::timeout(Duration::from_secs(30), socket.recv()).await;
    let Some(Ok(Message::Text(first))) = first.ok().flatten() else {
        return;
    };
    let Ok(start) = serde_json::from_str::<Value>(&first) else {
        return;
    };
    if start["type"] != "start" {
        return;
    }
    if !crate::backends::demand_omni(&app).await {
        send(&mut socket,json!({"type":"error","message":"Voice chat needs Eyes and ears. Set it up in Settings → Downloads."})).await;
        return;
    }
    let Some(model_use) = app.backends.begin_model_use() else {
        send(
            &mut socket,
            json!({"type":"error","message":"The model is changing. Try again shortly."}),
        )
        .await;
        return;
    };
    if !app.backends.running("omni") {
        send(
            &mut socket,
            json!({"type":"error","message":"Eyes and ears is not running."}),
        )
        .await;
        return;
    }
    if app.voice_calls.load(std::sync::atomic::Ordering::SeqCst)
        > app.backends.running_slots("omni").unwrap_or(1)
    {
        send(
            &mut socket,
            json!({"type":"error","message":"All Omni slots are occupied by voice calls."}),
        )
        .await;
        return;
    }
    let settings = app.settings.lock().unwrap().clone();
    high_quality(app.clone(), socket, &start, settings, model_use).await;
}

fn remove_empty_voice_chat(app: &App, chat: &str) {
    let mut chats = app.chats.lock().unwrap();
    if chats.iter().any(|c| {
        c["id"] == chat
            && c["source"] == "voice"
            && c["messages"].as_array().is_some_and(|v| v.is_empty())
    }) {
        let mut next = chats.clone();
        next.retain(|c| c["id"] != chat);
        if store::save_array(&app.data, "chats.json", &next).is_ok() {
            *chats = next;
            app.emit(json!({"type":"chat.deleted","id":chat}));
        }
    }
}

struct VoiceChatGuard(App, String);

impl Drop for VoiceChatGuard {
    fn drop(&mut self) {
        self.0.voice_chats.lock().unwrap().remove(&self.1);
    }
}

/// Runs each stage in order: the call reserves its Omni slot for each turn, so ASR, chat and speech share it.
async fn high_quality(
    app: App,
    mut socket: WebSocket,
    start: &Value,
    settings: Value,
    _model_use: std::sync::Arc<crate::backends::ModelUseLease>,
) {
    let Ok(chat) = make_chat(&app, start["chatId"].as_str()) else {
        return;
    };
    let reserved = {
        let mut voice_chats = app.voice_chats.lock().unwrap();
        let running = app.running.lock().unwrap();
        !running.contains_key(&chat) && voice_chats.insert(chat.clone())
    };
    if !reserved {
        send(
            &mut socket,
            json!({"type":"error","message":"That chat is already active."}),
        )
        .await;
        return;
    }
    let _reservation = VoiceChatGuard(app.clone(), chat.clone());
    let language = liquid_core::i18n::call_language(&settings).to_owned();
    let asr = liquid_core::i18n::call_asr_language(&settings).to_owned();
    // Turns are sent by the browser and answered one at a time, so the call is turn-based by design.
    send(&mut socket, json!({"type":"ready","inputRate":16000,"outputRate":24000,"cameraAvailable":true,"chatId":chat,"configurationConfirmed":true,"voiceSupported":true,"effectiveLanguage":language,"turnBased":true,"realtimeFullDuplex":false})).await;
    send(&mut socket, json!({"type":"state","state":"listening"})).await;
    let (commands, mut command_rx) =
        mpsc::unbounded_channel::<(u64, Vec<u8>, String, Option<String>, CancellationToken)>();
    let (tx, mut rx) = mpsc::unbounded_channel::<(u64, Value, Option<Vec<u8>>)>();
    let worker_app = app.clone();
    let worker_chat = chat.clone();
    let worker = tokio::spawn(async move {
        while let Some((id, pcm, typed, frame, cancel)) = command_rx.recv().await {
            if cancel.is_cancelled() {
                continue;
            }
            let result: anyhow::Result<()> = async {
                let heard = if !typed.is_empty() { typed } else {
                    let wav = media::pcm16_wav(&pcm, 16000);
                    tokio::select! { _=cancel.cancelled()=>return Ok(()), result=media::completion(&wav, &asr)=>result? }
                };
                if heard.trim().is_empty() { anyhow::bail!("Could not hear any speech."); }
                let _ = tx.send((id, json!({"type":"user","text":heard}), None));
                let mut request = heard.clone();
                if let Some(image) = frame {
                    let description = tokio::select! { _=cancel.cancelled()=>return Ok(()), result=describe(image)=>result.unwrap_or_default() };
                    if !description.is_empty() { request.push_str(&format!("\n\n(On camera right now: {description})")); }
                }
                let mut events = worker_app.events.subscribe();
                let response = crate::chat::send_voice(&worker_app, &worker_chat, &json!({"text":request}), &language);
                if !response.status().is_success() { anyhow::bail!("Could not start the voice reply (HTTP {}).", response.status()); }
                // Each sentence is spoken as soon as it is complete, so speech starts while the reply is still
                // being written. A separate task speaks them in order, and the event loop keeps reading the reply.
                let (pieces_tx, pieces_rx) = mpsc::unbounded_channel::<String>();
                let speaker = tokio::spawn(speak_pieces(
                    id,
                    tx.clone(),
                    cancel.clone(),
                    pieces_rx,
                ));
                // The part of the reply already queued for speech. Visible text only grows, so this is a prefix.
                let mut queued = String::new();
                let answer = loop {
                    tokio::select! {
                        _=cancel.cancelled()=> {
                            crate::chat::stop(&worker_app, &worker_chat);
                            while worker_app.running.lock().unwrap().contains_key(&worker_chat) {
                                tokio::time::sleep(Duration::from_millis(25)).await;
                            }
                            return Ok(());
                        }
                        event=events.recv()=>{
                            let Ok(raw)=event else { anyhow::bail!("Voice reply event stream closed.") };
                            let Ok(event)=serde_json::from_str::<Value>(&raw) else {continue};
                            if event["type"]=="delta" && event["chatId"]==worker_chat {
                                let _=tx.send((id,json!({"type":"assistant","text":event["text"]}),None));
                                if let Some(visible) = event["text"].as_str() {
                                    if let Some(rest) = visible.strip_prefix(queued.as_str()) {
                                        if let Some(cut) = sentence_cut(rest) {
                                            let piece = rest[..cut].to_owned();
                                            queued.push_str(&piece);
                                            let _ = pieces_tx.send(piece);
                                        }
                                    }
                                }
                            }
                            if event["type"]=="chat" && event["chat"]["id"]==worker_chat {
                                if let Some(last)=event["chat"]["messages"].as_array().and_then(|m|m.last()).filter(|m|m["role"]=="assistant") {
                                    if let Some(step)=last["steps"].as_array().and_then(|s|s.last()) {let _=tx.send((id,json!({"type":"tool","label":step["label"],"status":step["status"]}),None));}
                                    match last["status"].as_str() {
                                        Some("done")=>break last["content"].as_str().unwrap_or("").to_owned(),
                                        Some("error")=>anyhow::bail!("{}",last["error"].as_str().unwrap_or("Voice reply failed.")),
                                        Some("stopped")=>return Ok(()), _=>{}
                                    }
                                }
                            }
                        }
                    }
                };
                if answer.is_empty() { anyhow::bail!("Voice reply was empty."); }
                // Whatever the reply has that was not queued yet: the rest after the part already spoken.
                let rest = answer.strip_prefix(queued.as_str()).map(str::to_owned).unwrap_or_else(|| {
                    let common = answer.chars().zip(queued.chars()).take_while(|(a, b)| a == b).map(|(a, _)| a.len_utf8()).sum::<usize>();
                    answer[common..].to_owned()
                });
                for piece in speech_pieces(&rest) {
                    let _ = pieces_tx.send(piece);
                }
                drop(pieces_tx);
                speaker.await.map_err(|err| anyhow::anyhow!("Speech stopped: {err}"))??;
                Ok(())
            }.await;
            if !cancel.is_cancelled() {
                if let Err(error) = result {
                    let _ = tx.send((
                        id,
                        json!({"type":"error","message":error.to_string()}),
                        None,
                    ));
                }
                let _ = tx.send((id, json!({"type":"turn.done"}), None));
                let _ = tx.send((id, json!({"type":"state","state":"listening"}), None));
            }
        }
    });
    let mut heard = Vec::new();
    let mut frame = None;
    let mut sequence = 0u64;
    let mut active = CancellationToken::new();
    loop {
        tokio::select! {
            message=socket.recv()=>{
                let Some(Ok(message))=message else {break};
                match message {
                    Message::Binary(bytes)=>{if heard.len()+bytes.len()<=16000*2*60 {heard.extend_from_slice(&bytes)} else {heard.clear();send(&mut socket,json!({"type":"error","message":"Voice input is too long."})).await;}},
                    Message::Text(raw)=>{
                        let Ok(message)=serde_json::from_str::<Value>(&raw) else {continue};
                        match message["type"].as_str().unwrap_or("") {
                            "stop"=>break,
                            "cancel"=>{active.cancel();heard.clear();sequence+=1;send(&mut socket,json!({"type":"state","state":"listening"})).await;},
                            "frame"=>{frame=message["image"].as_str().filter(|s|s.len()<3_000_000 && is_jpeg_base64(s)).map(str::to_owned);},
                            "text"|"commit"=>{
                                let typed=message["text"].as_str().unwrap_or("").trim().chars().take(8000).collect::<String>();
                                if typed.is_empty() && heard.len()<16000*2*3/10 {heard.clear();continue;}
                                active.cancel();active=CancellationToken::new();sequence+=1;
                                let pcm=std::mem::take(&mut heard);
                                if commands.send((sequence,pcm,typed,frame.take(),active.clone())).is_err(){break;}
                                send(&mut socket,json!({"type":"state","state":"thinking"})).await;
                            }, _=>{}
                        }
                    }, Message::Close(_)=>break, _=>{}
                }
            },
            event=rx.recv()=>{
                let Some((id,event,audio))=event else {break};
                if id!=sequence {continue;}
                if !send(&mut socket,event).await {break;}
                if let Some(audio)=audio {
                    send(&mut socket,json!({"type":"state","state":"speaking"})).await;
                    if socket.send(Message::Binary(audio.into())).await.is_err(){break;}
                }
            }
        }
    }
    active.cancel();
    drop(commands);
    let _ = worker.await;
    remove_empty_voice_chat(&app, &chat);
}

/// Speaks sentences in order. Each one is synthesized and sent as audio, with the sample rate checked.
async fn speak_pieces(
    id: u64,
    tx: mpsc::UnboundedSender<(u64, Value, Option<Vec<u8>>)>,
    cancel: CancellationToken,
    mut pieces: mpsc::UnboundedReceiver<String>,
) -> anyhow::Result<()> {
    let mut rate = None;
    while let Some(piece) = pieces.recv().await {
        let piece = piece.trim().to_owned();
        if piece.is_empty() {
            continue;
        }
        let wav = tokio::select! { _=cancel.cancelled()=>return Ok(()), result=media::synthesize(&piece)=>result? };
        let format = media::wav_format(&wav)
            .ok_or_else(|| anyhow::anyhow!("Speech returned invalid audio."))?;
        anyhow::ensure!(
            format.encoding == 1 && format.channels == 1 && format.bits == 16 && format.rate > 0,
            "Speech returned an unsupported audio format."
        );
        anyhow::ensure!(
            rate.is_none_or(|old| old == format.rate),
            "Speech sample rate changed during the reply."
        );
        let event = if rate.is_none() {
            json!({"type":"audio.format","rate":format.rate})
        } else {
            json!({"type":"audio.chunk"})
        };
        rate = Some(format.rate);
        let _ = tx.send((id, event, Some(format.data.to_vec())));
    }
    Ok(())
}

/// Byte length of the first complete sentence in `text`: up to the first sentence end that comes after at least
/// 40 characters, so fragments such as "Yes." are kept with the sentence that follows them.
fn sentence_cut(text: &str) -> Option<usize> {
    text.char_indices()
        .filter(|(_, c)| matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '\n'))
        .map(|(i, c)| i + c.len_utf8())
        .find(|end| text[..*end].chars().count() >= 40)
}

/// Bound each synthesis request and start playback as soon as the first piece is ready.
fn speech_pieces(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut piece = String::new();
    for word in text.split_whitespace() {
        if !piece.is_empty() && piece.chars().count() + word.chars().count() + 1 > 400 {
            pieces.push(std::mem::take(&mut piece));
        }
        if !piece.is_empty() {
            piece.push(' ');
        }
        piece.push_str(word);
        if piece.chars().count() >= 120 && word.ends_with(['.', '!', '?', '。', '！', '？']) {
            pieces.push(std::mem::take(&mut piece));
        }
    }
    if !piece.is_empty() {
        pieces.push(piece);
    }
    pieces
}

use std::path::PathBuf;

/// The two model roles the app runs. Serialized IDs stay `small` / `omni`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelRole {
    Small,
    Omni,
}

/// Requested device. `Auto` lets the engine choose among supported devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadSpec {
    pub role: ModelRole,
    pub weights: PathBuf,
    pub projector: Option<PathBuf>,
    pub context_tokens: usize,
    /// Maximum generations that may run at the same time for this model.
    pub max_concurrent: usize,
    pub device: DevicePreference,
}

/// What a loaded model can actually do. Reported truthfully; never inferred
/// from settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Capabilities {
    pub text: bool,
    pub vision: bool,
    pub asr: bool,
    pub tts: bool,
    pub voice_streaming: bool,
    pub realtime_full_duplex: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub max_output_tokens: u32,
}

/// Stable events produced by any engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallDelta(String),
    AudioChunk {
        pcm16: Vec<u8>,
        sample_rate: u32,
    },
    Usage {
        prompt_tokens: u64,
        output_tokens: u64,
    },
    Finished,
}

/// Lifecycle state of one model role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    Unloaded,
    Loading,
    Ready,
    Busy,
    Unloading,
    Failed(String),
}

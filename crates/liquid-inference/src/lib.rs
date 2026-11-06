//! In-process inference boundary for Liquid Agent.
//!
//! This crate owns model lifecycle (load/unload/status), leases, and
//! cancellable generation streams. It deliberately has no server URLs and no
//! knowledge of HTTP. Concrete engines implement [`Engine`] and
//! [`LoadedModel`]; the production engine (mistral.rs) is added in a later
//! phase, so the manager is testable with [`fixture::FixtureEngine`].

pub mod error;
pub mod fixture;
pub mod manager;
#[cfg(feature = "mistralrs")]
pub mod mistralrs_adapter;
pub mod registry;
pub mod traits;
pub mod types;

pub use error::InferenceError;
pub use manager::InferenceManager;
#[cfg(feature = "mistralrs")]
pub use mistralrs_adapter::MistralEngine;
pub use traits::{Engine, EventStream, LoadFuture, LoadedModel};
pub use types::{
    Capabilities, ChatMessage, ChatRequest, DevicePreference, InferenceEvent, LoadSpec, ModelRole,
    ModelState, SPEECH_SAMPLE_RATE,
};

use std::{future::Future, pin::Pin, sync::Arc};

use futures_util::stream::BoxStream;
use tokio_util::sync::CancellationToken;

use crate::{
    error::InferenceError,
    types::{Capabilities, ChatRequest, ComputeDevice, InferenceEvent, LoadSpec},
};

/// A stream of events for one generation. Yields `Err(Cancelled)` and then
/// ends if the cancellation token fires; otherwise ends after `Finished`.
pub type EventStream = BoxStream<'static, Result<InferenceEvent, InferenceError>>;

pub type LoadFuture =
    Pin<Box<dyn Future<Output = Result<Arc<dyn LoadedModel>, InferenceError>> + Send + 'static>>;

/// Loads models. Implemented by the mistral.rs adapter and by test fixtures.
pub trait Engine: Send + Sync + 'static {
    fn load(&self, spec: LoadSpec) -> LoadFuture;
}

/// A loaded model. Shared across requests via `Arc`; dropping the last `Arc`
/// frees the weights.
pub trait LoadedModel: Send + Sync + 'static {
    fn capabilities(&self) -> Capabilities;

    /// The device the weights were loaded on. Engines that only run on the CPU keep the default.
    fn device(&self) -> ComputeDevice {
        ComputeDevice::Cpu
    }

    /// Starts one generation. The engine must stop promptly once `cancel` is
    /// cancelled (not merely when the stream is dropped).
    fn generate(&self, request: ChatRequest, cancel: CancellationToken) -> EventStream;
}

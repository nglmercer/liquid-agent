//! Scripted test engine. Produces deterministic token streams, counts loads,
//! generations, cancellations and drops, and can simulate load failures.

use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use futures_util::stream::{self, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::error::InferenceError;
use crate::traits::{Engine, EventStream, LoadFuture, LoadedModel};
use crate::types::{Capabilities, ChatRequest, InferenceEvent, LoadSpec};

/// Scripted behavior for a [`FixtureEngine`].
#[derive(Debug, Clone)]
pub struct FixtureConfig {
    /// Capabilities reported by the loaded model.
    pub capabilities: Capabilities,
    /// Text tokens emitted, in order, by each generation.
    pub tokens: Vec<String>,
    /// Sleep before each token. Cancellation interrupts the sleep.
    pub token_delay: Duration,
    /// Sleep inside `load` before the result is returned.
    pub load_delay: Duration,
    /// If set, `load` fails with `InferenceError::Load` carrying this message.
    pub load_error: Option<String>,
    /// If set, `load` fails with `InferenceError::Unsupported` (checked first).
    pub unsupported: Option<String>,
}

impl Default for FixtureConfig {
    fn default() -> Self {
        Self {
            capabilities: Capabilities {
                text: true,
                ..Capabilities::default()
            },
            tokens: vec!["Hello".into(), " world".into()],
            token_delay: Duration::ZERO,
            load_delay: Duration::ZERO,
            load_error: None,
            unsupported: None,
        }
    }
}

/// Counters shared between a [`FixtureEngine`] and its loaded models.
#[derive(Debug, Default)]
pub struct FixtureCounters {
    /// Successful loads.
    pub loads: AtomicUsize,
    /// Generation streams started.
    pub generations: AtomicUsize,
    /// Generations that observed cancellation.
    pub cancellations: AtomicUsize,
    /// Loaded models dropped.
    pub models_dropped: AtomicUsize,
}

/// An [`Engine`] that follows a [`FixtureConfig`] script.
#[derive(Debug)]
pub struct FixtureEngine {
    config: FixtureConfig,
    counters: Arc<FixtureCounters>,
}

impl FixtureEngine {
    /// Creates a fixture engine with the given script and fresh counters.
    pub fn new(config: FixtureConfig) -> Self {
        Self {
            config,
            counters: Arc::new(FixtureCounters::default()),
        }
    }

    /// Returns a handle to the counters shared with models this engine loads.
    pub fn counters(&self) -> Arc<FixtureCounters> {
        Arc::clone(&self.counters)
    }
}

impl Engine for FixtureEngine {
    fn load(&self, spec: LoadSpec) -> LoadFuture {
        let config = self.config.clone();
        let counters = Arc::clone(&self.counters);
        Box::pin(async move {
            if let Some(msg) = &config.unsupported {
                return Err(InferenceError::Unsupported(msg.clone()));
            }
            tokio::time::sleep(config.load_delay).await;
            if let Some(message) = &config.load_error {
                return Err(InferenceError::Load {
                    role: spec.role,
                    message: message.clone(),
                });
            }
            counters.loads.fetch_add(1, Ordering::SeqCst);
            let model: Arc<dyn LoadedModel> = Arc::new(FixtureModel { config, counters });
            Ok(model)
        })
    }
}

/// A model loaded from a [`FixtureEngine`].
struct FixtureModel {
    config: FixtureConfig,
    counters: Arc<FixtureCounters>,
}

impl Drop for FixtureModel {
    fn drop(&mut self) {
        self.counters.models_dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl LoadedModel for FixtureModel {
    fn capabilities(&self) -> Capabilities {
        self.config.capabilities
    }

    fn generate(&self, request: ChatRequest, cancel: CancellationToken) -> EventStream {
        self.counters.generations.fetch_add(1, Ordering::SeqCst);
        let prompt_bytes: u64 = request
            .messages
            .iter()
            .map(|m| m.content.len() as u64)
            .sum();
        let run = Run {
            pending: self.config.tokens.iter().cloned().collect(),
            emitted: 0,
            usage_sent: false,
            prompt_tokens: prompt_bytes / 4,
            token_delay: self.config.token_delay,
            cancel,
            counters: Arc::clone(&self.counters),
        };
        stream::unfold(Some(run), next_event).boxed()
    }
}

type Item = Result<InferenceEvent, InferenceError>;

/// State of one generation stream.
struct Run {
    pending: VecDeque<String>,
    emitted: u64,
    usage_sent: bool,
    prompt_tokens: u64,
    token_delay: Duration,
    cancel: CancellationToken,
    counters: Arc<FixtureCounters>,
}

/// Produces the next item of a generation. `None` state ends the stream.
async fn next_event(state: Option<Run>) -> Option<(Item, Option<Run>)> {
    let mut run = state?;
    if let Some(token) = run.pending.pop_front() {
        if cancelled_during(&run.cancel, run.token_delay).await {
            return Some(cancelled(&run.counters));
        }
        run.emitted += 1;
        return Some((Ok(InferenceEvent::TextDelta(token)), Some(run)));
    }
    if run.usage_sent {
        return Some((Ok(InferenceEvent::Finished), None));
    }
    if run.cancel.is_cancelled() {
        return Some(cancelled(&run.counters));
    }
    run.usage_sent = true;
    let usage = InferenceEvent::Usage {
        prompt_tokens: run.prompt_tokens,
        output_tokens: run.emitted,
    };
    Some((Ok(usage), Some(run)))
}

/// Waits `delay`, returning `true` early if `cancel` fires (or already fired).
async fn cancelled_during(cancel: &CancellationToken, delay: Duration) -> bool {
    if cancel.is_cancelled() {
        return true;
    }
    tokio::select! {
        biased;
        _ = cancel.cancelled() => true,
        _ = tokio::time::sleep(delay) => false,
    }
}

/// Records the cancellation and produces the terminal item; the stream ends.
fn cancelled(counters: &FixtureCounters) -> (Item, Option<Run>) {
    counters.cancellations.fetch_add(1, Ordering::SeqCst);
    (Err(InferenceError::Cancelled), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, DevicePreference, ModelRole};
    use std::path::PathBuf;

    fn spec() -> LoadSpec {
        LoadSpec {
            role: ModelRole::Small,
            weights: PathBuf::from("/x"),
            projector: None,
            context_tokens: 512,
            max_concurrent: 1,
            device: DevicePreference::Auto,
        }
    }

    fn request() -> ChatRequest {
        ChatRequest::new(vec![ChatMessage::text("user", "hi")], 16)
    }

    #[tokio::test]
    async fn default_config_streams_tokens_usage_and_finish() {
        let engine = FixtureEngine::new(FixtureConfig::default());
        let counters = engine.counters();
        let model = engine.load(spec()).await.expect("load");

        let events: Vec<Item> = model
            .generate(request(), CancellationToken::new())
            .collect()
            .await;

        assert_eq!(
            events,
            vec![
                Ok(InferenceEvent::TextDelta("Hello".into())),
                Ok(InferenceEvent::TextDelta(" world".into())),
                Ok(InferenceEvent::Usage {
                    prompt_tokens: 0,
                    output_tokens: 2,
                }),
                Ok(InferenceEvent::Finished),
            ]
        );
        assert_eq!(counters.generations.load(Ordering::SeqCst), 1);
        assert_eq!(counters.loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancelled_before_consuming_yields_cancelled_then_ends() {
        let engine = FixtureEngine::new(FixtureConfig::default());
        let counters = engine.counters();
        let model = engine.load(spec()).await.expect("load");

        let cancel = CancellationToken::new();
        cancel.cancel();
        let mut stream = model.generate(request(), cancel);

        assert_eq!(stream.next().await, Some(Err(InferenceError::Cancelled)));
        assert!(stream.next().await.is_none());
        assert_eq!(counters.cancellations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn load_error_is_reported_with_message() {
        let engine = FixtureEngine::new(FixtureConfig {
            load_error: Some("boom".into()),
            ..FixtureConfig::default()
        });

        let err = engine.load(spec()).await.err();

        assert_eq!(
            err,
            Some(InferenceError::Load {
                role: ModelRole::Small,
                message: "boom".into(),
            })
        );
    }
}

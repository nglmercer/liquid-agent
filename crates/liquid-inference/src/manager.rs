//! In-process model manager: lifecycle, generation leases, and cancellation.
//!
//! The manager owns one slot per [`ModelRole`]. Each slot holds the loaded
//! model, a semaphore that bounds concurrent generations, and the time of the
//! last generation. Lifecycle state lives in [`Registry`].
//!
//! Locking: a single `std::sync::Mutex` guards the state. It is only held for
//! short, synchronous sections and is never held across an `.await` or while
//! a generation stream is polled.

use std::{
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard},
    task::{Context, Poll},
    time::Instant,
};

use futures_util::Stream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::{CancellationToken, DropGuard};

use crate::{
    error::InferenceError,
    registry::Registry,
    traits::{Engine, EventStream, LoadedModel},
    types::{
        Capabilities, ChatRequest, ComputeDevice, InferenceEvent, LoadSpec, ModelRole, ModelState,
    },
};

/// Owns the loaded models and hands out leases for generation.
///
/// Cloning is cheap: all clones share the same state, so one clone can run a
/// generation while another unloads the model.
#[derive(Clone)]
pub struct InferenceManager {
    inner: Arc<Inner>,
}

struct Inner {
    engine: Arc<dyn Engine>,
    state: Mutex<State>,
}

struct State {
    registry: Registry,
    small: Slot,
    omni: Slot,
}

/// Per-role resources. `model` and `semaphore` are present together, only
/// while the role is `Ready` or `Busy`, or `Unloading` before the drain ends.
#[derive(Default)]
struct Slot {
    model: Option<Arc<dyn LoadedModel>>,
    semaphore: Option<Arc<Semaphore>>,
    max_concurrent: u32,
    last_used: Option<Instant>,
}

impl State {
    fn slot(&self, role: ModelRole) -> &Slot {
        match role {
            ModelRole::Small => &self.small,
            ModelRole::Omni => &self.omni,
        }
    }

    fn slot_mut(&mut self, role: ModelRole) -> &mut Slot {
        match role {
            ModelRole::Small => &mut self.small,
            ModelRole::Omni => &mut self.omni,
        }
    }

    /// The registry state, with `Busy` derived from in-flight permits.
    fn derived_status(&self, role: ModelRole) -> ModelState {
        match self.registry.status(role) {
            ModelState::Ready => {
                let slot = self.slot(role);
                match &slot.semaphore {
                    Some(sem) if sem.available_permits() < slot.max_concurrent as usize => {
                        ModelState::Busy
                    }
                    _ => ModelState::Ready,
                }
            }
            other => other,
        }
    }
}

impl InferenceManager {
    /// Creates a manager that loads models through `engine`.
    pub fn new(engine: Arc<dyn Engine>) -> Self {
        Self {
            inner: Arc::new(Inner {
                engine,
                state: Mutex::new(State {
                    registry: Registry::new(),
                    small: Slot::default(),
                    omni: Slot::default(),
                }),
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Loads the model described by `spec` into the slot for `spec.role`.
    ///
    /// Fails with `Unsupported` if `max_concurrent` is zero or too large, and
    /// with `InvalidTransition` if the role is already loading, ready, or
    /// unloading. On engine failure the role is left `Failed` and the original
    /// error is returned.
    pub async fn load(&self, spec: LoadSpec) -> Result<(), InferenceError> {
        let role = spec.role;
        let max_concurrent = checked_permits(spec.max_concurrent)?;
        {
            let mut state = self.lock();
            state.registry.begin_load(role)?;
        }

        let result = self.inner.engine.load(spec).await;

        let mut state = self.lock();
        match result {
            Ok(model) => {
                let slot = state.slot_mut(role);
                slot.model = Some(model);
                slot.semaphore = Some(Arc::new(Semaphore::new(max_concurrent as usize)));
                slot.max_concurrent = max_concurrent;
                slot.last_used = Some(Instant::now());
                state.registry.finish_load(role)?;
                Ok(())
            }
            Err(err) => {
                // The state is Loading here (begin_load above), so this
                // transition cannot fail; the original error is what matters.
                let _ = state.registry.fail_load(role, failure_message(&err));
                Err(err)
            }
        }
    }

    /// Unloads the model for `role`, waiting for in-flight generations to end.
    ///
    /// New generations are refused as `Busy` as soon as this is called. The
    /// model is dropped after the last permit is released. Unloading an
    /// unloaded role succeeds immediately. Calling this while the role is
    /// already unloading fails with `InvalidTransition`.
    pub async fn unload(&self, role: ModelRole) -> Result<(), InferenceError> {
        // The drain runs in its own task so that dropping this future (for
        // example, when the caller's request is cancelled) cannot leave the
        // role stuck in `Unloading`.
        let manager = self.clone();
        tokio::spawn(async move { manager.drain_and_unload(role).await })
            .await
            .map_err(|err| InferenceError::Engine(format!("unload task failed: {err}")))?
    }

    async fn drain_and_unload(&self, role: ModelRole) -> Result<(), InferenceError> {
        let (semaphore, max_concurrent) = {
            let mut state = self.lock();
            if state.registry.status(role) == ModelState::Unloaded {
                return Ok(());
            }
            state.registry.begin_unload(role)?;
            let slot = state.slot(role);
            (slot.semaphore.clone(), slot.max_concurrent)
        };

        // Wait until every in-flight generation has released its permit. No
        // new permits can be taken because the state is Unloading.
        let drained = match semaphore {
            Some(sem) => Some(sem.acquire_many_owned(max_concurrent).await),
            None => None,
        };

        // Detach the model under the lock, then drop it outside the lock so a
        // slow weight free does not block status or generate calls.
        let model = {
            let mut state = self.lock();
            let slot = state.slot_mut(role);
            slot.semaphore = None;
            slot.max_concurrent = 0;
            slot.model.take()
        };
        drop(model);

        {
            let mut state = self.lock();
            state.registry.finish_unload(role)?;
        }

        match drained {
            Some(Err(_)) => Err(InferenceError::Engine(
                "generation drain failed: semaphore closed".to_string(),
            )),
            _ => Ok(()),
        }
    }

    /// Returns the lifecycle state of `role`. A ready model with generations
    /// in flight is reported as `Busy`.
    pub fn status(&self, role: ModelRole) -> ModelState {
        self.lock().derived_status(role)
    }

    /// Returns the capabilities of the loaded model for `role`.
    ///
    /// Fails with `NotLoaded` unless the role is `Ready` or `Busy`.
    pub fn capabilities(&self, role: ModelRole) -> Result<Capabilities, InferenceError> {
        Ok(self.loaded(role)?.capabilities())
    }

    /// Returns the device the loaded model for `role` runs on.
    ///
    /// Fails with `NotLoaded` unless the role is `Ready` or `Busy`.
    pub fn device(&self, role: ModelRole) -> Result<ComputeDevice, InferenceError> {
        Ok(self.loaded(role)?.device())
    }

    fn loaded(&self, role: ModelRole) -> Result<Arc<dyn LoadedModel>, InferenceError> {
        let state = self.lock();
        match state.derived_status(role) {
            ModelState::Ready | ModelState::Busy => {}
            _ => return Err(InferenceError::NotLoaded(role)),
        }
        state
            .slot(role)
            .model
            .clone()
            .ok_or(InferenceError::NotLoaded(role))
    }

    /// Starts one generation on the model for `role`.
    ///
    /// This call is synchronous and does not queue. It fails with `NotLoaded`
    /// if no model is present, and with `Busy` if the role is unloading or all
    /// generation permits are taken. Dropping the returned stream cancels the
    /// engine request and releases its permit. Cancelling `cancel` makes the
    /// stream yield `Err(Cancelled)`.
    pub fn generate(
        &self,
        role: ModelRole,
        request: ChatRequest,
        cancel: CancellationToken,
    ) -> Result<EventStream, InferenceError> {
        let (model, permit) = {
            let mut state = self.lock();
            if state.registry.status(role) == ModelState::Unloading {
                return Err(InferenceError::Busy(role));
            }
            let slot = state.slot(role);
            let model = slot.model.clone().ok_or(InferenceError::NotLoaded(role))?;
            let semaphore = slot
                .semaphore
                .clone()
                .ok_or(InferenceError::NotLoaded(role))?;
            let permit = semaphore
                .try_acquire_owned()
                .map_err(|_| InferenceError::Busy(role))?;
            state.slot_mut(role).last_used = Some(Instant::now());
            (model, permit)
        };

        let engine_cancel = cancel.child_token();
        let cancel_on_drop = engine_cancel.clone().drop_guard();
        let inner = model.generate(request, engine_cancel);

        let stream: EventStream = Box::pin(GenerationStream {
            inner: Some(inner),
            permit: Some(permit),
            model: Some(model),
            _cancel_on_drop: cancel_on_drop,
        });
        Ok(stream)
    }

    /// Returns the time a generation last started for `role`, if ever.
    pub fn last_used(&self, role: ModelRole) -> Option<Instant> {
        self.lock().slot(role).last_used
    }
}

/// Validates `max_concurrent` and converts it to the semaphore's permit type.
fn checked_permits(max_concurrent: usize) -> Result<u32, InferenceError> {
    if max_concurrent == 0 {
        return Err(InferenceError::Unsupported(
            "max_concurrent must be at least 1".to_string(),
        ));
    }
    u32::try_from(max_concurrent)
        .ok()
        .filter(|&n| (n as usize) <= Semaphore::MAX_PERMITS)
        .ok_or_else(|| {
            InferenceError::Unsupported(format!("max_concurrent {max_concurrent} is too large"))
        })
}

/// The text stored in `Failed(..)` for an engine error.
fn failure_message(err: &InferenceError) -> String {
    match err {
        InferenceError::Load { message, .. } => message.clone(),
        other => other.to_string(),
    }
}

/// Wraps an engine stream and holds the resources a generation needs.
///
/// The permit is released as soon as the inner stream ends, so the slot frees
/// even if the caller keeps this value alive. Dropping the value cancels the
/// engine request through `_cancel_on_drop`.
struct GenerationStream {
    inner: Option<EventStream>,
    permit: Option<OwnedSemaphorePermit>,
    model: Option<Arc<dyn LoadedModel>>,
    _cancel_on_drop: DropGuard,
}

impl GenerationStream {
    fn release(&mut self) {
        self.inner = None;
        self.permit = None;
        self.model = None;
    }
}

impl Stream for GenerationStream {
    type Item = Result<InferenceEvent, InferenceError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        let polled = inner.as_mut().poll_next(cx);
        if let Poll::Ready(None) = polled {
            this.release();
        }
        polled
    }
}

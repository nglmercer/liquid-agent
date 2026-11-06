//! Lifecycle state machine for model roles.
//!
//! This module is pure logic: it holds no engine handles and performs no
//! async work. The [`crate::InferenceManager`] drives it under a short lock
//! and attaches the actual models and generation slots separately.
//!
//! `Busy` is never stored here. The manager derives it from `Ready` while
//! generations hold permits.

use crate::{
    error::InferenceError,
    types::{ModelRole, ModelState},
};

/// Tracks one [`ModelState`] per [`ModelRole`] and validates transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registry {
    small: ModelState,
    omni: ModelState,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// Creates a registry where both roles are [`ModelState::Unloaded`].
    pub fn new() -> Self {
        Self {
            small: ModelState::Unloaded,
            omni: ModelState::Unloaded,
        }
    }

    /// Returns the stored state for `role`.
    pub fn status(&self, role: ModelRole) -> ModelState {
        self.slot(role).clone()
    }

    /// Moves `Unloaded` or `Failed` to `Loading`.
    pub fn begin_load(&mut self, role: ModelRole) -> Result<(), InferenceError> {
        self.transition(role, &["Unloaded", "Failed"], ModelState::Loading)
    }

    /// Moves `Loading` to `Ready`.
    pub fn finish_load(&mut self, role: ModelRole) -> Result<(), InferenceError> {
        self.transition(role, &["Loading"], ModelState::Ready)
    }

    /// Moves `Loading` to `Failed(message)`.
    pub fn fail_load(
        &mut self,
        role: ModelRole,
        message: impl Into<String>,
    ) -> Result<(), InferenceError> {
        self.transition(role, &["Loading"], ModelState::Failed(message.into()))
    }

    /// Moves `Ready` to `Unloading`. `Unloaded` is a no-op that succeeds.
    pub fn begin_unload(&mut self, role: ModelRole) -> Result<(), InferenceError> {
        if self.slot(role) == &ModelState::Unloaded {
            return Ok(());
        }
        self.transition(role, &["Ready"], ModelState::Unloading)
    }

    /// Moves `Unloading` to `Unloaded`.
    pub fn finish_unload(&mut self, role: ModelRole) -> Result<(), InferenceError> {
        self.transition(role, &["Unloading"], ModelState::Unloaded)
    }

    fn slot(&self, role: ModelRole) -> &ModelState {
        match role {
            ModelRole::Small => &self.small,
            ModelRole::Omni => &self.omni,
        }
    }

    fn slot_mut(&mut self, role: ModelRole) -> &mut ModelState {
        match role {
            ModelRole::Small => &mut self.small,
            ModelRole::Omni => &mut self.omni,
        }
    }

    fn transition(
        &mut self,
        role: ModelRole,
        allowed_from: &[&str],
        next: ModelState,
    ) -> Result<(), InferenceError> {
        let to = state_name(&next);
        let slot = self.slot_mut(role);
        let from = state_name(slot);
        if allowed_from.contains(&from) {
            *slot = next;
            Ok(())
        } else {
            Err(InferenceError::InvalidTransition {
                role,
                from: from.to_string(),
                to: to.to_string(),
            })
        }
    }
}

/// Returns the state name used in `InvalidTransition` errors.
fn state_name(state: &ModelState) -> &'static str {
    match state {
        ModelState::Unloaded => "Unloaded",
        ModelState::Loading => "Loading",
        ModelState::Ready => "Ready",
        ModelState::Busy => "Busy",
        ModelState::Unloading => "Unloading",
        ModelState::Failed(_) => "Failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid(role: ModelRole, from: &str, to: &str) -> InferenceError {
        InferenceError::InvalidTransition {
            role,
            from: from.to_string(),
            to: to.to_string(),
        }
    }

    fn ready(role: ModelRole) -> Registry {
        let mut reg = Registry::new();
        reg.begin_load(role).unwrap_or_else(|e| panic!("{e}"));
        reg.finish_load(role).unwrap_or_else(|e| panic!("{e}"));
        reg
    }

    #[test]
    fn fresh_registry_is_unloaded_for_both_roles() {
        let reg = Registry::new();
        assert_eq!(reg.status(ModelRole::Small), ModelState::Unloaded);
        assert_eq!(reg.status(ModelRole::Omni), ModelState::Unloaded);
    }

    #[test]
    fn load_then_unload_cycle_is_legal() {
        let mut reg = ready(ModelRole::Small);
        assert_eq!(reg.status(ModelRole::Small), ModelState::Ready);
        assert_eq!(reg.begin_unload(ModelRole::Small), Ok(()));
        assert_eq!(reg.status(ModelRole::Small), ModelState::Unloading);
        assert_eq!(reg.finish_unload(ModelRole::Small), Ok(()));
        assert_eq!(reg.status(ModelRole::Small), ModelState::Unloaded);
    }

    #[test]
    fn begin_load_while_loading_is_invalid() {
        let mut reg = Registry::new();
        reg.begin_load(ModelRole::Small)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            reg.begin_load(ModelRole::Small),
            Err(invalid(ModelRole::Small, "Loading", "Loading"))
        );
    }

    #[test]
    fn begin_load_while_ready_is_invalid_until_unloaded() {
        let mut reg = ready(ModelRole::Omni);
        assert_eq!(
            reg.begin_load(ModelRole::Omni),
            Err(invalid(ModelRole::Omni, "Ready", "Loading"))
        );
        assert_eq!(reg.status(ModelRole::Omni), ModelState::Ready);
    }

    #[test]
    fn failed_load_records_message_and_allows_retry() {
        let mut reg = Registry::new();
        reg.begin_load(ModelRole::Small)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reg.fail_load(ModelRole::Small, "boom"), Ok(()));
        assert_eq!(
            reg.status(ModelRole::Small),
            ModelState::Failed("boom".to_string())
        );
        assert_eq!(reg.begin_load(ModelRole::Small), Ok(()));
        assert_eq!(reg.status(ModelRole::Small), ModelState::Loading);
    }

    #[test]
    fn begin_unload_when_unloaded_is_a_noop() {
        let mut reg = Registry::new();
        assert_eq!(reg.begin_unload(ModelRole::Small), Ok(()));
        assert_eq!(reg.status(ModelRole::Small), ModelState::Unloaded);
    }

    #[test]
    fn begin_unload_while_loading_is_invalid() {
        let mut reg = Registry::new();
        reg.begin_load(ModelRole::Small)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            reg.begin_unload(ModelRole::Small),
            Err(invalid(ModelRole::Small, "Loading", "Unloading"))
        );
    }

    #[test]
    fn begin_unload_while_unloading_is_invalid() {
        let mut reg = ready(ModelRole::Small);
        reg.begin_unload(ModelRole::Small)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            reg.begin_unload(ModelRole::Small),
            Err(invalid(ModelRole::Small, "Unloading", "Unloading"))
        );
    }

    #[test]
    fn finish_without_matching_begin_is_invalid() {
        let mut reg = Registry::new();
        assert_eq!(
            reg.finish_load(ModelRole::Small),
            Err(invalid(ModelRole::Small, "Unloaded", "Ready"))
        );
        assert_eq!(
            reg.fail_load(ModelRole::Small, "x"),
            Err(invalid(ModelRole::Small, "Unloaded", "Failed"))
        );
        assert_eq!(
            reg.finish_unload(ModelRole::Small),
            Err(invalid(ModelRole::Small, "Unloaded", "Unloaded"))
        );
        assert_eq!(reg.status(ModelRole::Small), ModelState::Unloaded);
    }

    #[test]
    fn roles_transition_independently() {
        let mut reg = ready(ModelRole::Small);
        assert_eq!(reg.status(ModelRole::Omni), ModelState::Unloaded);
        assert_eq!(reg.begin_load(ModelRole::Omni), Ok(()));
        assert_eq!(reg.status(ModelRole::Small), ModelState::Ready);
        assert_eq!(reg.status(ModelRole::Omni), ModelState::Loading);
    }
}

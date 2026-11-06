use crate::types::ModelRole;

/// Errors surfaced by the inference boundary. Variants are stable so the
/// server can map them to user-facing messages without string matching.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InferenceError {
    /// The role has no loaded model.
    #[error("model {0:?} is not loaded")]
    NotLoaded(ModelRole),
    /// The role is at capacity (all generation slots taken) or is draining.
    #[error("model {0:?} is busy")]
    Busy(ModelRole),
    /// A lifecycle operation is not allowed from the current state.
    #[error("invalid transition for {role:?}: {from} -> {to}")]
    InvalidTransition {
        role: ModelRole,
        from: String,
        to: String,
    },
    /// Loading failed. `message` is the engine's original error text.
    #[error("failed to load {role:?}: {message}")]
    Load { role: ModelRole, message: String },
    /// The model, quantization, or architecture is not supported. The message
    /// must be actionable for the user.
    #[error("unsupported model: {0}")]
    Unsupported(String),
    /// A generation or engine-side error.
    #[error("engine error: {0}")]
    Engine(String),
    /// The generation was cancelled by the caller.
    #[error("generation cancelled")]
    Cancelled,
}

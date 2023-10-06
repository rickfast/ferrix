use internal::{InferRequest, InferResponse};
use thiserror::Error;

pub mod internal;
pub mod python;

pub trait Model: Send + Sync {
    fn load(&mut self) -> ModelResult<()>;
    fn loaded(&self) -> bool;
    fn predict(&self, request: InferRequest) -> ModelResult<InferResponse>;
}

pub type ModelResult<T> = std::result::Result<T, anyhow::Error>;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("model load error error: {0}")]
    Load(String),
    #[error("prediction error: {0}")]
    Prediction(String),
    #[error(transparent)]
    Wrapped(Box<dyn std::error::Error + Send + Sync>),
}

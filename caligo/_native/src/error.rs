//! Error types for the caligo-codecs crate.

use thiserror::Error;

/// Top-level error type for codec operations.
#[derive(Error, Debug)]
pub enum CodecError {
    #[error("Encoding error: {0}")]
    Encode(#[from] crate::polar::encoder::EncoderError),
    
    #[error("Decoding error: {0}")]
    Decode(#[from] crate::polar::decoder::DecoderError),
    
    #[error("Configuration error: {0}")]
    Config(String),
}

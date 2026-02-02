//! Caligo Codecs: High-performance Polar codec for QKD reconciliation.
//!
//! This crate provides Rust-accelerated encoding and decoding for Polar codes,
//! with Python bindings via PyO3.
//!
//! # Architecture
//!
//! ```text
//! Python (caligo.reconciliation)
//!     │
//!     ▼
//! PyO3 Bindings (this module)
//!     │
//!     ▼
//! Rust Core (polar::encoder, polar::decoder)
//! ```
//!
//! # GIL Release
//!
//! All compute-intensive methods release the GIL via `py.allow_threads()`,
//! enabling true parallelism across Python threads.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use bitvec::prelude::*;

pub mod error;
pub mod polar;

use polar::construction::{construct_frozen_mask, ConstructionMethod};
use polar::encoder::PolarEncoder;
use polar::decoder::SCDecoder;
use polar::crc;

/// Python-visible Polar codec implementing the SISOCodec protocol pattern.
///
/// This class wraps the Rust encoder and decoder, providing:
/// - `encode()`: Message → Codeword
/// - `decode_hard()`: Received bits → Message (hard decision)
/// - `decode_soft()`: Channel LLRs → (Extrinsic LLRs, Message) (soft decision)
///
/// All heavy computation releases the GIL for parallel execution.
#[pyclass(name = "PolarCodec")]
pub struct PyPolarCodec {
    encoder: PolarEncoder,
    decoder: SCDecoder,
    #[allow(dead_code)]
    frozen_mask: BitVec<u64, Lsb0>,
    block_length: usize,
    message_length: usize,
    crc_length: usize,
}

#[pymethods]
impl PyPolarCodec {
    /// Create a new Polar codec.
    ///
    /// # Arguments
    /// * `block_length` - Code block length N (must be power of 2)
    /// * `message_length` - Number of information bits K
    /// * `design_snr_db` - Design SNR in dB for frozen bit construction
    /// * `crc_length` - CRC length (0 or 16 supported). Default: 0 for Phase 1.
    ///
    /// # Raises
    /// * `ValueError` - If block_length is not a power of 2
    /// * `ValueError` - If message_length > block_length
    #[new]
    #[pyo3(signature = (block_length, message_length, *, design_snr_db=2.0, crc_length=0))]
    fn new(
        block_length: usize,
        message_length: usize,
        design_snr_db: f64,
        crc_length: usize,
    ) -> PyResult<Self> {
        if !block_length.is_power_of_two() {
            return Err(PyValueError::new_err(format!(
                "block_length {} is not a power of 2",
                block_length
            )));
        }
        
        if message_length > block_length {
            return Err(PyValueError::new_err(format!(
                "message_length {} exceeds block_length {}",
                message_length, block_length
            )));
        }
        
        if crc_length != 0 && crc_length != 16 {
            return Err(PyValueError::new_err(
                "crc_length must be 0 or 16"
            ));
        }
        
        // Total information bits = message + CRC
        let k_total = message_length + crc_length;
        
        // Construct frozen mask
        let frozen_mask = construct_frozen_mask(
            block_length,
            k_total,
            design_snr_db,
            ConstructionMethod::GaussianApproximation,
        );
        
        let encoder = PolarEncoder::new(block_length, frozen_mask.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let decoder = SCDecoder::new(block_length, frozen_mask.clone());
        
        Ok(Self {
            encoder,
            decoder,
            frozen_mask,
            block_length,
            message_length,
            crc_length,
        })
    }
    
    /// Encode message bits to codeword.
    ///
    /// GIL is released during encoding computation.
    ///
    /// # Arguments
    /// * `message` - Information bits of shape (message_length,), dtype=uint8
    ///
    /// # Returns
    /// Encoded codeword of shape (block_length,), dtype=uint8
    ///
    /// # Raises
    /// * `ValueError` - If message shape or dtype is incorrect
    fn encode<'py>(
        &self,
        py: Python<'py>,
        message: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let msg_slice = message.as_slice()
            .map_err(|_| PyValueError::new_err("Message must be contiguous"))?;
        
        if msg_slice.len() != self.message_length {
            return Err(PyValueError::new_err(format!(
                "Message length {} does not match expected {}",
                msg_slice.len(), self.message_length
            )));
        }
        
        // Prepare message with optional CRC
        let full_message: Vec<u8> = if self.crc_length > 0 {
            crc::append_crc16(msg_slice)
        } else {
            msg_slice.to_vec()
        };
        
        // Release GIL for encoding
        let codeword = py.allow_threads(|| {
            self.encoder.encode(&full_message)
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(codeword.into_pyarray_bound(py))
    }
    
    /// Hard-decision decoding from received bits.
    ///
    /// GIL is released during decoding computation.
    ///
    /// # Arguments
    /// * `received` - Received bits of shape (block_length,), dtype=uint8
    /// * `qber` - Estimated quantum bit error rate (0.0 to 0.5)
    ///
    /// # Returns
    /// Tuple of (decoded_message, converged, path_metric)
    /// - decoded_message: shape (message_length,), dtype=uint8
    /// - converged: bool (always True for SC L=1)
    /// - path_metric: float (sum of LLR penalties)
    #[pyo3(signature = (received, *, qber=0.05))]
    fn decode_hard<'py>(
        &mut self,
        py: Python<'py>,
        received: PyReadonlyArray1<'py, u8>,
        qber: f32,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, bool, f32)> {
        let recv_slice = received.as_slice()
            .map_err(|_| PyValueError::new_err("Received must be contiguous"))?;
        
        if recv_slice.len() != self.block_length {
            return Err(PyValueError::new_err(format!(
                "Received length {} does not match block_length {}",
                recv_slice.len(), self.block_length
            )));
        }
        
        // Release GIL for decoding
        let recv_vec: Vec<u8> = recv_slice.to_vec();
        let result = py.allow_threads(|| {
            self.decoder.decode_hard(&recv_vec, qber)
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Strip CRC if present
        let message = if self.crc_length > 0 {
            crc::strip_crc16(&result.message)
                .ok_or_else(|| PyRuntimeError::new_err("Failed to strip CRC"))?
        } else {
            result.message
        };
        
        let msg_array = PyArray1::from_vec_bound(py, message);
        
        Ok((msg_array, result.converged, result.path_metric))
    }
    
    /// Soft-decision decoding from channel LLRs.
    ///
    /// GIL is released during decoding computation.
    ///
    /// # Arguments
    /// * `llr_channel` - Channel LLRs of shape (block_length,), dtype=float32
    ///   Convention: positive LLR means bit=0 more likely
    ///
    /// # Returns
    /// Tuple of (extrinsic_llr, decoded_message, converged, iterations, path_metric)
    /// - extrinsic_llr: shape (block_length,), dtype=float32 (stub: returns zeros)
    /// - decoded_message: shape (message_length,), dtype=uint8
    /// - converged: bool
    /// - iterations: int (always 1 for SC)
    /// - path_metric: float
    fn decode_soft<'py>(
        &mut self,
        py: Python<'py>,
        llr_channel: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<u8>>, bool, i32, f32)> {
        let llr_slice = llr_channel.as_slice()
            .map_err(|_| PyValueError::new_err("LLR must be contiguous"))?;
        
        if llr_slice.len() != self.block_length {
            return Err(PyValueError::new_err(format!(
                "LLR length {} does not match block_length {}",
                llr_slice.len(), self.block_length
            )));
        }
        
        // Release GIL for decoding
        let llr_vec: Vec<f32> = llr_slice.to_vec();
        let result = py.allow_threads(|| {
            self.decoder.decode(&llr_vec)
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Strip CRC if present
        let message = if self.crc_length > 0 {
            crc::strip_crc16(&result.message)
                .ok_or_else(|| PyRuntimeError::new_err("Failed to strip CRC"))?
        } else {
            result.message
        };
        
        // Extrinsic LLR computation is a stub for Phase 1
        // Full implementation in Phase 2 with SCL path metrics
        let extrinsic = vec![0.0f32; self.block_length];
        
        let ext_array = PyArray1::from_vec_bound(py, extrinsic);
        let msg_array = PyArray1::from_vec_bound(py, message);
        
        Ok((ext_array, msg_array, result.converged, 1, result.path_metric))
    }
    
    // =========================================================================
    // Properties (SISOCodec protocol)
    // =========================================================================
    
    /// Code block length N.
    #[getter]
    fn block_length(&self) -> usize {
        self.block_length
    }
    
    /// Information bit length k (excluding CRC).
    #[getter]
    fn message_length(&self) -> usize {
        self.message_length
    }
    
    /// Effective code rate R = k / N.
    #[getter]
    fn rate(&self) -> f64 {
        self.message_length as f64 / self.block_length as f64
    }
    
    /// CRC bit length (0 or 16).
    #[getter]
    fn crc_length(&self) -> usize {
        self.crc_length
    }
    
    /// Total information bits including CRC.
    #[getter]
    fn k_total(&self) -> usize {
        self.message_length + self.crc_length
    }
}

/// Python module definition.
#[pymodule]
fn caligo_codecs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPolarCodec>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

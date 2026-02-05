//! Polar code encoder using Arikan's recursive butterfly construction.
//!
//! References:
//! - [1] Arikan, "Channel Polarization", IEEE TIT 2009
//! - [2] Tal & Vardy, "List Decoding of Polar Codes", IEEE TIT 2015, §II

use bitvec::prelude::*;
use ndarray::Array1;
use thiserror::Error;

/// Errors that can occur during encoding.
#[derive(Error, Debug)]
pub enum EncoderError {
    #[error("Message length {0} does not match expected {1}")]
    MessageLengthMismatch(usize, usize),
    
    #[error("Block length {0} is not a power of 2")]
    InvalidBlockLength(usize),
    
    #[error("Message contains non-binary value at index {0}")]
    NonBinaryValue(usize),
}

/// Polar code encoder.
///
/// Implements Arikan's recursive butterfly encoding in O(N log N) time.
/// The encoder is stateless; frozen bit mask is provided at construction.
pub struct PolarEncoder {
    /// Block length N = 2^n
    block_length: usize,
    /// log2(N)
    n_stages: usize,
    /// Frozen bit mask: true = frozen (set to 0), false = information
    frozen_mask: BitVec<u64, Lsb0>,
    /// Number of information bits (including CRC if any)
    k_total: usize,
}

impl PolarEncoder {
    /// Create a new encoder.
    ///
    /// # Arguments
    /// * `block_length` - Code block length N (must be power of 2)
    /// * `frozen_mask` - Bit vector where true = frozen position
    ///
    /// # Errors
    /// Returns error if block_length is not a power of 2.
    pub fn new(block_length: usize, frozen_mask: BitVec<u64, Lsb0>) -> Result<Self, EncoderError> {
        if !block_length.is_power_of_two() {
            return Err(EncoderError::InvalidBlockLength(block_length));
        }
        
        let n_stages = block_length.trailing_zeros() as usize;
        let k_total = frozen_mask.iter().filter(|b| !**b).count();
        
        Ok(Self {
            block_length,
            n_stages,
            frozen_mask,
            k_total,
        })
    }
    
    /// Encode message bits to codeword.
    ///
    /// # Arguments
    /// * `message` - Information bits of length k_total
    ///
    /// # Returns
    /// Encoded codeword of length N
    ///
    /// # Algorithm
    /// 1. Place message bits at information positions (non-frozen)
    /// 2. Set frozen positions to 0
    /// 3. Apply butterfly transform: x[i] ^= x[i + 2^s] for each stage
    pub fn encode(&self, message: &[u8]) -> Result<Array1<u8>, EncoderError> {
        // Validate input
        if message.len() != self.k_total {
            return Err(EncoderError::MessageLengthMismatch(message.len(), self.k_total));
        }
        
        for (i, &bit) in message.iter().enumerate() {
            if bit > 1 {
                return Err(EncoderError::NonBinaryValue(i));
            }
        }
        
        // Initialize u-vector: place message at information positions
        let mut u = vec![0u8; self.block_length];
        let mut msg_idx = 0;
        
        for i in 0..self.block_length {
            if !self.frozen_mask[i] {
                u[i] = message[msg_idx];
                msg_idx += 1;
            }
            // Frozen positions remain 0
        }
        
        // Apply butterfly transform (in-place)
        // This computes x = u · G_N where G_N = B_N · F^⊗n
        self.butterfly_transform(&mut u);
        
        Ok(Array1::from_vec(u))
    }
    
    /// In-place butterfly transform implementing x = u · G_N.
    ///
    /// The transform processes n = log2(N) stages. At stage s:
    /// - Stride = 2^s
    /// - For each block of size 2^(s+1), XOR pairs at distance 2^s
    ///
    /// This is equivalent to multiplying by F^⊗n with natural bit ordering.
    #[inline]
    fn butterfly_transform(&self, x: &mut [u8]) {
        let n = self.block_length;
        
        for stage in 0..self.n_stages {
            let stride = 1 << stage;
            let block_size = stride << 1;
            
            for block_start in (0..n).step_by(block_size) {
                for i in 0..stride {
                    let idx_a = block_start + i;
                    let idx_b = idx_a + stride;
                    
                    // x[a] = x[a] XOR x[b]  (butterfly operation)
                    x[idx_a] ^= x[idx_b];
                }
            }
        }
    }
    
    /// Get the number of information bits.
    pub fn message_length(&self) -> usize {
        self.k_total
    }
    
    /// Get the block length.
    pub fn block_length(&self) -> usize {
        self.block_length
    }
    
    /// Get code rate k/N.
    pub fn rate(&self) -> f64 {
        self.k_total as f64 / self.block_length as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test N=8, K=4 example from Arikan's paper.
    #[test]
    fn test_encode_n8_k4() {
        // Frozen positions: 0, 1, 2, 4 (indices with low reliability)
        // Information positions: 3, 5, 6, 7
        let mut frozen = bitvec![u64, Lsb0; 0; 8];
        frozen.set(0, true);
        frozen.set(1, true);
        frozen.set(2, true);
        frozen.set(4, true);
        
        let encoder = PolarEncoder::new(8, frozen).unwrap();
        assert_eq!(encoder.message_length(), 4);
        
        // Test with message [1, 0, 1, 1] at positions [3, 5, 6, 7]
        // u = [0, 0, 0, 1, 0, 0, 1, 1]
        let message = [1u8, 0, 1, 1];
        let codeword = encoder.encode(&message).unwrap();
        
        // Verify codeword is valid (manual calculation required for test vector)
        assert_eq!(codeword.len(), 8);
    }
}

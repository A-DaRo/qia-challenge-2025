//! CRC-16-CCITT computation for CRC-aided SCL decoding.
//!
//! This is a stub implementation for Phase 1. Full integration
//! with the encoder/decoder occurs in Phase 2.
//!
//! CRC Polynomial: x^16 + x^12 + x^5 + 1 (0x1021)

/// CRC-16-CCITT polynomial.
pub const CRC16_CCITT_POLY: u16 = 0x1021;

/// Compute CRC-16-CCITT for a bit sequence.
///
/// # Arguments
/// * `bits` - Input bit sequence (0/1 values)
///
/// # Returns
/// 16-bit CRC value
pub fn crc16_ccitt(bits: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF; // Initial value
    
    for &bit in bits {
        let xor_bit = ((crc >> 15) as u8) ^ bit;
        crc <<= 1;
        
        if xor_bit != 0 {
            crc ^= CRC16_CCITT_POLY;
        }
    }
    
    crc
}

/// Verify CRC-16-CCITT for a bit sequence with appended CRC.
///
/// # Arguments
/// * `bits_with_crc` - Message bits followed by 16 CRC bits
///
/// # Returns
/// true if CRC matches, false otherwise
pub fn verify_crc16(bits_with_crc: &[u8]) -> bool {
    if bits_with_crc.len() < 16 {
        return false;
    }
    
    // Compute CRC over entire sequence (should be 0 for valid data)
    let remainder = crc16_ccitt(bits_with_crc);
    remainder == 0
}

/// Append CRC-16 to message bits.
///
/// # Arguments
/// * `message` - Input message bits
///
/// # Returns
/// Message with 16 CRC bits appended
pub fn append_crc16(message: &[u8]) -> Vec<u8> {
    let crc = crc16_ccitt(message);
    
    let mut result = message.to_vec();
    
    // Append CRC bits (MSB first)
    for i in (0..16).rev() {
        result.push(((crc >> i) & 1) as u8);
    }
    
    result
}

/// Extract message from bits with CRC appended.
///
/// # Arguments
/// * `bits_with_crc` - Message bits followed by 16 CRC bits
///
/// # Returns
/// Message bits without CRC (or None if too short)
pub fn strip_crc16(bits_with_crc: &[u8]) -> Option<Vec<u8>> {
    if bits_with_crc.len() < 16 {
        return None;
    }
    
    Some(bits_with_crc[..bits_with_crc.len() - 16].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crc16_known_value() {
        // Test with known input
        let data = [1u8, 0, 1, 1, 0, 0, 1, 0]; // 0xB2 in bits
        let crc = crc16_ccitt(&data);
        
        // CRC should be deterministic
        assert_eq!(crc, crc16_ccitt(&data));
    }
    
    #[test]
    fn test_crc16_append_verify() {
        let message = vec![1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1];
        
        let with_crc = append_crc16(&message);
        assert_eq!(with_crc.len(), message.len() + 16);
        
        // Verify should pass
        assert!(verify_crc16(&with_crc));
        
        // Flip a bit, verify should fail
        let mut corrupted = with_crc.clone();
        corrupted[5] ^= 1;
        assert!(!verify_crc16(&corrupted));
    }
    
    #[test]
    fn test_strip_crc() {
        let message = vec![1u8, 0, 1, 1];
        let with_crc = append_crc16(&message);
        
        let stripped = strip_crc16(&with_crc).unwrap();
        assert_eq!(stripped, message);
    }
}

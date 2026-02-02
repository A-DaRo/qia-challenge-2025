//! Integration tests for Polar encoder.

use caligo_codecs::polar::construction::{construct_frozen_mask, ConstructionMethod};
use caligo_codecs::polar::encoder::PolarEncoder;

#[test]
fn test_encoder_n8_k4() {
    let frozen = construct_frozen_mask(8, 4, 2.0, ConstructionMethod::Bhattacharyya);
    let encoder = PolarEncoder::new(8, frozen).unwrap();
    
    assert_eq!(encoder.block_length(), 8);
    assert_eq!(encoder.message_length(), 4);
    
    let message = vec![1u8, 0, 1, 1];
    let codeword = encoder.encode(&message).unwrap();
    
    assert_eq!(codeword.len(), 8);
    
    // Verify all codeword bits are binary
    for &bit in codeword.iter() {
        assert!(bit == 0 || bit == 1);
    }
}

#[test]
fn test_encoder_n1024_k512() {
    let frozen = construct_frozen_mask(1024, 512, 2.0, ConstructionMethod::GaussianApproximation);
    let encoder = PolarEncoder::new(1024, frozen).unwrap();
    
    assert_eq!(encoder.block_length(), 1024);
    assert_eq!(encoder.message_length(), 512);
    
    // Test with alternating pattern
    let message: Vec<u8> = (0..512).map(|i| (i % 2) as u8).collect();
    let codeword = encoder.encode(&message).unwrap();
    
    assert_eq!(codeword.len(), 1024);
    
    // Check code rate
    assert!((encoder.rate() - 0.5).abs() < 0.001);
}

#[test]
fn test_encoder_error_handling() {
    let frozen = construct_frozen_mask(8, 4, 2.0, ConstructionMethod::Bhattacharyya);
    let encoder = PolarEncoder::new(8, frozen).unwrap();
    
    // Wrong message length
    let result = encoder.encode(&[1, 0, 1]);
    assert!(result.is_err());
    
    // Non-binary value
    let result = encoder.encode(&[1, 0, 2, 1]);
    assert!(result.is_err());
}

#[test]
fn test_encoder_invalid_block_length() {
    use bitvec::prelude::*;
    
    // Not power of 2
    let frozen = bitvec![u64, Lsb0; 0; 7];
    let result = PolarEncoder::new(7, frozen);
    assert!(result.is_err());
}

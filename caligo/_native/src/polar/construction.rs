//! Polar code construction via Gaussian Approximation.
//!
//! Computes frozen bit positions based on channel reliability estimates.
//! Uses Bhattacharyya parameter evolution for computational efficiency.
//!
//! References:
//! - [1] Tal & Vardy, "How to Construct Polar Codes", IEEE TIT 2013
//! - [2] Trifonov, "Efficient Design and Decoding of Polar Codes", IEEE TCom 2012

use bitvec::prelude::*;

/// Channel construction methods.
#[derive(Debug, Clone, Copy)]
pub enum ConstructionMethod {
    /// Bhattacharyya parameter evolution (fast, approximate)
    Bhattacharyya,
    /// Gaussian Approximation (more accurate for AWGN)
    GaussianApproximation,
}

/// Construct frozen bit mask for Polar code.
///
/// # Arguments
/// * `block_length` - N = 2^n
/// * `message_length` - K (number of information bits)
/// * `design_snr_db` - Design SNR in dB (for AWGN channel)
/// * `method` - Construction method
///
/// # Returns
/// BitVec where true = frozen position, false = information position
pub fn construct_frozen_mask(
    block_length: usize,
    message_length: usize,
    design_snr_db: f64,
    method: ConstructionMethod,
) -> BitVec<u64, Lsb0> {
    assert!(block_length.is_power_of_two());
    assert!(message_length <= block_length);
    
    let n_frozen = block_length - message_length;
    
    // Compute reliability for each bit position
    let reliabilities = match method {
        ConstructionMethod::Bhattacharyya => {
            bhattacharyya_construction(block_length, design_snr_db)
        }
        ConstructionMethod::GaussianApproximation => {
            gaussian_approximation_construction(block_length, design_snr_db)
        }
    };
    
    // Sort indices by reliability (ascending = least reliable first)
    let mut indices: Vec<usize> = (0..block_length).collect();
    indices.sort_by(|&a, &b| {
        reliabilities[a].partial_cmp(&reliabilities[b]).unwrap()
    });
    
    // Mark n_frozen least reliable positions as frozen
    let mut frozen = bitvec![u64, Lsb0; 0; block_length];
    for &idx in indices.iter().take(n_frozen) {
        frozen.set(idx, true);
    }
    
    frozen
}

/// Bhattacharyya parameter construction.
///
/// Computes Z(W_i) for each synthetic channel using:
/// - Z(W^-) = 2Z - Z^2  (worse channel)
/// - Z(W^+) = Z^2       (better channel)
///
/// Returns negative log reliability (higher = more reliable).
fn bhattacharyya_construction(block_length: usize, design_snr_db: f64) -> Vec<f64> {
    let n = block_length.trailing_zeros() as usize;
    
    // Initial Bhattacharyya parameter for AWGN
    // Z = exp(-SNR) for BSC approximation
    let snr_linear = 10_f64.powf(design_snr_db / 10.0);
    let z_init = (-snr_linear).exp().min(1.0 - 1e-10);
    
    // Initialize all channels with same reliability
    let mut z = vec![z_init; block_length];
    
    // Polarization stages
    for stage in 0..n {
        let half = 1 << stage;
        let mut z_new = vec![0.0; block_length];
        
        for i in 0..block_length {
            let pair_idx = i ^ half;  // Partner in butterfly
            
            if i < pair_idx {
                // Compute polarized channels
                let z_minus = 2.0 * z[i] - z[i] * z[i];  // Worse channel
                let z_plus = z[i] * z[i];                 // Better channel
                
                z_new[i] = z_minus;
                z_new[pair_idx] = z_plus;
            }
        }
        
        z = z_new;
    }
    
    // Convert to reliability: -log(Z) (higher = more reliable)
    z.iter().map(|&zi| {
        if zi <= 0.0 {
            f64::INFINITY  // Perfect channel
        } else if zi >= 1.0 {
            0.0  // Useless channel
        } else {
            -zi.ln()
        }
    }).collect()
}

/// Gaussian Approximation construction.
///
/// Tracks mean LLR evolution through polar transform stages.
/// More accurate than Bhattacharyya for moderate SNR.
fn gaussian_approximation_construction(block_length: usize, design_snr_db: f64) -> Vec<f64> {
    let n = block_length.trailing_zeros() as usize;
    
    // Initial LLR mean for AWGN: μ = 2/σ² = 2 * SNR_linear
    let snr_linear = 10_f64.powf(design_snr_db / 10.0);
    let mu_init = 2.0 * snr_linear;
    
    let mut mu = vec![mu_init; block_length];
    
    // GA update functions (approximations from Trifonov)
    let phi = |x: f64| -> f64 {
        if x < 0.0 {
            0.0
        } else if x < 10.0 {
            (-0.4527 * x.powf(0.86) + 0.0218).exp()
        } else {
            (std::f64::consts::PI * x).sqrt() * (-x / 4.0).exp()
        }
    };
    
    let phi_inv = |y: f64| -> f64 {
        if y <= 0.0 || y >= 1.0 {
            0.0
        } else if y > 0.0 && y <= 1e-10 {
            // Large μ approximation
            4.0 * (-y.ln()) - std::f64::consts::PI.ln()
        } else {
            // Numerical inversion via Newton-Raphson
            let mut x = 1.0;
            for _ in 0..20 {
                let fx = phi(x) - y;
                let dfx = -0.4527 * 0.86 * x.powf(-0.14) * phi(x);
                if dfx.abs() < 1e-15 {
                    break;
                }
                x -= fx / dfx;
                x = x.max(0.001);
            }
            x
        }
    };
    
    // Polarization stages
    for stage in 0..n {
        let half = 1 << stage;
        let mut mu_new = vec![0.0; block_length];
        
        for i in 0..block_length {
            let pair_idx = i ^ half;
            
            if i < pair_idx {
                // GA update rules
                // μ^- ≈ φ^{-1}(1 - (1 - φ(μ))²)
                // μ^+ = 2μ
                let phi_mu = phi(mu[i]);
                let mu_minus = phi_inv(1.0 - (1.0 - phi_mu).powi(2));
                let mu_plus = 2.0 * mu[i];
                
                mu_new[i] = mu_minus;
                mu_new[pair_idx] = mu_plus;
            }
        }
        
        mu = mu_new;
    }
    
    // Return mean LLR as reliability (higher = more reliable)
    mu
}

/// Convenience function to get information bit indices.
pub fn get_information_indices(frozen_mask: &BitVec<u64, Lsb0>) -> Vec<usize> {
    frozen_mask.iter()
        .enumerate()
        .filter(|(_, is_frozen)| !**is_frozen)
        .map(|(i, _)| i)
        .collect()
}

/// Convenience function to get frozen bit indices.
pub fn get_frozen_indices(frozen_mask: &BitVec<u64, Lsb0>) -> Vec<usize> {
    frozen_mask.iter()
        .enumerate()
        .filter(|(_, is_frozen)| **is_frozen)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_construction_n8_k4() {
        let frozen = construct_frozen_mask(8, 4, 0.0, ConstructionMethod::Bhattacharyya);
        
        // Should have exactly 4 frozen positions
        assert_eq!(frozen.count_ones(), 4);
        assert_eq!(frozen.count_zeros(), 4);
        
        // Position 0 should always be frozen (least reliable)
        assert!(frozen[0]);
    }
    
    #[test]
    fn test_construction_n1024_k512() {
        let frozen = construct_frozen_mask(1024, 512, 2.0, ConstructionMethod::GaussianApproximation);
        
        assert_eq!(frozen.count_ones(), 512);  // n_frozen
        assert_eq!(frozen.count_zeros(), 512); // n_info
    }
}

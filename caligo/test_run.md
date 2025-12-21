============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-7.4.4, pluggy-1.6.0 -- /home/adaro/projects/qia_25/qia/bin/python3
cachedir: .pytest_cache
rootdir: /home/adaro/projects/qia_25/qia-challenge-2025/caligo
configfile: pyproject.toml
plugins: cov-7.0.0
collecting ... collected 1114 items

tests/test_phase_boundary_reconciliation_to_amplification.py::test_p34_001_reconciled_bits_can_be_partitioned_and_formatted_to_ot_keys PASSED [  0%]
tests/test_phase_boundary_reconciliation_to_amplification.py::test_p34_010_deliberate_mismatch_raises_contract_violation PASSED [  0%]
tests/test_phase_boundary_sifting_to_reconciliation.py::test_p23_001_sifting_phase_result_drives_single_block_reconciliation PASSED [  0%]
tests/test_phase_boundary_sifting_to_reconciliation.py::test_p23_010_sifting_phase_result_above_hard_limit_rejected PASSED [  0%]
tests/test_phase_boundary_sifting_to_reconciliation.py::test_p23_020_bitarray_to_numpy_to_bytes_is_byte_per_bit_not_packed_bits PASSED [  0%]
tests/test_phase_boundary_simulation_to_reconciliation.py::test_p0r_001_profile_to_nsm_qber_and_rate_are_consistent PASSED [  0%]
tests/test_phase_boundary_simulation_to_reconciliation.py::test_p0r_010_infeasible_profiles_are_flagged PASSED [  0%]
tests/e2e/test_nsm_boundaries.py::TestSecurityConditionBoundary::test_exact_boundary_condition PASSED [  0%]
tests/e2e/test_nsm_boundaries.py::TestSecurityConditionBoundary::test_one_bit_below_boundary PASSED [  0%]
tests/e2e/test_nsm_boundaries.py::TestSecurityConditionBoundary::test_one_bit_above_boundary PASSED [  0%]
tests/e2e/test_nsm_boundaries.py::TestSecurityConditionBoundary::test_leakage_tracker_boundary PASSED [  0%]
tests/e2e/test_nsm_boundaries.py::TestSecurityConditionBoundary::test_margin_calculation_precision PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestHeraldedModelStress::test_low_detection_rate_security PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestHeraldedModelStress::test_variable_detection_rate PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestHeraldedModelStress::test_heralding_loss_composition PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestEtaSemantics::test_eta_zero_no_loss PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestEtaSemantics::test_eta_one_complete_loss PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestEtaSemantics::test_eta_monotonicity PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestEtaSemantics::test_eta_bounds_validation PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestDeltaTiming::test_zero_timing_insecure PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestDeltaTiming::test_timing_tradeoff PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestCombinedParameterStress::test_worst_case_parameters PASSED [  1%]
tests/e2e/test_nsm_boundaries.py::TestCombinedParameterStress::test_best_case_parameters PASSED [  2%]
tests/e2e/test_nsm_boundaries.py::TestCombinedParameterStress::test_real_world_scenario PASSED [  2%]
tests/e2e/test_parallel_simulation.py::TestParallelSimulationE2E::test_10k_pairs_parallel_4_workers PASSED [  2%]
tests/e2e/test_parallel_simulation.py::TestParallelSimulationE2E::test_50k_pairs_parallel_8_workers PASSED [  2%]
tests/e2e/test_parallel_simulation.py::TestParallelSimulationE2E::test_realistic_noise_model PASSED [  2%]
tests/e2e/test_parallel_simulation.py::TestParallelSimulationE2E::test_ideal_channel PASSED [  2%]
tests/e2e/test_parallel_simulation.py::TestParallelSimulationE2E::test_high_noise_channel PASSED [  2%]
tests/e2e/test_parallel_simulation.py::TestParallelSimulationE2E::test_uneven_batch_distribution PASSED [  2%]
tests/e2e/test_parallel_simulation.py::TestParallelSimulationE2E::test_single_worker_large_batch PASSED [  2%]
tests/e2e/test_parallel_simulation.py::TestParallelSimulationE2E::test_many_small_batches PASSED [  2%]
tests/e2e/test_phase_e_protocol.py::test_phase_e_end_to_end_ot_agreement[0] FAILED [  2%]
tests/e2e/test_phase_e_protocol.py::test_phase_e_end_to_end_ot_agreement[1] FAILED [  3%]
tests/e2e/test_phase_e_protocol.py::test_phase_e_blind_reconciliation_ot_agreement[0] FAILED [  3%]
tests/e2e/test_phase_e_protocol.py::test_phase_e_blind_reconciliation_ot_agreement[1] FAILED [  3%]
tests/e2e/test_recon_complexity.py::TestBlindMiscalibration::test_heuristic_too_low PASSED [  3%]
tests/e2e/test_recon_complexity.py::TestBlindMiscalibration::test_heuristic_too_high PASSED [  3%]
tests/e2e/test_recon_complexity.py::TestPessimisticRateWaste::test_rate_waste_quantification PASSED [  3%]
tests/e2e/test_recon_complexity.py::TestPessimisticRateWaste::test_efficiency_metric PASSED [  3%]
tests/e2e/test_recon_complexity.py::TestPessimisticRateWaste::test_rate_stepping_granularity PASSED [  3%]
tests/e2e/test_recon_complexity.py::TestTinyBlockConstraints::test_minimum_block_size PASSED [  3%]
tests/e2e/test_recon_complexity.py::TestTinyBlockConstraints::test_small_payload_handling PASSED [  3%]
tests/e2e/test_recon_complexity.py::TestTinyBlockConstraints::test_hash_collision_on_small_blocks PASSED [  3%]
tests/e2e/test_recon_complexity.py::TestTinyBlockConstraints::test_statistical_qber_variance_small_blocks PASSED [  4%]
tests/e2e/test_recon_complexity.py::TestIterationBudget::test_iteration_limit_respected PASSED [  4%]
tests/e2e/test_recon_complexity.py::TestIterationBudget::test_early_termination_saves_iterations PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestHighQBERSaturation::test_baseline_high_qber_selects_minimum_rate PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestHighQBERSaturation::test_blind_exhausts_iterations_on_high_error PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestHighQBERSaturation::test_leakage_accounted_on_failure PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestTimingViolations::test_timing_violation_error_type PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestTimingViolations::test_timing_barrier_enforcement PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestCodecStability::test_multiple_blocks_no_memory_accumulation PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestCodecStability::test_blind_state_isolated_between_blocks PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestErrorRecoveryPatterns::test_leakage_exceeded_is_recoverable PASSED [  4%]
tests/e2e/test_stress_conditions.py::TestErrorRecoveryPatterns::test_security_errors_hierarchy PASSED [  5%]
tests/e2e/test_system_stress.py::TestLongWaitTiming::test_protocol_handles_long_delays PASSED [  5%]
tests/e2e/test_system_stress.py::TestLongWaitTiming::test_multiple_sessions_independent PASSED [  5%]
tests/e2e/test_system_stress.py::TestLongWaitTiming::test_state_persistence_across_waits PASSED [  5%]
tests/e2e/test_system_stress.py::TestIncompatibleConfigurations::test_frame_size_mismatch_detection PASSED [  5%]
tests/e2e/test_system_stress.py::TestIncompatibleConfigurations::test_rate_incompatibility_detection PASSED [  5%]
tests/e2e/test_system_stress.py::TestIncompatibleConfigurations::test_topology_consistency PASSED [  5%]
tests/e2e/test_system_stress.py::TestIncompatibleConfigurations::test_pattern_rate_correspondence PASSED [  5%]
tests/e2e/test_system_stress.py::TestHighLossChannel::test_extreme_loss_qber_estimation PASSED [  5%]
tests/e2e/test_system_stress.py::TestHighLossChannel::test_minimum_detections_for_security PASSED [  5%]
tests/e2e/test_system_stress.py::TestHighLossChannel::test_abort_on_insufficient_bits PASSED [  5%]
tests/e2e/test_system_stress.py::TestHighLossChannel::test_loss_budget_accounting PASSED [  6%]
tests/e2e/test_system_stress.py::TestHighLossChannel::test_adaptive_block_size_under_loss PASSED [  6%]
tests/e2e/test_system_stress.py::TestThroughputLimits::test_blocks_per_second_estimate PASSED [  6%]
tests/e2e/test_system_stress.py::TestThroughputLimits::test_memory_usage_bounded PASSED [  6%]
tests/e2e/test_system_stress.py::TestErrorHandlingRobustness::test_corrupted_message_detection PASSED [  6%]
tests/e2e/test_system_stress.py::TestErrorHandlingRobustness::test_out_of_order_messages PASSED [  6%]
tests/e2e/test_system_stress.py::TestErrorHandlingRobustness::test_duplicate_block_handling PASSED [  6%]
tests/e2e/test_system_stress.py::TestErrorHandlingRobustness::test_negative_values_rejected PASSED [  6%]
tests/e2e/test_system_stress.py::TestConcurrentAccess::test_leakage_tracker_thread_safety PASSED [  6%]
tests/e2e/test_system_stress.py::TestConcurrentAccess::test_strategy_stateless_operations PASSED [  6%]
tests/integration/test_nsm_parameter_enforcement.py::TestStackRunnerConfigConversion::test_depolarise_config_preserves_fidelity PASSED [  6%]
tests/integration/test_nsm_parameter_enforcement.py::TestStackRunnerConfigConversion::test_heralded_config_preserves_detector_fields PASSED [  7%]
tests/integration/test_nsm_parameter_enforcement.py::TestDepolariseLinkBuild::test_depolarise_installs_expected_magic_model_params PASSED [  7%]
tests/integration/test_nsm_parameter_enforcement.py::TestDepolariseLinkBuild::test_depolarise_auto_selection PASSED [  7%]
tests/integration/test_nsm_parameter_enforcement.py::TestHeraldedLinkBuild::test_heralded_installs_detector_and_dark_counts PASSED [  7%]
tests/integration/test_nsm_parameter_enforcement.py::TestHeraldedLinkBuild::test_heralded_auto_selection_with_low_eta PASSED [  7%]
tests/integration/test_nsm_parameter_enforcement.py::TestHeraldedLinkBuild::test_heralded_with_dark_count_only PASSED [  7%]
tests/integration/test_nsm_parameter_enforcement.py::TestDeviceNoiseInstallation::test_t1t2_noise_installed_in_memory PASSED [  7%]
tests/integration/test_nsm_parameter_enforcement.py::TestDeviceNoiseInstallation::test_gate_depolar_installed PASSED [  7%]
tests/integration/test_nsm_parameter_enforcement.py::TestDeviceNoiseInstallation::test_no_noise_without_flag PASSED [  7%]
tests/integration/test_parallel_protocol.py::TestParallelProtocolIntegration::test_parallel_generation_produces_valid_data PASSED [  7%]
tests/integration/test_parallel_protocol.py::TestParallelProtocolIntegration::test_sifting_compatibility PASSED [  7%]
tests/integration/test_parallel_protocol.py::TestParallelProtocolIntegration::test_qber_estimation_compatibility PASSED [  7%]
tests/integration/test_parallel_protocol.py::TestParallelProtocolIntegration::test_sequential_vs_parallel_statistical_equivalence PASSED [  8%]
tests/integration/test_parallel_protocol.py::TestParallelProtocolIntegration::test_different_noise_levels PASSED [  8%]
tests/integration/test_parallel_protocol.py::TestParallelProtocolIntegration::test_multiple_batch_sizes PASSED [  8%]
tests/integration/test_parallel_protocol.py::TestParallelProtocolIntegration::test_reproducibility_with_seeding PASSED [  8%]
tests/integration/test_protocol_wiring.py::TestYAMLInjection::test_baseline_yaml_creates_baseline_strategy PASSED [  8%]
tests/integration/test_protocol_wiring.py::TestYAMLInjection::test_blind_yaml_creates_blind_strategy PASSED [  8%]
tests/integration/test_protocol_wiring.py::TestYAMLInjection::test_reconciliation_type_from_string PASSED [  8%]
tests/integration/test_protocol_wiring.py::TestYAMLInjection::test_config_from_yaml_file PASSED [  8%]
tests/integration/test_protocol_wiring.py::TestMessageSequenceEnforcement::test_baseline_sends_syndrome_with_qber PASSED [  8%]
tests/integration/test_protocol_wiring.py::TestMessageSequenceEnforcement::test_blind_sends_without_measured_qber PASSED [  8%]
tests/integration/test_protocol_wiring.py::TestMessageSequenceEnforcement::test_blind_reveal_message_format PASSED [  8%]
tests/integration/test_protocol_wiring.py::TestQBERDependencyCheck::test_baseline_without_qber_raises PASSED [  9%]
tests/integration/test_protocol_wiring.py::TestQBERDependencyCheck::test_blind_without_qber_succeeds PASSED [  9%]
tests/integration/test_protocol_wiring.py::TestQBERDependencyCheck::test_reconciliation_type_requires_qber_property PASSED [  9%]
tests/integration/test_protocol_wiring.py::TestConfigurationValidation::test_invalid_frame_size_raises PASSED [  9%]
tests/integration/test_protocol_wiring.py::TestConfigurationValidation::test_invalid_max_iterations_raises PASSED [  9%]
tests/integration/test_protocol_wiring.py::TestConfigurationValidation::test_config_defaults PASSED [  9%]
tests/performance/test_ldpc_decode_benchmark.py::test_ldpc_decode_benchmark PASSED [  9%]
tests/performance/test_parallel_speedup.py::TestParallelSpeedup::test_parallel_not_catastrophically_slower[5000] PASSED [  9%]
tests/performance/test_parallel_speedup.py::TestParallelSpeedup::test_parallel_not_catastrophically_slower[10000] PASSED [  9%]
tests/reconciliation/test_blind_manager.py::TestBlindConfig::test_default_values PASSED [  9%]
tests/reconciliation/test_blind_manager.py::TestBlindConfig::test_custom_values PASSED [  9%]
tests/reconciliation/test_blind_manager.py::TestBlindIterationState::test_state_fields PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestBlindIterationState::test_converged_state PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestBlindReconciliationManager::test_initialize_creates_state PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestBlindReconciliationManager::test_should_continue_first_iteration PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestBlindReconciliationManager::test_should_not_continue_when_converged PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestBlindReconciliationManager::test_should_not_continue_max_iterations PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestBlindReconciliationManager::test_advance_iteration_increments PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestBlindReconciliationManager::test_advance_iteration_converges PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestLLRModulation::test_build_llr_for_state_first_iteration PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestLLRModulation::test_llr_signs_match_bits PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestLLRModulation::test_delta_modulation_increases PASSED [ 10%]
tests/reconciliation/test_blind_manager.py::TestEdgeCases::test_empty_bits PASSED [ 11%]
tests/reconciliation/test_blind_manager.py::TestEdgeCases::test_single_iteration_success PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestBuildChannelLLR::test_low_qber_high_magnitude PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestBuildChannelLLR::test_high_qber_low_magnitude PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestBuildChannelLLR::test_sign_matches_bits PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestBuildChannelLLR::test_output_shape PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestSyndromeGuidedRefinement::test_identical_syndromes PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestSyndromeGuidedRefinement::test_increases_uncertainty PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestDecodeResult::test_success_result PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestDecodeResult::test_failed_result PASSED [ 11%]
tests/reconciliation/test_bp_decoder.py::TestBeliefPropagationDecoder::test_decode_noiseless ERROR [ 11%]
tests/reconciliation/test_bp_decoder.py::TestBeliefPropagationDecoder::test_decode_low_noise ERROR [ 12%]
tests/reconciliation/test_bp_decoder.py::TestBeliefPropagationDecoder::test_returns_decode_result ERROR [ 12%]
tests/reconciliation/test_compiled_matrix.py::test_compute_sparse_checksum_is_deterministic PASSED [ 12%]
tests/reconciliation/test_compiled_matrix.py::test_compile_parity_check_matrix_syndrome_matches_sparse_matmul PASSED [ 12%]
tests/reconciliation/test_compiled_matrix.py::test_count_syndrome_errors_zero_when_matching_target PASSED [ 12%]
tests/reconciliation/test_compiled_matrix.py::test_count_syndrome_errors_positive_when_target_flipped PASSED [ 12%]
tests/reconciliation/test_compiled_matrix.py::test_compiled_cache_roundtrip_and_checksum_mismatch PASSED [ 12%]
tests/reconciliation/test_compiled_matrix.py::test_compute_syndrome_rejects_wrong_length PASSED [ 12%]
tests/reconciliation/test_contracts.py::test_syndrome_linearity ERROR    [ 12%]
tests/reconciliation/test_contracts.py::test_decoder_converged_implies_syndrome_match ERROR [ 12%]
tests/reconciliation/test_contracts.py::test_edge_qber_does_not_crash ERROR [ 12%]
tests/reconciliation/test_contracts.py::test_leakage_budget_exceeded_aborts ERROR [ 13%]
tests/reconciliation/test_factory.py::TestReconciliationType::test_req_fac_001_from_string_accepts_variants PASSED [ 13%]
tests/reconciliation/test_factory.py::TestReconciliationType::test_req_fac_002_from_string_rejects_unknown PASSED [ 13%]
tests/reconciliation/test_factory.py::TestReconciliationConfig::test_req_fac_010_post_init_rejects_invalid_bounds PASSED [ 13%]
tests/reconciliation/test_factory.py::TestReconciliationConfig::test_req_fac_011_from_dict_maps_type_and_preserves_extras PASSED [ 13%]
tests/reconciliation/test_factory.py::TestReconciliationConfig::test_req_fac_012_requires_and_skips_match_type PASSED [ 13%]
tests/reconciliation/test_factory.py::TestCreateReconciler::test_req_fac_020_baseline_selected_and_requires_qber PASSED [ 13%]
tests/reconciliation/test_factory.py::TestCreateReconciler::test_req_fac_021_interactive_raises_not_implemented PASSED [ 13%]
tests/reconciliation/test_factory.py::TestCreateReconciler::test_req_fac_022_blind_selected_and_metadata_says_no_qber_required PASSED [ 13%]
tests/reconciliation/test_factory.py::TestCreateReconciler::test_req_fac_023_channel_profile_argument_is_accepted PASSED [ 13%]
tests/reconciliation/test_factory.py::TestYamlHelpers::test_req_fac_040_from_yaml_file_loads_reconciliation_section PASSED [ 13%]
tests/reconciliation/test_factory.py::TestYamlHelpers::test_req_fac_041_create_reconciler_from_yaml_matches_manual PASSED [ 14%]
tests/reconciliation/test_factory.py::TestBlindGetOrchestratorWiring::test_req_fac_030_get_orchestrator_wiring_uses_matrix_path_and_config_fields PASSED [ 14%]
tests/reconciliation/test_factory_integration.py::test_fint_001_blind_factory_reconciler_runs_one_block_when_assets_exist FAILED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestHashComputation::test_deterministic PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestHashComputation::test_different_inputs_different_hashes PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestHashComputation::test_output_length PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestHashComputation::test_empty_input PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestHashVerification::test_matching_blocks_verify PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestHashVerification::test_mismatched_blocks_fail PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestHashVerification::test_single_bit_difference PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestCollisionProbability::test_collision_rate_empirical PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestDifferentHashSizes::test_output_bits_respected[32] PASSED [ 14%]
tests/reconciliation/test_hash_verifier.py::TestDifferentHashSizes::test_output_bits_respected[40] PASSED [ 15%]
tests/reconciliation/test_hash_verifier.py::TestDifferentHashSizes::test_output_bits_respected[50] PASSED [ 15%]
tests/reconciliation/test_hash_verifier.py::TestDifferentHashSizes::test_output_bits_respected[60] PASSED [ 15%]
tests/reconciliation/test_hash_verifier.py::TestDifferentHashSizes::test_default_50_bits PASSED [ 15%]
tests/reconciliation/test_integration.py::TestEncoderDecoderIntegration::test_encode_decode_noiseless ERROR [ 15%]
tests/reconciliation/test_integration.py::TestEncoderDecoderIntegration::test_encode_decode_with_noise ERROR [ 15%]
tests/reconciliation/test_integration.py::TestHashVerification::test_hash_after_decode ERROR [ 15%]
tests/reconciliation/test_integration.py::TestRateSelectionFlow::test_rate_selection_guides_matrix_choice ERROR [ 15%]
tests/reconciliation/test_integration.py::TestLeakageTracking::test_leakage_from_multiple_blocks PASSED [ 15%]
tests/reconciliation/test_integration.py::TestLeakageTracking::test_high_qber_triggers_abort PASSED [ 15%]
tests/reconciliation/test_integration.py::TestOrchestratorIntegration::test_orchestrator_initialization ERROR [ 15%]
tests/reconciliation/test_integration.py::TestOrchestratorIntegration::test_single_block_reconciliation ERROR [ 16%]
tests/reconciliation/test_integration.py::TestHighRatePatternBased::test_high_rate_with_pattern_rate_0_8 ERROR [ 16%]
tests/reconciliation/test_integration.py::TestHighRatePatternBased::test_high_rate_stress_rate_0_9 ERROR [ 16%]
tests/reconciliation/test_integration.py::TestErrorScenarios::test_hash_mismatch_detected PASSED [ 16%]
tests/reconciliation/test_integration.py::TestErrorScenarios::test_leakage_cap_exceeded PASSED [ 16%]
tests/reconciliation/test_leakage_tracker.py::TestComputeSafetyCap::test_safety_cap_formula PASSED [ 16%]
tests/reconciliation/test_leakage_tracker.py::TestComputeSafetyCap::test_cap_decreases_with_qber PASSED [ 16%]
tests/reconciliation/test_leakage_tracker.py::TestComputeSafetyCap::test_cap_scales_with_sifted_bits PASSED [ 16%]
tests/reconciliation/test_leakage_tracker.py::TestComputeSafetyCap::test_cap_with_epsilon PASSED [ 16%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageRecord::test_record_fields PASSED [ 16%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageRecord::test_total_leakage PASSED [ 16%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTracker::test_initial_leakage_zero PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTracker::test_record_block_accumulates PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTracker::test_check_safety_under_cap PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTracker::test_check_safety_over_cap PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTracker::test_should_abort_when_exceeded PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTracker::test_remaining_budget PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTracker::test_records_maintained PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTrackerEdgeCases::test_zero_cap_always_aborts PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTrackerEdgeCases::test_large_single_block PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestLeakageTrackerEdgeCases::test_blind_iteration_tracking PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestCircuitBreaker::test_one_bit_too_many PASSED [ 17%]
tests/reconciliation/test_leakage_tracker.py::TestCircuitBreaker::test_exactly_at_cap PASSED [ 18%]
tests/reconciliation/test_leakage_tracker.py::TestCircuitBreaker::test_blind_iteration_accumulation PASSED [ 18%]
tests/reconciliation/test_leakage_tracker.py::TestCircuitBreaker::test_reveal_triggers_circuit_breaker PASSED [ 18%]
tests/reconciliation/test_leakage_tracker.py::TestCircuitBreaker::test_abort_on_exceed_false_logs_only PASSED [ 18%]
tests/reconciliation/test_leakage_tracker.py::TestCircuitBreaker::test_multiple_blocks_accumulate PASSED [ 18%]
tests/reconciliation/test_leakage_tracker.py::TestCircuitBreaker::test_shortening_bits_counted PASSED [ 18%]
tests/reconciliation/test_leakage_tracker.py::TestCircuitBreaker::test_remaining_budget_negative_when_exceeded PASSED [ 18%]
tests/reconciliation/test_matrix_manager_contracts.py::test_from_directory_missing_dir_raises PASSED [ 18%]
tests/reconciliation/test_matrix_manager_contracts.py::test_from_directory_missing_matrix_file_raises PASSED [ 18%]
tests/reconciliation/test_matrix_manager_contracts.py::test_from_directory_loads_small_pool_and_caches_compiled PASSED [ 18%]
tests/reconciliation/test_matrix_manager_contracts.py::test_get_matrix_unknown_rate_raises PASSED [ 18%]
tests/reconciliation/test_matrix_manager_contracts.py::test_verify_checksum_matches_local PASSED [ 19%]
tests/reconciliation/test_matrix_manager_contracts.py::test_write_compiled_caches_writes_sidecar PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_encode_small_known_pattern PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_fuzzing_prime_lengths[7] PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_fuzzing_prime_lengths[61] PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_fuzzing_prime_lengths[64] PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_fuzzing_prime_lengths[65] PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_fuzzing_prime_lengths[127] PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_fuzzing_prime_lengths[4099] PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_all_zeros_frame PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestBitPackingFuzzing::test_all_ones_frame PASSED [ 19%]
tests/reconciliation/test_numba_kernels.py::TestVirtualGraphTopology::test_csr_structure PASSED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestVirtualGraphTopology::test_decoder_sees_correct_graph PASSED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestVirtualGraphTopology::test_decoder_known_erasure_pattern PASSED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestFreezeOptimization::test_frozen_bits_unchanged PASSED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestFreezeOptimization::test_nonfrozen_bits_evolve PASSED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestRNGDeterminism::test_xorshift_determinism SKIPPED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestGraphOperations::test_add_remove_edge PASSED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestBFSReachability::test_simple_graph PASSED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestACEComputation::test_ace_isolated_node PASSED [ 20%]
tests/reconciliation/test_numba_kernels.py::TestACEComputation::test_ace_well_connected PASSED [ 20%]
tests/reconciliation/test_orchestrator_multiblock.py::test_orchestrator_reconcile_key_multiblock_happy_path ERROR [ 20%]
tests/reconciliation/test_puncture_patterns.py::TestUntaintedProperty::test_untainted_property_holds ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestUntaintedProperty::test_forced_puncturing_is_minority ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestPatternDeterminism::test_same_seed_same_pattern ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestPatternDeterminism::test_different_seed_different_pattern ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestRecoverability::test_1step_recoverable_count ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestPatternProperties::test_pattern_size_matches_frame_size ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestPatternProperties::test_puncture_count_matches_rate_difference ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestPatternProperties::test_pattern_values_are_binary ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestEdgeCases::test_target_rate_must_exceed_mother_rate ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestEdgeCases::test_target_rate_equal_to_mother_rate ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestPatternStatistics::test_pattern_coverage_increases_with_rate ERROR [ 21%]
tests/reconciliation/test_puncture_patterns.py::TestPatternStatistics::test_forced_puncturing_increases_at_high_rates ERROR [ 21%]
tests/reconciliation/test_rate_selector.py::TestBinaryEntropy::test_entropy_zero PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestBinaryEntropy::test_entropy_one PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestBinaryEntropy::test_entropy_half PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestBinaryEntropy::test_entropy_symmetric PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestBinaryEntropy::test_entropy_typical_qber PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestRateSelection::test_low_qber_reliable_rate PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestRateSelection::test_moderate_qber_reliable_rate PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestRateSelection::test_high_qber_low_rate PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestRateSelection::test_reliability_over_efficiency PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestRateSelection::test_qber_rate_mapping[0.01-0.5] PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestRateSelection::test_qber_rate_mapping[0.03-0.5] PASSED [ 22%]
tests/reconciliation/test_rate_selector.py::TestRateSelection::test_qber_rate_mapping[0.06-0.5] PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestRateSelection::test_qber_rate_mapping[0.09-0.5] PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestShortening::test_no_shortening_low_qber PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestShortening::test_shortening_fits_frame PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestPuncturing::test_no_puncturing_same_rate PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestPuncturing::test_no_puncturing_lower_target PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestPuncturing::test_puncturing_higher_target PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestRateSelectionWithParameters::test_returns_rate_selection PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestRateSelectionWithParameters::test_includes_syndrome_length PASSED [ 23%]
tests/reconciliation/test_rate_selector.py::TestRateSelectionWithParameters::test_efficiency_computed PASSED [ 23%]
tests/reconciliation/strategies/test_baseline.py::TestQBERToRateMapping::test_low_qber_selects_high_rate PASSED [ 23%]
tests/reconciliation/strategies/test_baseline.py::TestQBERToRateMapping::test_high_qber_selects_low_rate PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestQBERToRateMapping::test_medium_qber_selects_medium_rate PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestQBERToRateMapping::test_rate_selection_formula PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestQBERToRateMapping::test_zero_qber_fallback PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestExactLeakageAccounting::test_syndrome_leakage_is_mother_rate PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestExactLeakageAccounting::test_block_result_leakage PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestExactLeakageAccounting::test_total_leakage_formula PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestHashFailureHandling::test_hash_mismatch_returns_unverified PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestHashFailureHandling::test_failed_hash_still_counts_leakage PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestHashFailureHandling::test_bob_decoder_failure PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestBobBaseline::test_bob_receives_and_decodes PASSED [ 24%]
tests/reconciliation/strategies/test_baseline.py::TestContextRequirements::test_requires_qber_estimation_property PASSED [ 25%]
tests/reconciliation/strategies/test_baseline.py::TestContextRequirements::test_context_without_qber_raises PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestSyndromeReuse::test_encode_called_once_on_success PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestSyndromeReuse::test_encode_called_once_across_iterations PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestSyndromeReuse::test_syndrome_unchanged_across_reveals PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestHotStartPersistence::test_messages_passed_between_iterations PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestHotStartPersistence::test_frozen_mask_updated_with_reveals PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestRevelationOrder::test_reveal_indices_match_modulation_order PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestRevelationOrder::test_modulation_indices_are_deterministic PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestNSMGating::test_high_heuristic_starts_with_preshortening PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestNSMGating::test_low_heuristic_no_preshortening PASSED [ 25%]
tests/reconciliation/strategies/test_blind.py::TestNSMGating::test_no_heuristic_no_preshortening PASSED [ 26%]
tests/reconciliation/strategies/test_blind.py::TestBlindStrategyProperties::test_does_not_require_qber_estimation PASSED [ 26%]
tests/reconciliation/strategies/test_blind.py::TestBlindStrategyProperties::test_context_qber_for_blind_gating_fallback PASSED [ 26%]
tests/reconciliation/strategies/test_blind.py::TestBlindLeakageAccounting::test_syndrome_plus_reveals_counted PASSED [ 26%]
tests/reconciliation/strategies/test_blind.py::TestBlindLeakageAccounting::test_block_result_revealed_leakage PASSED [ 26%]
tests/security/test_output_quality.py::TestKeyRandomness::test_chi_square_uniformity PASSED [ 26%]
tests/security/test_output_quality.py::TestKeyRandomness::test_byte_uniformity PASSED [ 26%]
tests/security/test_output_quality.py::TestKeyRandomness::test_serial_correlation PASSED [ 26%]
tests/security/test_output_quality.py::TestKeyRandomness::test_runs_test PASSED [ 26%]
tests/security/test_output_quality.py::TestObliviousness::test_alice_choice_indistinguishability PASSED [ 26%]
tests/security/test_output_quality.py::TestObliviousness::test_bob_message_privacy PASSED [ 26%]
tests/security/test_output_quality.py::TestNSMSecurityConditions::test_security_condition_calculation PASSED [ 27%]
tests/security/test_output_quality.py::TestNSMSecurityConditions::test_entropy_rate_bound PASSED [ 27%]
tests/security/test_output_quality.py::TestNSMSecurityConditions::test_timing_parameter_security PASSED [ 27%]
tests/security/test_output_quality.py::TestStatisticalHelpers::test_hash_collision_probability PASSED [ 27%]
tests/security/test_output_quality.py::TestStatisticalHelpers::test_qber_estimation_confidence_interval PASSED [ 27%]
tests/security/test_output_quality.py::TestStatisticalHelpers::test_syndrome_weight_statistics PASSED [ 27%]
tests/security/test_parallel_iid.py::TestParallelIIDPreservation::test_basis_balance PASSED [ 27%]
tests/security/test_parallel_iid.py::TestParallelIIDPreservation::test_basis_approx_independence PASSED [ 27%]
tests/security/test_parallel_iid.py::TestParallelIIDPreservation::test_batch_boundary_mixing_outcomes PASSED [ 27%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_init_default PASSED [ 27%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_init_custom_r PASSED [ 27%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_init_invalid_r_raises PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_set_storage_noise_r PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_set_invalid_r_raises PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_max_bound_entropy_rate_in_range[0.0] PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_max_bound_entropy_rate_in_range[0.1] PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_max_bound_entropy_rate_in_range[0.25] PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_max_bound_entropy_rate_in_range[0.5] PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_max_bound_entropy_rate_in_range[0.75] PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_max_bound_entropy_rate_in_range[1.0] PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_max_bound_at_r_zero PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_max_bound_at_r_one PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_dominant_bound_low_r PASSED [ 28%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_dominant_bound_high_r PASSED [ 29%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_dupuis_konig_bound PASSED [ 29%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_virtual_erasure_bound PASSED [ 29%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_compute_total_entropy_no_leakage PASSED [ 29%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_compute_total_entropy_with_leakage PASSED [ 29%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_compute_total_entropy_depleted PASSED [ 29%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_crossover_point PASSED [ 29%]
tests/test_amplification/test_entropy.py::TestNSMEntropyCalculator::test_gamma_function PASSED [ 29%]
tests/test_amplification/test_entropy.py::TestEntropyResult::test_entropy_result_creation PASSED [ 29%]
tests/test_amplification/test_formatter.py::TestAliceOTOutput::test_creation PASSED [ 29%]
tests/test_amplification/test_formatter.py::TestBobOTOutput::test_creation PASSED [ 29%]
tests/test_amplification/test_formatter.py::TestOTOutputFormatter::test_init_basic PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTOutputFormatter::test_init_with_seeds PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTOutputFormatter::test_init_invalid_length_raises PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTOutputFormatter::test_compute_alice_keys PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTOutputFormatter::test_compute_alice_keys_input_too_short PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTOutputFormatter::test_compute_bob_key_choice_0 PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTOutputFormatter::test_compute_bob_key_choice_1 PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTOutputFormatter::test_compute_bob_key_invalid_choice_raises PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTCorrectness::test_ot_correctness_choice_0 PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTCorrectness::test_ot_correctness_choice_1 PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTCorrectness::test_format_final_output_success PASSED [ 30%]
tests/test_amplification/test_formatter.py::TestOTCorrectness::test_format_final_output_ot_violation_raises PASSED [ 31%]
tests/test_amplification/test_formatter.py::TestDeterministicSeeds::test_same_seed_same_output PASSED [ 31%]
tests/test_amplification/test_formatter.py::TestDeterministicSeeds::test_different_seeds_different_output PASSED [ 31%]
tests/test_amplification/test_formatter_phase_contracts_integration.py::test_formatter_outputs_satisfy_oblivious_transfer_output_contract_choice_0 PASSED [ 31%]
tests/test_amplification/test_formatter_phase_contracts_integration.py::test_oblivious_transfer_output_detects_tampering_after_formatting PASSED [ 31%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_compute_final_length_positive PASSED [ 31%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_compute_final_length_death_valley PASSED [ 31%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_compute_final_length_monotonic PASSED [ 31%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_compute_final_length_leakage_reduces PASSED [ 31%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_compute_detailed PASSED [ 31%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_compute_detailed_efficiency PASSED [ 31%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_compute_detailed_is_viable PASSED [ 32%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_invalid_reconciled_length_raises PASSED [ 32%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_negative_leakage_raises PASSED [ 32%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_minimum_input_length PASSED [ 32%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_minimum_input_length_zero_entropy PASSED [ 32%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_security_penalty_property PASSED [ 32%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_epsilon_sec_property PASSED [ 32%]
tests/test_amplification/test_key_length.py::TestSecureKeyLengthCalculator::test_invalid_epsilon_raises PASSED [ 32%]
tests/test_amplification/test_key_length.py::TestKeyLengthResult::test_key_length_result_creation PASSED [ 32%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_init_basic PASSED [ 32%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_init_with_seed PASSED [ 32%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_init_invalid_input_raises PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_init_invalid_output_raises PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_init_output_exceeds_input_raises PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_hash_output_length PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_hash_output_binary PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_hash_wrong_input_length_raises PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_hash_deterministic PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_different_seeds_different_outputs PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_different_inputs_different_outputs PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_fft_direct_equivalence PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_get_matrix PASSED [ 33%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_random_bits_property PASSED [ 34%]
tests/test_amplification/test_toeplitz.py::TestToeplitzHasher::test_generate_seed_static PASSED [ 34%]
tests/test_amplification/test_toeplitz.py::TestToeplitzProperties::test_2_universal_property PASSED [ 34%]
tests/test_amplification/test_toeplitz.py::TestToeplitzProperties::test_output_distribution PASSED [ 34%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_select_single_returns_valid_basis PASSED [ 34%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_select_batch_returns_correct_size PASSED [ 34%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_select_batch_returns_only_valid_values PASSED [ 34%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_select_batch_approximately_uniform PASSED [ 34%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_select_batch_invalid_n_raises PASSED [ 34%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_seeded_selector_reproducible PASSED [ 34%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_different_seeds_different_results PASSED [ 34%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_select_weighted_uniform PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_select_weighted_biased PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisSelector::test_select_weighted_invalid_p_raises PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisHelpers::test_basis_to_string PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisHelpers::test_bases_match_same PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisHelpers::test_bases_match_different PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisHelpers::test_compute_matching_mask_all_match PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisHelpers::test_compute_matching_mask_none_match PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisHelpers::test_compute_matching_mask_partial PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisHelpers::test_compute_matching_mask_length_mismatch_raises PASSED [ 35%]
tests/test_quantum/test_basis.py::TestBasisConstants::test_basis_constants_values PASSED [ 35%]
tests/test_quantum/test_basis_additional.py::test_basis_selector_seeded_is_deterministic PASSED [ 35%]
tests/test_quantum/test_basis_additional.py::test_basis_selector_generate_rejects_invalid_num_bases PASSED [ 36%]
tests/test_quantum/test_basis_additional.py::test_basis_selector_generate_values_are_bits PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchConfig::test_default_config PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchConfig::test_custom_config PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchConfig::test_invalid_pairs_per_batch_raises PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchConfig::test_invalid_max_batches_raises PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_configure_returns_batch_count PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_configure_exact_multiple PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_configure_exceeds_max_raises PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_configure_invalid_total_raises PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_start_batch PASSED [ 36%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_start_batch_last_partial PASSED [ 37%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_complete_batch PASSED [ 37%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_iterate_batches PASSED [ 37%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_iterate_batches_unconfigured_raises PASSED [ 37%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_get_aggregated_results PASSED [ 37%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_get_aggregated_no_batches_raises PASSED [ 37%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_reset PASSED [ 37%]
tests/test_quantum/test_batching.py::TestBatchingManager::test_estimate_memory_usage PASSED [ 37%]
tests/test_quantum/test_batching.py::TestBatchState::test_batch_states PASSED [ 37%]
tests/test_quantum/test_epr.py::TestEPRGenerationConfig::test_default_config PASSED [ 37%]
tests/test_quantum/test_epr.py::TestEPRGenerationConfig::test_custom_config PASSED [ 37%]
tests/test_quantum/test_epr.py::TestEPRGenerator::test_initial_state PASSED [ 38%]
tests/test_quantum/test_epr.py::TestEPRGenerator::test_generate_batch_sync PASSED [ 38%]
tests/test_quantum/test_epr.py::TestEPRGenerator::test_generate_multiple_batches PASSED [ 38%]
tests/test_quantum/test_epr.py::TestEPRGenerator::test_reset_counters PASSED [ 38%]
tests/test_quantum/test_epr.py::TestEPRGenerator::test_custom_config PASSED [ 38%]
tests/test_quantum/test_epr.py::TestEPRBatch::test_epr_batch_creation PASSED [ 38%]
tests/test_quantum/test_epr_generator_generator_mode.py::test_generate_batch_retries_then_succeeds_without_context PASSED [ 38%]
tests/test_quantum/test_epr_generator_generator_mode.py::test_generate_batch_raises_after_all_retries PASSED [ 38%]
tests/test_quantum/test_epr_generator_generator_mode.py::test_generate_batch_uses_context_time_when_available PASSED [ 38%]
tests/test_quantum/test_epr_generator_generator_mode.py::test_generate_batch_wraps_noniterable_qubit_ref PASSED [ 38%]
tests/test_quantum/test_factory.py::TestCaligoConfig::test_default_values PASSED [ 38%]
tests/test_quantum/test_factory.py::TestCaligoConfig::test_custom_parallel_config PASSED [ 39%]
tests/test_quantum/test_factory.py::TestCaligoConfig::test_custom_network_config PASSED [ 39%]
tests/test_quantum/test_factory.py::TestCaligoConfig::test_custom_security_epsilon PASSED [ 39%]
tests/test_quantum/test_factory.py::TestEPRGenerationFactory::test_create_sequential_strategy_when_disabled PASSED [ 39%]
tests/test_quantum/test_factory.py::TestEPRGenerationFactory::test_create_parallel_strategy_when_enabled PASSED [ 39%]
tests/test_quantum/test_factory.py::TestEPRGenerationFactory::test_create_sequential_explicit PASSED [ 39%]
tests/test_quantum/test_factory.py::TestEPRGenerationFactory::test_create_parallel_explicit PASSED [ 39%]
tests/test_quantum/test_factory.py::TestEPRGenerationFactory::test_factory_passes_network_config PASSED [ 39%]
tests/test_quantum/test_factory.py::TestSequentialEPRStrategy::test_generate_returns_four_lists PASSED [ 39%]
tests/test_quantum/test_factory.py::TestSequentialEPRStrategy::test_generate_correct_count PASSED [ 39%]
tests/test_quantum/test_factory.py::TestSequentialEPRStrategy::test_generate_binary_values PASSED [ 39%]
tests/test_quantum/test_factory.py::TestSequentialEPRStrategy::test_generate_invalid_pairs_raises PASSED [ 40%]
tests/test_quantum/test_factory.py::TestSequentialEPRStrategy::test_generate_noise_affects_qber PASSED [ 40%]
tests/test_quantum/test_factory.py::TestParallelEPRStrategy::test_generate_returns_four_lists PASSED [ 40%]
tests/test_quantum/test_factory.py::TestParallelEPRStrategy::test_generate_correct_count PASSED [ 40%]
tests/test_quantum/test_factory.py::TestParallelEPRStrategy::test_shutdown_releases_resources PASSED [ 40%]
tests/test_quantum/test_factory.py::TestParallelEPRStrategy::test_context_manager PASSED [ 40%]
tests/test_quantum/test_factory.py::TestStrategyInterface::test_sequential_implements_protocol PASSED [ 40%]
tests/test_quantum/test_factory.py::TestStrategyInterface::test_parallel_implements_protocol PASSED [ 40%]
tests/test_quantum/test_factory.py::TestStrategyInterface::test_strategies_have_same_signature PASSED [ 40%]
tests/test_quantum/test_factory.py::TestStrategyInterface::test_polymorphic_usage PASSED [ 40%]
tests/test_quantum/test_factory.py::TestFactoryWithStrategies::test_factory_sequential_generates_data PASSED [ 40%]
tests/test_quantum/test_factory.py::TestFactoryWithStrategies::test_factory_parallel_generates_data PASSED [ 41%]
tests/test_quantum/test_factory.py::TestFactoryWithStrategies::test_end_to_end_workflow PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementBuffer::test_initial_state PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementBuffer::test_add_single_outcome PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementBuffer::test_add_batch PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementBuffer::test_buffer_grows_automatically PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementBuffer::test_clear_resets_count PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementBuffer::test_get_batch_returns_copies PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementExecutor::test_measure_qubit_sync PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementExecutor::test_measure_qubit_sync_predetermined PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementExecutor::test_measure_batch_sync PASSED [ 41%]
tests/test_quantum/test_measurement.py::TestMeasurementExecutor::test_measure_batch_sync_predetermined PASSED [ 42%]
tests/test_quantum/test_measurement.py::TestMeasurementExecutor::test_get_results PASSED [ 42%]
tests/test_quantum/test_measurement.py::TestMeasurementExecutor::test_clear_resets_state PASSED [ 42%]
tests/test_quantum/test_measurement.py::TestMeasurementResult::test_measurement_result_creation PASSED [ 42%]
tests/test_quantum/test_measurement_generator_fallback.py::test_measure_qubit_falls_back_when_netqasm_unavailable PASSED [ 42%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_default_values PASSED [ 42%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_default_workers_cpu_based PASSED [ 42%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_custom_workers PASSED [ 42%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_custom_pairs_per_batch PASSED [ 42%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_enabled_flag PASSED [ 42%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_isolation_level_process PASSED [ 42%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_isolation_level_thread PASSED [ 42%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_invalid_isolation_level_raises PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_invalid_workers_raises PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_invalid_pairs_per_batch_raises PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_invalid_timeout_raises PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestParallelEPRConfig::test_shuffle_results_toggle PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestEPRWorkerResult::test_construction PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestEPRWorkerResult::test_generation_time_default PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestEPRWorkerResult::test_generation_time_custom PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_init PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_batch_count_calculation PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_generate_parallel_invalid_pairs_raises PASSED [ 43%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_generate_parallel_returns_correct_count PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_generate_parallel_binary_values PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_generate_parallel_basis_distribution PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_shutdown_idempotent PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_context_manager PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestParallelEPROrchestrator::test_result_shuffling PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestWorkerFunction::test_worker_returns_correct_structure PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestWorkerFunction::test_worker_different_batches_different_ids PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestWorkerFunction::test_worker_respects_num_pairs PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestWorkerFunction::test_worker_noise_affects_qber PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestParallelIntegration::test_large_scale_generation PASSED [ 44%]
tests/test_quantum/test_parallel.py::TestParallelIntegration::test_multiple_generations_same_orchestrator PASSED [ 45%]
tests/test_scripts/test_generate_ldpc_matrices_args.py::test_generate_all_creates_files PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestSaturationDetection::test_untainted_set_computation PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestSaturationDetection::test_regime_transition_detection PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestNestingProperty::test_pattern_nesting PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestNestingProperty::test_blind_protocol_nesting PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestACEScoreValidation::test_ace_score_computation PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestACEScoreValidation::test_ace_guided_selection_prefers_high_ace PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestMotherCodeLoading::test_load_valid_mother_matrix PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestMotherCodeLoading::test_corrupted_matrix_fails_fast PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestMotherCodeLoading::test_wrong_rate_matrix_fails PASSED [ 45%]
tests/test_scripts/test_hybrid_patterns.py::TestMotherCodeLoading::test_missing_patterns_fails PASSED [ 46%]
tests/test_scripts/test_hybrid_patterns.py::TestMotherCodeLoading::test_pattern_rate_lookup PASSED [ 46%]
tests/test_scripts/test_hybrid_patterns.py::TestMotherCodeLoading::test_modulation_indices PASSED [ 46%]
tests/test_scripts/test_hybrid_patterns.py::TestMotherCodeLoading::test_compiled_topology PASSED [ 46%]
tests/test_scripts/test_hybrid_patterns.py::TestHybridPatternGeneration::test_generate_pattern_range PASSED [ 46%]
tests/test_scripts/test_hybrid_patterns.py::TestHybridPatternGeneration::test_n2_size_ordering PASSED [ 46%]
tests/test_scripts/test_peg_degree_distribution.py::test_degree_distribution_normalizes_and_converts PASSED [ 46%]
tests/test_scripts/test_peg_degree_distribution.py::test_degree_distribution_invalid_raises[degrees0-probs0] PASSED [ 46%]
tests/test_scripts/test_peg_degree_distribution.py::test_degree_distribution_invalid_raises[degrees1-probs1] PASSED [ 46%]
tests/test_scripts/test_peg_degree_distribution.py::test_degree_distribution_invalid_raises[degrees2-probs2] PASSED [ 46%]
tests/test_scripts/test_peg_degree_distribution.py::test_degree_distribution_zero_sum_raises PASSED [ 46%]
tests/test_scripts/test_peg_numba_capacity_growth.py::test_peg_numba_retries_on_capacity_exceeded PASSED [ 47%]
tests/test_security/test_bounds.py::TestConstants::test_qber_conservative_threshold PASSED [ 47%]
tests/test_security/test_bounds.py::TestConstants::test_qber_absolute_threshold PASSED [ 47%]
tests/test_security/test_bounds.py::TestConstants::test_r_tilde PASSED   [ 47%]
tests/test_security/test_bounds.py::TestConstants::test_r_crossover PASSED [ 47%]
tests/test_security/test_bounds.py::TestConstants::test_default_epsilon PASSED [ 47%]
tests/test_security/test_bounds.py::TestValidation::test_validate_r_valid_range PASSED [ 47%]
tests/test_security/test_bounds.py::TestValidation::test_validate_r_below_zero PASSED [ 47%]
tests/test_security/test_bounds.py::TestValidation::test_validate_r_above_one PASSED [ 47%]
tests/test_security/test_bounds.py::TestValidation::test_validate_nu_valid_range PASSED [ 47%]
tests/test_security/test_bounds.py::TestValidation::test_validate_nu_invalid PASSED [ 47%]
tests/test_security/test_bounds.py::TestGFunction::test_g_at_zero PASSED [ 48%]
tests/test_security/test_bounds.py::TestGFunction::test_g_at_half PASSED [ 48%]
tests/test_security/test_bounds.py::TestGFunction::test_g_monotonic PASSED [ 48%]
tests/test_security/test_bounds.py::TestGammaFunction::test_gamma_above_half_identity PASSED [ 48%]
tests/test_security/test_bounds.py::TestGammaFunction::test_gamma_at_half PASSED [ 48%]
tests/test_security/test_bounds.py::TestGammaFunction::test_gamma_below_half_valid PASSED [ 48%]
tests/test_security/test_bounds.py::TestGammaFunction::test_gamma_at_zero PASSED [ 48%]
tests/test_security/test_bounds.py::TestGammaFunction::test_gamma_negative PASSED [ 48%]
tests/test_security/test_bounds.py::TestCollisionEntropyRate::test_complete_depolarization PASSED [ 48%]
tests/test_security/test_bounds.py::TestCollisionEntropyRate::test_perfect_storage PASSED [ 48%]
tests/test_security/test_bounds.py::TestCollisionEntropyRate::test_intermediate_values PASSED [ 48%]
tests/test_security/test_bounds.py::TestCollisionEntropyRate::test_invalid_r PASSED [ 49%]
tests/test_security/test_bounds.py::TestCollisionEntropyRate::test_monotonic_decreasing PASSED [ 49%]
tests/test_security/test_bounds.py::TestDupuisKonigBound::test_complete_depolarization PASSED [ 49%]
tests/test_security/test_bounds.py::TestDupuisKonigBound::test_literature_values PASSED [ 49%]
tests/test_security/test_bounds.py::TestDupuisKonigBound::test_invalid_r PASSED [ 49%]
tests/test_security/test_bounds.py::TestLupoVirtualErasureBound::test_complete_depolarization PASSED [ 49%]
tests/test_security/test_bounds.py::TestLupoVirtualErasureBound::test_perfect_storage PASSED [ 49%]
tests/test_security/test_bounds.py::TestLupoVirtualErasureBound::test_erven_value PASSED [ 49%]
tests/test_security/test_bounds.py::TestLupoVirtualErasureBound::test_literature_values PASSED [ 49%]
tests/test_security/test_bounds.py::TestLupoVirtualErasureBound::test_invalid_r PASSED [ 49%]
tests/test_security/test_bounds.py::TestMaxBoundEntropy::test_selects_larger_bound PASSED [ 49%]
tests/test_security/test_bounds.py::TestMaxBoundEntropy::test_equals_max_of_bounds PASSED [ 50%]
tests/test_security/test_bounds.py::TestMaxBoundEntropy::test_crossover_point PASSED [ 50%]
tests/test_security/test_bounds.py::TestMaxBoundEntropy::test_dk_better_for_high_noise PASSED [ 50%]
tests/test_security/test_bounds.py::TestMaxBoundEntropy::test_lupo_better_for_low_noise PASSED [ 50%]
tests/test_security/test_bounds.py::TestMaxBoundEntropy::test_in_valid_range PASSED [ 50%]
tests/test_security/test_bounds.py::TestRationalAdversaryBound::test_capped_at_half PASSED [ 50%]
tests/test_security/test_bounds.py::TestRationalAdversaryBound::test_equals_half_for_high_noise PASSED [ 50%]
tests/test_security/test_bounds.py::TestRationalAdversaryBound::test_equals_max_bound_for_low_noise PASSED [ 50%]
tests/test_security/test_bounds.py::TestRationalAdversaryBound::test_erven_value PASSED [ 50%]
tests/test_security/test_bounds.py::TestRationalAdversaryBound::test_invalid_r PASSED [ 50%]
tests/test_security/test_bounds.py::TestBoundedStorageEntropy::test_full_storage_equals_max_bound PASSED [ 50%]
tests/test_security/test_bounds.py::TestBoundedStorageEntropy::test_no_storage_equals_half PASSED [ 50%]
tests/test_security/test_bounds.py::TestBoundedStorageEntropy::test_interpolation PASSED [ 51%]
tests/test_security/test_bounds.py::TestBoundedStorageEntropy::test_erven_experimental PASSED [ 51%]
tests/test_security/test_bounds.py::TestBoundedStorageEntropy::test_invalid_parameters PASSED [ 51%]
tests/test_security/test_bounds.py::TestStrongConverseExponent::test_below_capacity_zero PASSED [ 51%]
tests/test_security/test_bounds.py::TestStrongConverseExponent::test_above_capacity_positive PASSED [ 51%]
tests/test_security/test_bounds.py::TestStrongConverseExponent::test_perfect_storage_capacity_one PASSED [ 51%]
tests/test_security/test_bounds.py::TestStrongConverseExponent::test_complete_depolarization_capacity_zero PASSED [ 51%]
tests/test_security/test_bounds.py::TestStrongConverseExponent::test_invalid_r PASSED [ 51%]
tests/test_security/test_bounds.py::TestExtractableKeyRate::test_erven_positive_rate PASSED [ 51%]
tests/test_security/test_bounds.py::TestExtractableKeyRate::test_high_qber_negative_rate PASSED [ 51%]
tests/test_security/test_bounds.py::TestExtractableKeyRate::test_perfect_channel_maximum_rate PASSED [ 51%]
tests/test_security/test_bounds.py::TestExtractableKeyRate::test_invalid_parameters PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_max_bound_in_valid_range[0.0] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_max_bound_in_valid_range[0.1] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_max_bound_in_valid_range[0.25] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_max_bound_in_valid_range[0.5] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_max_bound_in_valid_range[0.75] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_max_bound_in_valid_range[0.9] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_max_bound_in_valid_range[1.0] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_rational_bound_in_valid_range[0.0] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_rational_bound_in_valid_range[0.25] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_rational_bound_in_valid_range[0.5] PASSED [ 52%]
tests/test_security/test_bounds.py::TestPropertyBased::test_rational_bound_in_valid_range[0.75] PASSED [ 53%]
tests/test_security/test_bounds.py::TestPropertyBased::test_rational_bound_in_valid_range[1.0] PASSED [ 53%]
tests/test_security/test_bounds.py::TestPropertyBased::test_bounded_storage_in_valid_range[0.5-0.0] PASSED [ 53%]
tests/test_security/test_bounds.py::TestPropertyBased::test_bounded_storage_in_valid_range[0.5-0.5] PASSED [ 53%]
tests/test_security/test_bounds.py::TestPropertyBased::test_bounded_storage_in_valid_range[0.5-1.0] PASSED [ 53%]
tests/test_security/test_bounds.py::TestPropertyBased::test_bounded_storage_in_valid_range[0.75-0.002] PASSED [ 53%]
tests/test_security/test_bounds.py::TestPropertyBased::test_bounded_storage_in_valid_range[0.9-0.1] PASSED [ 53%]
tests/test_security/test_feasibility.py::TestComputeExpectedQBER::test_perfect_channel PASSED [ 53%]
tests/test_security/test_feasibility.py::TestComputeExpectedQBER::test_fidelity_contribution PASSED [ 53%]
tests/test_security/test_feasibility.py::TestComputeExpectedQBER::test_intrinsic_error_contribution PASSED [ 53%]
tests/test_security/test_feasibility.py::TestComputeExpectedQBER::test_combined_contributions PASSED [ 53%]
tests/test_security/test_feasibility.py::TestComputeExpectedQBER::test_invalid_fidelity PASSED [ 54%]
tests/test_security/test_feasibility.py::TestComputeExpectedQBER::test_invalid_detection_efficiency PASSED [ 54%]
tests/test_security/test_feasibility.py::TestComputeExpectedQBER::test_clamped_to_half PASSED [ 54%]
tests/test_security/test_feasibility.py::TestComputeStorageCapacity::test_perfect_storage PASSED [ 54%]
tests/test_security/test_feasibility.py::TestComputeStorageCapacity::test_complete_depolarization PASSED [ 54%]
tests/test_security/test_feasibility.py::TestComputeStorageCapacity::test_intermediate_values PASSED [ 54%]
tests/test_security/test_feasibility.py::TestFeasibilityCheckerInit::test_valid_initialization PASSED [ 54%]
tests/test_security/test_feasibility.py::TestFeasibilityCheckerInit::test_invalid_storage_noise PASSED [ 54%]
tests/test_security/test_feasibility.py::TestFeasibilityCheckerInit::test_invalid_storage_rate PASSED [ 54%]
tests/test_security/test_feasibility.py::TestFeasibilityCheckerInit::test_invalid_qber PASSED [ 54%]
tests/test_security/test_feasibility.py::TestFeasibilityCheckerInit::test_invalid_security_parameter PASSED [ 54%]
tests/test_security/test_feasibility.py::TestFeasibilityCheckerFromNSMParameters::test_from_nsm_parameters PASSED [ 55%]
tests/test_security/test_feasibility.py::TestFeasibilityCheckerFromNSMParameters::test_qber_computed_from_params PASSED [ 55%]
tests/test_security/test_feasibility.py::TestQBERThresholdCheck::test_below_conservative_passes PASSED [ 55%]
tests/test_security/test_feasibility.py::TestQBERThresholdCheck::test_between_thresholds_warning PASSED [ 55%]
tests/test_security/test_feasibility.py::TestQBERThresholdCheck::test_above_absolute_fails PASSED [ 55%]
tests/test_security/test_feasibility.py::TestQBERThresholdCheck::test_raises_on_failure PASSED [ 55%]
tests/test_security/test_feasibility.py::TestQBERThresholdCheck::test_uses_expected_qber_default PASSED [ 55%]
tests/test_security/test_feasibility.py::TestStorageCapacityConstraint::test_erven_params_pass PASSED [ 55%]
tests/test_security/test_feasibility.py::TestStorageCapacityConstraint::test_high_storage_rate_fails PASSED [ 55%]
tests/test_security/test_feasibility.py::TestStorageCapacityConstraint::test_raises_on_failure PASSED [ 55%]
tests/test_security/test_feasibility.py::TestStorageCapacityConstraint::test_close_to_threshold_warning PASSED [ 55%]
tests/test_security/test_feasibility.py::TestStrictlyLessCondition::test_low_qber_passes PASSED [ 56%]
tests/test_security/test_feasibility.py::TestStrictlyLessCondition::test_high_qber_fails PASSED [ 56%]
tests/test_security/test_feasibility.py::TestStrictlyLessCondition::test_raises_on_failure PASSED [ 56%]
tests/test_security/test_feasibility.py::TestStrictlyLessCondition::test_tight_margin_warning PASSED [ 56%]
tests/test_security/test_feasibility.py::TestBatchSizeFeasibility::test_large_batch_passes PASSED [ 56%]
tests/test_security/test_feasibility.py::TestBatchSizeFeasibility::test_small_batch_may_fail PASSED [ 56%]
tests/test_security/test_feasibility.py::TestBatchSizeFeasibility::test_raises_on_failure PASSED [ 56%]
tests/test_security/test_feasibility.py::TestBatchSizeFeasibility::test_expected_length_in_result PASSED [ 56%]
tests/test_security/test_feasibility.py::TestComputeMinBatchSize::test_erven_params_reasonable_size PASSED [ 56%]
tests/test_security/test_feasibility.py::TestComputeMinBatchSize::test_larger_key_needs_larger_batch PASSED [ 56%]
tests/test_security/test_feasibility.py::TestRunPreflightChecks::test_erven_params_feasible PASSED [ 56%]
tests/test_security/test_feasibility.py::TestRunPreflightChecks::test_all_checks_executed PASSED [ 57%]
tests/test_security/test_feasibility.py::TestRunPreflightChecks::test_without_batch_size PASSED [ 57%]
tests/test_security/test_feasibility.py::TestRunPreflightChecks::test_fails_on_qber_violation PASSED [ 57%]
tests/test_security/test_feasibility.py::TestRunPreflightChecks::test_fails_on_storage_violation PASSED [ 57%]
tests/test_security/test_feasibility.py::TestRunPreflightChecks::test_report_contains_warnings PASSED [ 57%]
tests/test_security/test_feasibility.py::TestRunPreflightChecks::test_report_str_method PASSED [ 57%]
tests/test_security/test_feasibility.py::TestFeasibilityResult::test_result_attributes PASSED [ 57%]
tests/test_security/test_feasibility.py::TestPreflightReport::test_default_values PASSED [ 57%]
tests/test_security/test_feasibility.py::TestPreflightReport::test_str_representation PASSED [ 57%]
tests/test_security/test_feasibility.py::TestIntegrationScenarios::test_erven_experimental_setup PASSED [ 57%]
tests/test_security/test_feasibility.py::TestIntegrationScenarios::test_high_fidelity_low_noise PASSED [ 57%]
tests/test_security/test_feasibility.py::TestIntegrationScenarios::test_challenging_but_feasible PASSED [ 57%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_basic_computation PASSED [ 58%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_larger_test_sample_smaller_penalty PASSED [ 58%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_larger_key_fraction_smaller_penalty PASSED [ 58%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_realistic_values PASSED [ 58%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_small_sample_large_penalty PASSED [ 58%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_invalid_n PASSED [ 58%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_invalid_k PASSED [ 58%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_invalid_epsilon PASSED [ 58%]
tests/test_security/test_finite_key.py::TestComputeStatisticalFluctuation::test_scaling_with_epsilon PASSED [ 58%]
tests/test_security/test_finite_key.py::TestHoeffdingDetectionInterval::test_basic_interval PASSED [ 58%]
tests/test_security/test_finite_key.py::TestHoeffdingDetectionInterval::test_interval_symmetric PASSED [ 58%]
tests/test_security/test_finite_key.py::TestHoeffdingDetectionInterval::test_larger_n_tighter_interval PASSED [ 59%]
tests/test_security/test_finite_key.py::TestHoeffdingDetectionInterval::test_interval_clamped PASSED [ 59%]
tests/test_security/test_finite_key.py::TestHoeffdingDetectionInterval::test_realistic_detection_efficiency PASSED [ 59%]
tests/test_security/test_finite_key.py::TestHoeffdingDetectionInterval::test_invalid_n PASSED [ 59%]
tests/test_security/test_finite_key.py::TestHoeffdingDetectionInterval::test_invalid_p_expected PASSED [ 59%]
tests/test_security/test_finite_key.py::TestHoeffdingDetectionInterval::test_invalid_epsilon PASSED [ 59%]
tests/test_security/test_finite_key.py::TestHoeffdingCountInterval::test_basic_count PASSED [ 59%]
tests/test_security/test_finite_key.py::TestHoeffdingCountInterval::test_counts_are_integers PASSED [ 59%]
tests/test_security/test_finite_key.py::TestHoeffdingCountInterval::test_expected_in_interval PASSED [ 59%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_positive_key_length PASSED [ 59%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_high_qber_zero_length PASSED [ 59%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_small_n_zero_length PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_larger_n_larger_key PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_lower_qber_larger_key PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_better_ec_efficiency_larger_key PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_erven_experimental_parameters PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_invalid_n PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_invalid_qber PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_invalid_storage_params PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_invalid_ec_efficiency PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_invalid_epsilon PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeFiniteKeyLength::test_invalid_test_fraction PASSED [ 60%]
tests/test_security/test_finite_key.py::TestComputeOptimalTestFraction::test_returns_valid_fraction PASSED [ 61%]
tests/test_security/test_finite_key.py::TestComputeOptimalTestFraction::test_typical_range PASSED [ 61%]
tests/test_security/test_finite_key.py::TestComputeOptimalTestFraction::test_optimizes_key_length PASSED [ 61%]
tests/test_security/test_finite_key.py::TestComputeMinNForKeyLength::test_finds_minimum_n PASSED [ 61%]
tests/test_security/test_finite_key.py::TestComputeMinNForKeyLength::test_larger_target_needs_larger_n PASSED [ 61%]
tests/test_security/test_finite_key.py::TestComputeMinNForKeyLength::test_higher_qber_needs_larger_n PASSED [ 61%]
tests/test_security/test_finite_key.py::TestParameterizedScenarios::test_mu_scaling[100000-5000-expected_range0] PASSED [ 61%]
tests/test_security/test_finite_key.py::TestParameterizedScenarios::test_mu_scaling[10000-500-expected_range1] PASSED [ 61%]
tests/test_security/test_finite_key.py::TestParameterizedScenarios::test_mu_scaling[1000-100-expected_range2] PASSED [ 61%]
tests/test_security/test_finite_key.py::TestParameterizedScenarios::test_key_length_scenarios[0.75-0.002-0.02-500000-True] PASSED [ 61%]
tests/test_security/test_finite_key.py::TestParameterizedScenarios::test_key_length_scenarios[0.5-0.01-0.03-500000-True] PASSED [ 61%]
tests/test_security/test_finite_key.py::TestParameterizedScenarios::test_key_length_scenarios[0.9-0.1-0.05-100000-False] PASSED [ 62%]
tests/test_security/test_finite_key.py::TestParameterizedScenarios::test_key_length_scenarios[0.3-0.001-0.03-500000-True] PASSED [ 62%]
tests/test_security/test_finite_key.py::TestEdgeCases::test_zero_qber PASSED [ 62%]
tests/test_security/test_finite_key.py::TestEdgeCases::test_boundary_qber PASSED [ 62%]
tests/test_security/test_finite_key.py::TestEdgeCases::test_zero_storage_rate PASSED [ 62%]
tests/test_security/test_finite_key.py::TestEdgeCases::test_full_storage_rate PASSED [ 62%]
tests/test_security/test_finite_key.py::TestEdgeCases::test_test_fraction_extremes PASSED [ 62%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_commit_returns_valid_result PASSED [ 62%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_commit_different_data_different_commitment PASSED [ 62%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_commit_same_data_different_nonce PASSED [ 62%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_verify_valid_commitment PASSED [ 62%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_verify_wrong_data_fails PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_verify_wrong_nonce_fails PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_verify_wrong_commitment_fails PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_commit_bases PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_verify_bases PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_verify_bases_wrong_bases_fails PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_short_nonce_raises PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_custom_nonce_length PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_hash_data_static PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestSHA256Commitment::test_generate_nonce_static PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestCommitmentBinding::test_binding_property PASSED [ 63%]
tests/test_sifting/test_commitment.py::TestCommitmentHiding::test_commitment_reveals_nothing PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_validate_perfect_detection PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_validate_within_tolerance PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_validate_outside_tolerance_fails PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_validate_with_basis_balance PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_validate_unbalanced_bases_fails PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_validate_no_attempts_fails PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_hoeffding_bound_calculation PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_hoeffding_bound_passes_large_sample PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_invalid_expected_rate_raises PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_invalid_tolerance_raises PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_invalid_confidence_raises PASSED [ 64%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_required_samples_for_tolerance PASSED [ 65%]
tests/test_sifting/test_detection_validator.py::TestDetectionValidator::test_properties PASSED [ 65%]
tests/test_sifting/test_detection_validator.py::TestHoeffdingBound::test_hoeffding_bound_creation PASSED [ 65%]
tests/test_sifting/test_detection_validator.py::TestHoeffdingBound::test_hoeffding_formula PASSED [ 65%]
tests/test_sifting/test_detection_validator.py::TestValidationResult::test_validation_result_creation PASSED [ 65%]
tests/test_sifting/test_detection_validator.py::TestValidationResult::test_default_values PASSED [ 65%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_estimate_no_errors PASSED [ 65%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_estimate_all_errors PASSED [ 65%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_estimate_partial_errors PASSED [ 65%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_adjusted_qber_includes_penalty PASSED [ 65%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_compute_mu_penalty_formula PASSED [ 65%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_compute_mu_penalty_smaller_with_more_data PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_compute_mu_penalty_invalid_raises PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_exceeds_hard_limit_raises PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_exceeds_warning_limit_sets_flag PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_validate_passes_for_good_qber PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_validate_raises_for_bad_qber PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_empty_test_set_raises PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_length_mismatch_raises PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimator::test_invalid_epsilon_raises PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimate::test_qber_estimate_creation PASSED [ 66%]
tests/test_sifting/test_qber.py::TestQBEREstimate::test_default_confidence_level PASSED [ 66%]
tests/test_sifting/test_sifter.py::TestSifter::test_compute_sifted_key_all_match PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_compute_sifted_key_none_match PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_compute_sifted_key_partial_match PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_i0_i1_partition PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_length_mismatch_raises PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_extract_partition_keys PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_select_test_subset PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_select_test_subset_min_size PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_expected_matches PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_compute_match_rate PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSifter::test_compute_match_rate_empty PASSED [ 67%]
tests/test_sifting/test_sifter.py::TestSiftingResult::test_sifting_result_creation PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionInstantiation::test_default_values PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionInstantiation::test_explicit_model_values PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionInstantiation::test_invalid_link_model_rejected PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionInstantiation::test_invalid_eta_semantics_rejected PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionInstantiation::test_all_valid_link_models PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionInstantiation::test_all_valid_eta_semantics PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestResolveLinkModel::test_explicit_perfect_returns_perfect PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestResolveLinkModel::test_explicit_depolarise_returns_depolarise PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestResolveLinkModel::test_explicit_heralded_returns_heralded PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestResolveLinkModel::test_auto_with_full_params_returns_heralded PASSED [ 68%]
tests/test_simulation/test_channel_model_selection.py::TestResolveLinkModel::test_auto_without_eta_returns_depolarise PASSED [ 69%]
tests/test_simulation/test_channel_model_selection.py::TestResolveLinkModel::test_auto_with_perfect_fidelity_returns_perfect PASSED [ 69%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionIntegration::test_model_selection_round_trip PASSED [ 69%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionIntegration::test_model_resolution_consistency PASSED [ 69%]
tests/test_simulation/test_channel_model_selection.py::TestChannelModelSelectionIntegration::test_frozen_dataclass PASSED [ 69%]
tests/test_simulation/test_network_builder.py::TestCaligoNetworkBuilderCreation::test_valid_creation PASSED [ 69%]
tests/test_simulation/test_network_builder.py::TestCaligoNetworkBuilderCreation::test_creation_with_channel_params PASSED [ 69%]
tests/test_simulation/test_network_builder.py::TestCaligoNetworkBuilderCreation::test_default_channel_params PASSED [ 69%]
tests/test_simulation/test_network_builder.py::TestCaligoNetworkBuilderProperties::test_nsm_params_property PASSED [ 69%]
tests/test_simulation/test_network_builder.py::TestCaligoNetworkBuilderProperties::test_channel_params_property PASSED [ 69%]
tests/test_simulation/test_network_builder.py::TestCaligoNetworkBuilderBuildMethods::test_build_two_node_network_requires_squidasm PASSED [ 69%]
tests/test_simulation/test_network_builder.py::TestCaligoNetworkBuilderBuildMethods::test_build_two_node_network_custom_names PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestCaligoNetworkBuilderBuildMethods::test_build_stack_config_requires_squidasm PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestPerfectNetworkConfig::test_requires_squidasm PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestPerfectNetworkConfig::test_custom_node_names PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestPerfectNetworkConfig::test_custom_num_qubits PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestRealisticNetworkConfig::test_requires_squidasm PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestRealisticNetworkConfig::test_custom_fidelity PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestRealisticNetworkConfig::test_custom_t1_t2 PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestErvenExperimentalConfig::test_requires_squidasm PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestNetworkConfigSummary::test_creation PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestNetworkConfigSummary::test_from_builder PASSED [ 70%]
tests/test_simulation/test_network_builder.py::TestNetworkConfigSummary::test_from_builder_custom_names PASSED [ 71%]
tests/test_simulation/test_network_builder.py::TestNetworkConfigSummary::test_is_secure_based_on_qber PASSED [ 71%]
tests/test_simulation/test_network_builder.py::TestNetworkConfigSummary::test_frozen_dataclass PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelCreation::test_valid_creation PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelCreation::test_creation_with_different_r PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelCreation::test_invalid_r_below_zero PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelCreation::test_invalid_r_above_one PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelCreation::test_r_edge_cases PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelCreation::test_invalid_delta_t PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelProperties::test_depolar_prob PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelProperties::test_depolar_prob_extremes PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelApplyNoise::test_apply_noise_shape PASSED [ 71%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelApplyNoise::test_apply_noise_preserves_trace PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelApplyNoise::test_apply_noise_invalid_shape PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelApplyNoise::test_apply_noise_depolarization PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelApplyNoise::test_apply_noise_perfect_storage PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelDerivedMethods::test_get_effective_fidelity PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelDerivedMethods::test_effective_fidelity_extremes PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelDerivedMethods::test_get_min_entropy_bound PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestNSMStorageNoiseModelDerivedMethods::test_repr PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileCreation::test_valid_creation PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileCreation::test_default_transmission_loss PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileCreation::test_frozen_dataclass PASSED [ 72%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileInvariants::test_inv_cnp_001_source_fidelity_bounds PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileInvariants::test_inv_cnp_002_detector_efficiency_bounds PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileInvariants::test_inv_cnp_003_detector_error_bounds PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileInvariants::test_inv_cnp_004_dark_count_bounds PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileInvariants::test_inv_cnp_005_transmission_loss_bounds PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileDerivedProperties::test_total_qber_calculation PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileDerivedProperties::test_total_qber_perfect PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileDerivedProperties::test_is_secure PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileDerivedProperties::test_is_secure_false_for_high_qber PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileDerivedProperties::test_is_feasible PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileDerivedProperties::test_is_feasible_false_for_very_high_qber PASSED [ 73%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileDerivedProperties::test_security_margin PASSED [ 74%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileFactoryMethods::test_perfect PASSED [ 74%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileFactoryMethods::test_from_erven_experimental PASSED [ 74%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileFactoryMethods::test_realistic PASSED [ 74%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileFactoryMethods::test_realistic_custom_values PASSED [ 74%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileConversion::test_to_nsm_parameters PASSED [ 74%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileConversion::test_to_nsm_parameters_custom_storage PASSED [ 74%]
tests/test_simulation/test_noise_models.py::TestChannelNoiseProfileDiagnostics::test_get_diagnostic_info PASSED [ 74%]
tests/test_simulation/test_noise_models_contracts.py::test_channel_noise_profile_rejects_invalid_inputs PASSED [ 74%]
tests/test_simulation/test_noise_models_contracts.py::test_qber_conditional_range_and_monotonicity PASSED [ 74%]
tests/test_simulation/test_noise_models_contracts.py::test_snr_is_finite_and_nonnegative PASSED [ 74%]
tests/test_simulation/test_noise_models_contracts.py::test_to_nsm_parameters_has_positive_derived_quantities PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestTimeConstants::test_nanosecond_is_base_unit PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestTimeConstants::test_microsecond_conversion PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestTimeConstants::test_millisecond_conversion PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestTimeConstants::test_second_conversion PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestNSMParametersCreation::test_valid_creation PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestNSMParametersCreation::test_default_values PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestNSMParametersCreation::test_frozen_dataclass PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestNSMParametersInvariants::test_inv_nsm_001_storage_noise_lower_bound PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestNSMParametersInvariants::test_inv_nsm_001_storage_noise_upper_bound PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestNSMParametersInvariants::test_inv_nsm_001_edge_cases PASSED [ 75%]
tests/test_simulation/test_physical_model.py::TestNSMParametersInvariants::test_inv_nsm_002_storage_rate_bounds PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersInvariants::test_inv_nsm_003_dimension_must_be_2 PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersInvariants::test_inv_nsm_004_delta_t_positive PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersInvariants::test_inv_nsm_005_fidelity_bounds PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersInvariants::test_inv_nsm_006_detection_efficiency_bounds PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_depolar_prob PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_depolar_prob_extremes PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_qber_channel PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_qber_channel_with_detector_error PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_storage_capacity PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_storage_capacity_edge_cases PASSED [ 76%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_security_possible PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_security_impossible_high_qber PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestNSMParametersDerivedProperties::test_storage_security_satisfied PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestNSMParametersFactoryMethods::test_from_erven_experimental PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestNSMParametersFactoryMethods::test_for_testing PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestNSMParametersFactoryMethods::test_for_testing_custom_values PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestChannelParametersCreation::test_valid_creation PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestChannelParametersCreation::test_default_values PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestChannelParametersCreation::test_frozen_dataclass PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestChannelParametersInvariants::test_inv_ch_001_length_non_negative PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestChannelParametersInvariants::test_inv_ch_002_attenuation_non_negative PASSED [ 77%]
tests/test_simulation/test_physical_model.py::TestChannelParametersInvariants::test_inv_ch_003_speed_positive PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersInvariants::test_inv_ch_004_t1_positive PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersInvariants::test_inv_ch_005_t2_positive_and_less_than_t1 PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersInvariants::test_inv_ch_006_cycle_time_positive PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersDerivedProperties::test_propagation_delay_zero_length PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersDerivedProperties::test_propagation_delay_with_length PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersDerivedProperties::test_total_loss_db PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersDerivedProperties::test_total_loss_zero_length PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersDerivedProperties::test_transmittance_zero_loss PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersDerivedProperties::test_transmittance_with_loss PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersFactoryMethods::test_for_testing PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestChannelParametersFactoryMethods::test_from_erven_experimental PASSED [ 78%]
tests/test_simulation/test_physical_model.py::TestCreateDepolarNoiseModel::test_requires_netsquid PASSED [ 79%]
tests/test_simulation/test_physical_model.py::TestCreateT1T2NoiseModel::test_requires_netsquid PASSED [ 79%]
tests/test_simulation/test_physical_model_pdc.py::test_pdc_probability_rejects_nonpositive_mu PASSED [ 79%]
tests/test_simulation/test_physical_model_pdc.py::test_pdc_probability_negative_n_is_zero PASSED [ 79%]
tests/test_simulation/test_physical_model_pdc.py::test_pdc_probability_distribution_sums_close_to_one PASSED [ 79%]
tests/test_simulation/test_physical_model_pdc.py::test_p_sent_matches_definition PASSED [ 79%]
tests/test_simulation/test_physical_model_pdc.py::test_p_b_noclick_is_probability_and_decreases_with_eta PASSED [ 79%]
tests/test_simulation/test_physical_model_pdc.py::test_p_b_noclick_min_is_not_below_vacuum_probability PASSED [ 79%]
tests/test_simulation/test_timing.py::TestTimingBarrierState::test_states_exist PASSED [ 79%]
tests/test_simulation/test_timing.py::TestTimingBarrierState::test_states_are_unique PASSED [ 79%]
tests/test_simulation/test_timing.py::TestTimingBarrierCreation::test_valid_creation PASSED [ 79%]
tests/test_simulation/test_timing.py::TestTimingBarrierCreation::test_creation_with_custom_delta_t PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierCreation::test_creation_with_strict_mode_false PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierCreation::test_invalid_delta_t_zero PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierCreation::test_invalid_delta_t_negative PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierProperties::test_initial_state_is_idle PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierProperties::test_quantum_complete_time_initially_none PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierProperties::test_timing_compliant_initially_true PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierStateTransitions::test_mark_quantum_complete_transitions_to_waiting PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierStateTransitions::test_mark_quantum_complete_records_time PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierStateTransitions::test_mark_quantum_complete_twice_raises_in_strict PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierStateTransitions::test_mark_quantum_complete_twice_warns_in_lenient PASSED [ 80%]
tests/test_simulation/test_timing.py::TestTimingBarrierStateTransitions::test_wait_delta_t_transitions_to_ready PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierStateTransitions::test_wait_delta_t_without_mark_raises PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierStateTransitions::test_reset_returns_to_idle PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierBasisRevelation::test_cannot_reveal_in_idle PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierBasisRevelation::test_cannot_reveal_immediately_after_mark PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierBasisRevelation::test_cannot_reveal_immediately_lenient_mode PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierBasisRevelation::test_can_reveal_after_wait PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierCompliance::test_assert_raises_in_idle PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierCompliance::test_assert_raises_without_wait PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierCompliance::test_compliance_tracking PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierElapsedTime::test_elapsed_time_zero_before_mark PASSED [ 81%]
tests/test_simulation/test_timing.py::TestTimingBarrierElapsedTime::test_elapsed_time_after_mark PASSED [ 82%]
tests/test_simulation/test_timing.py::TestTimingBarrierDiagnostics::test_get_diagnostic_info PASSED [ 82%]
tests/test_simulation/test_timing.py::TestTimingBarrierDiagnostics::test_repr PASSED [ 82%]
tests/test_simulation/test_timing.py::TestTimingHelperFunctions::test_get_sim_time_without_netsquid PASSED [ 82%]
tests/test_simulation/test_timing.py::TestTimingHelperFunctions::test_sim_run_without_netsquid PASSED [ 82%]
tests/test_simulation/test_timing.py::TestTimingBarrierGeneratorPattern::test_wait_is_generator PASSED [ 82%]
tests/test_simulation/test_timing.py::TestTimingBarrierGeneratorPattern::test_generator_yields_once PASSED [ 82%]
tests/test_simulation/test_timing.py::TestTimingBarrierGeneratorPattern::test_yield_from_pattern PASSED [ 82%]
tests/test_simulation/test_verifier.py::TestNSMVerificationResult::test_result_creation PASSED [ 82%]
tests/test_simulation/test_verifier.py::TestNSMVerificationResult::test_result_immutable PASSED [ 82%]
tests/test_simulation/test_verifier.py::TestVerifyNSMSecurityCondition::test_secure_parameters_pass PASSED [ 82%]
tests/test_simulation/test_verifier.py::TestVerifyNSMSecurityCondition::test_marginal_parameters_pass PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestVerifyNSMSecurityCondition::test_insecure_parameters_fail_strict PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestVerifyNSMSecurityCondition::test_insecure_parameters_return_false_lenient PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestVerifyNSMSecurityCondition::test_hard_limit_exceeded_raises PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestVerifyNSMSecurityCondition::test_conservative_threshold_warning PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestValidateQBERMeasurement::test_valid_measurement PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestValidateQBERMeasurement::test_invalid_measurement PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestValidateQBERMeasurement::test_strict_mode_raises PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestValidateQBERMeasurement::test_exact_match PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestValidateTimingCompliance::test_sufficient_wait PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestValidateTimingCompliance::test_wait_with_tolerance PASSED [ 83%]
tests/test_simulation/test_verifier.py::TestValidateTimingCompliance::test_insufficient_wait_strict PASSED [ 84%]
tests/test_simulation/test_verifier.py::TestValidateTimingCompliance::test_insufficient_wait_lenient PASSED [ 84%]
tests/test_simulation/test_verifier.py::TestPreflightSecurityCheck::test_secure_config_passes PASSED [ 84%]
tests/test_simulation/test_verifier.py::TestPreflightSecurityCheck::test_insecure_config_fails PASSED [ 84%]
tests/test_simulation/test_verifier.py::TestPreflightSecurityCheck::test_preflight_uses_expected_qber PASSED [ 84%]
tests/test_simulation/test_verifier.py::TestPostflightSecurityCheck::test_postflight_with_measured_qber PASSED [ 84%]
tests/test_simulation/test_verifier.py::TestPostflightSecurityCheck::test_postflight_fails_on_high_qber PASSED [ 84%]
tests/test_types/test_exceptions.py::TestExceptionHierarchy::test_caligo_error_is_base PASSED [ 84%]
tests/test_types/test_exceptions.py::TestExceptionHierarchy::test_simulation_error_subclasses PASSED [ 84%]
tests/test_types/test_exceptions.py::TestExceptionHierarchy::test_security_error_subclasses PASSED [ 84%]
tests/test_types/test_exceptions.py::TestExceptionHierarchy::test_protocol_error_subclasses PASSED [ 84%]
tests/test_types/test_exceptions.py::TestExceptionHierarchy::test_exception_can_be_raised PASSED [ 85%]
tests/test_types/test_exceptions.py::TestExceptionHierarchy::test_exception_message PASSED [ 85%]
tests/test_types/test_exceptions.py::TestProtocolPhase::test_all_phases_defined PASSED [ 85%]
tests/test_types/test_exceptions.py::TestProtocolPhase::test_phase_values PASSED [ 85%]
tests/test_types/test_exceptions.py::TestAbortReason::test_phase_i_abort_reasons PASSED [ 85%]
tests/test_types/test_exceptions.py::TestAbortReason::test_phase_ii_abort_reasons PASSED [ 85%]
tests/test_types/test_exceptions.py::TestAbortReason::test_phase_iii_abort_reasons PASSED [ 85%]
tests/test_types/test_exceptions.py::TestAbortReason::test_phase_iv_abort_reasons PASSED [ 85%]
tests/test_types/test_keys.py::TestObliviousKey::test_valid_key_creation PASSED [ 85%]
tests/test_types/test_keys.py::TestObliviousKey::test_custom_security_param PASSED [ 85%]
tests/test_types/test_keys.py::TestObliviousKey::test_custom_creation_time PASSED [ 85%]
tests/test_types/test_keys.py::TestObliviousKey::test_inv_key_001_length_mismatch PASSED [ 85%]
tests/test_types/test_keys.py::TestObliviousKey::test_inv_key_002_security_param_zero PASSED [ 86%]
tests/test_types/test_keys.py::TestObliviousKey::test_inv_key_002_security_param_one PASSED [ 86%]
tests/test_types/test_keys.py::TestObliviousKey::test_inv_key_002_security_param_negative PASSED [ 86%]
tests/test_types/test_keys.py::TestObliviousKey::test_inv_key_003_negative_creation_time PASSED [ 86%]
tests/test_types/test_keys.py::TestObliviousKey::test_frozen_immutable PASSED [ 86%]
tests/test_types/test_keys.py::TestAliceObliviousKey::test_valid_alice_key PASSED [ 86%]
tests/test_types/test_keys.py::TestAliceObliviousKey::test_inv_alice_001_s0_length_mismatch PASSED [ 86%]
tests/test_types/test_keys.py::TestAliceObliviousKey::test_inv_alice_001_s1_length_mismatch PASSED [ 86%]
tests/test_types/test_keys.py::TestAliceObliviousKey::test_inv_alice_002_invalid_security_param PASSED [ 86%]
tests/test_types/test_keys.py::TestBobObliviousKey::test_valid_bob_key_choice_0 PASSED [ 86%]
tests/test_types/test_keys.py::TestBobObliviousKey::test_valid_bob_key_choice_1 PASSED [ 86%]
tests/test_types/test_keys.py::TestBobObliviousKey::test_inv_bob_001_length_mismatch PASSED [ 87%]
tests/test_types/test_keys.py::TestBobObliviousKey::test_inv_bob_002_invalid_choice_bit PASSED [ 87%]
tests/test_types/test_keys.py::TestBobObliviousKey::test_inv_bob_002_negative_choice_bit PASSED [ 87%]
tests/test_types/test_measurements.py::TestMeasurementRecord::test_valid_record PASSED [ 87%]
tests/test_types/test_measurements.py::TestMeasurementRecord::test_inv_meas_001_invalid_outcome PASSED [ 87%]
tests/test_types/test_measurements.py::TestMeasurementRecord::test_inv_meas_002_invalid_basis PASSED [ 87%]
tests/test_types/test_measurements.py::TestMeasurementRecord::test_inv_meas_003_negative_round_id PASSED [ 87%]
tests/test_types/test_measurements.py::TestMeasurementRecord::test_inv_meas_004_negative_timestamp PASSED [ 87%]
tests/test_types/test_measurements.py::TestMeasurementRecord::test_detected_false PASSED [ 87%]
tests/test_types/test_measurements.py::TestRoundResult::test_valid_round_result PASSED [ 87%]
tests/test_types/test_measurements.py::TestRoundResult::test_is_valid_both_detected PASSED [ 87%]
tests/test_types/test_measurements.py::TestRoundResult::test_is_valid_alice_not_detected PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_bases_match_true PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_bases_match_false PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_outcomes_match_true PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_outcomes_match_false PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_contributes_to_sifted_key_true PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_contributes_to_sifted_key_false_bases_differ PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_has_error_true PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_has_error_false_outcomes_match PASSED [ 88%]
tests/test_types/test_measurements.py::TestRoundResult::test_invalid_alice_outcome PASSED [ 88%]
tests/test_types/test_measurements.py::TestDetectionEvent::test_valid_detection_event PASSED [ 88%]
tests/test_types/test_measurements.py::TestDetectionEvent::test_invalid_round_id PASSED [ 89%]
tests/test_types/test_measurements.py::TestDetectionEvent::test_invalid_timestamp PASSED [ 89%]
tests/test_types/test_phase_boundary_quantum_to_sifting.py::test_p12_001_measurement_and_bases_build_valid_quantum_phase_result PASSED [ 89%]
tests/test_types/test_phase_boundary_quantum_to_sifting.py::test_p12_010_quantum_phase_result_length_mismatch_rejected PASSED [ 89%]
tests/test_types/test_phase_boundary_quantum_to_sifting.py::test_p12_020_quantum_phase_result_invalid_values_rejected PASSED [ 89%]
tests/test_types/test_phase_contracts.py::TestQuantumPhaseResult::test_valid_quantum_result PASSED [ 89%]
tests/test_types/test_phase_contracts.py::TestQuantumPhaseResult::test_post_q_001_outcomes_length_mismatch PASSED [ 89%]
tests/test_types/test_phase_contracts.py::TestQuantumPhaseResult::test_post_q_002_bases_length_mismatch PASSED [ 89%]
tests/test_types/test_phase_contracts.py::TestQuantumPhaseResult::test_post_q_003_invalid_outcomes PASSED [ 89%]
tests/test_types/test_phase_contracts.py::TestQuantumPhaseResult::test_post_q_004_invalid_bases PASSED [ 89%]
tests/test_types/test_phase_contracts.py::TestQuantumPhaseResult::test_empty_result_valid PASSED [ 89%]
tests/test_types/test_phase_contracts.py::TestSiftingPhaseResult::test_valid_sifting_result PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestSiftingPhaseResult::test_post_s_001_key_length_mismatch PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestSiftingPhaseResult::test_post_s_002_qber_adjusted_calculation PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestSiftingPhaseResult::test_post_s_003_qber_exceeds_limit PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestReconciliationPhaseResult::test_valid_reconciliation_result PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestReconciliationPhaseResult::test_post_r_002_hash_not_verified PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestAmplificationPhaseResult::test_valid_amplification_result PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestAmplificationPhaseResult::test_post_amp_001_zero_key_length PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestAmplificationPhaseResult::test_post_amp_002_insufficient_entropy PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestObliviousTransferOutput::test_valid_ot_output PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestObliviousTransferOutput::test_post_ot_001_alice_key_length_mismatch PASSED [ 90%]
tests/test_types/test_phase_contracts.py::TestObliviousTransferOutput::test_post_ot_003_bob_key_mismatch PASSED [ 91%]
tests/test_types/test_phase_contracts_negative.py::test_reconciliation_post_r_001_leakage_cap_exceeded_raises PASSED [ 91%]
tests/test_types/test_phase_contracts_negative.py::test_reconciliation_leakage_within_cap_no_raise PASSED [ 91%]
tests/test_types/test_phase_contracts_negative.py::test_ot_post_ot_002_bob_key_length_mismatch_raises PASSED [ 91%]
tests/test_utils/test_bitarray_utils.py::TestXorBitarrays::test_xor_basic PASSED [ 91%]
tests/test_utils/test_bitarray_utils.py::TestXorBitarrays::test_xor_all_zeros PASSED [ 91%]
tests/test_utils/test_bitarray_utils.py::TestXorBitarrays::test_xor_with_self PASSED [ 91%]
tests/test_utils/test_bitarray_utils.py::TestXorBitarrays::test_xor_length_mismatch PASSED [ 91%]
tests/test_utils/test_bitarray_utils.py::TestXorBitarrays::test_xor_empty PASSED [ 91%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayToBytes::test_byte_aligned PASSED [ 91%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayToBytes::test_multiple_bytes PASSED [ 91%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayToBytes::test_empty PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestBytesToBitarray::test_single_byte PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestBytesToBitarray::test_multiple_bytes PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestBytesToBitarray::test_empty PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestBytesToBitarray::test_roundtrip PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestRandomBitarray::test_correct_length PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestRandomBitarray::test_randomness PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestRandomBitarray::test_different_each_call PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestRandomBitarray::test_negative_length PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestHammingDistance::test_identical_arrays PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestHammingDistance::test_completely_different PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestHammingDistance::test_known_distance PASSED [ 92%]
tests/test_utils/test_bitarray_utils.py::TestHammingDistance::test_length_mismatch PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestHammingDistance::test_empty PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestSliceBitarray::test_basic_slice PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestSliceBitarray::test_single_index PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestSliceBitarray::test_empty_indices PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestSliceBitarray::test_out_of_bounds PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestSliceBitarray::test_negative_index PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestSliceBitarray::test_preserves_order PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayFromNumpy::test_basic_conversion PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayFromNumpy::test_uint8_array PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayFromNumpy::test_empty_array PASSED [ 93%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayFromNumpy::test_invalid_values PASSED [ 94%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayToNumpy::test_basic_conversion PASSED [ 94%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayToNumpy::test_dtype PASSED [ 94%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayToNumpy::test_empty PASSED [ 94%]
tests/test_utils/test_bitarray_utils.py::TestBitarrayToNumpy::test_roundtrip PASSED [ 94%]
tests/test_utils/test_logging.py::TestGetLogger::test_returns_logger PASSED [ 94%]
tests/test_utils/test_logging.py::TestGetLogger::test_logger_name_preserved PASSED [ 94%]
tests/test_utils/test_logging.py::TestGetLogger::test_hierarchical_logger_names PASSED [ 94%]
tests/test_utils/test_logging.py::TestSetupScriptLogging::test_returns_logger PASSED [ 94%]
tests/test_utils/test_logging.py::TestSetupScriptLogging::test_creates_log_file PASSED [ 94%]
tests/test_utils/test_logging.py::TestSetupScriptLogging::test_writes_to_log_file PASSED [ 94%]
tests/test_utils/test_logging.py::TestSetupScriptLogging::test_idempotent_setup PASSED [ 95%]
tests/test_utils/test_logging.py::TestSetupScriptLogging::test_log_level_configuration PASSED [ 95%]
tests/test_utils/test_logging.py::TestSetupScriptLogging::test_terminal_output_disabled_by_default PASSED [ 95%]
tests/test_utils/test_logging.py::TestSetupScriptLogging::test_terminal_output_enabled PASSED [ 95%]
tests/test_utils/test_logging.py::TestResetLoggingState::test_reset_allows_reconfiguration PASSED [ 95%]
tests/test_utils/test_math.py::TestBinaryEntropy::test_entropy_at_zero PASSED [ 95%]
tests/test_utils/test_math.py::TestBinaryEntropy::test_entropy_at_one PASSED [ 95%]
tests/test_utils/test_math.py::TestBinaryEntropy::test_entropy_at_half PASSED [ 95%]
tests/test_utils/test_math.py::TestBinaryEntropy::test_entropy_symmetry PASSED [ 95%]
tests/test_utils/test_math.py::TestBinaryEntropy::test_entropy_bounds PASSED [ 95%]
tests/test_utils/test_math.py::TestBinaryEntropy::test_entropy_invalid_negative PASSED [ 95%]
tests/test_utils/test_math.py::TestBinaryEntropy::test_entropy_invalid_above_one PASSED [ 96%]
tests/test_utils/test_math.py::TestBinaryEntropy::test_entropy_known_value PASSED [ 96%]
tests/test_utils/test_math.py::TestChannelCapacity::test_capacity_at_zero_qber PASSED [ 96%]
tests/test_utils/test_math.py::TestChannelCapacity::test_capacity_at_half_qber PASSED [ 96%]
tests/test_utils/test_math.py::TestChannelCapacity::test_capacity_decreases_with_qber PASSED [ 96%]
tests/test_utils/test_math.py::TestChannelCapacity::test_capacity_invalid_qber PASSED [ 96%]
tests/test_utils/test_math.py::TestFiniteSizePenalty::test_penalty_positive PASSED [ 96%]
tests/test_utils/test_math.py::TestFiniteSizePenalty::test_penalty_decreases_with_sample_size PASSED [ 96%]
tests/test_utils/test_math.py::TestFiniteSizePenalty::test_penalty_increases_with_security PASSED [ 96%]
tests/test_utils/test_math.py::TestFiniteSizePenalty::test_penalty_invalid_n PASSED [ 96%]
tests/test_utils/test_math.py::TestFiniteSizePenalty::test_penalty_invalid_k PASSED [ 96%]
tests/test_utils/test_math.py::TestFiniteSizePenalty::test_penalty_invalid_epsilon PASSED [ 97%]
tests/test_utils/test_math.py::TestFiniteSizePenalty::test_penalty_reasonable_magnitude PASSED [ 97%]
tests/test_utils/test_math.py::TestGammaFunction::test_gamma_at_zero PASSED [ 97%]
tests/test_utils/test_math.py::TestGammaFunction::test_gamma_at_one PASSED [ 97%]
tests/test_utils/test_math.py::TestGammaFunction::test_gamma_decreases_with_noise PASSED [ 97%]
tests/test_utils/test_math.py::TestGammaFunction::test_gamma_invalid_r PASSED [ 97%]
tests/test_utils/test_math.py::TestGammaFunction::test_gamma_typical_value PASSED [ 97%]
tests/test_utils/test_math.py::TestSmoothMinEntropyRate::test_entropy_rate_positive_for_low_qber PASSED [ 97%]
tests/test_utils/test_math.py::TestSmoothMinEntropyRate::test_entropy_rate_with_perfect_storage PASSED [ 97%]
tests/test_utils/test_math.py::TestKeyLengthBound::test_key_length_positive_for_good_params PASSED [ 97%]
tests/test_utils/test_math.py::TestKeyLengthBound::test_key_length_zero_for_bad_params PASSED [ 97%]
tests/test_utils/test_math.py::TestKeyLengthBound::test_key_length_decreases_with_leakage PASSED [ 98%]
tests/test_utils/test_math.py::TestKeyLengthBound::test_key_length_with_nsm_gamma PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_binary_entropy_rejects_out_of_range PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_binary_entropy_edges_are_zero[0.0] PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_binary_entropy_edges_are_zero[1.0] PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_binary_entropy_midpoint_is_one PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_channel_capacity_rejects_out_of_range PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_suggested_ldpc_rate_low_qber_returns_highest_rate PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_suggested_ldpc_rate_near_hard_limit_returns_lowest_rate PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_suggested_ldpc_rate_safety_margin_reduces_rate_or_equal PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_blind_reconciliation_initial_config_thresholds[0.0-0-3] PASSED [ 98%]
tests/test_utils/test_math_contracts.py::test_blind_reconciliation_initial_config_thresholds[0.019999-0-3] PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_blind_reconciliation_initial_config_thresholds[0.02-0-3] PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_blind_reconciliation_initial_config_thresholds[0.049999-0-3] PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_blind_reconciliation_initial_config_thresholds[0.05-0-3] PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_blind_reconciliation_initial_config_thresholds[0.079999-0-3] PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_blind_reconciliation_initial_config_thresholds[0.08-0-3] PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_finite_size_penalty_preconditions PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_finite_size_penalty_decreases_with_larger_test_set PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_gamma_function_preconditions_and_edges PASSED [ 99%]
tests/test_utils/test_math_contracts.py::test_key_length_bound_gamma_branches_floor_at_zero PASSED [ 99%]
tests/test_utils/test_numba_kernels.py::test_bfs_reachable_matches_python PASSED [ 99%]
tests/test_utils/test_numba_kernels.py::test_ace_viterbi_matches_python_pass_fail PASSED [100%]

==================================== ERRORS ====================================
_____ ERROR at setup of TestBeliefPropagationDecoder.test_decode_noiseless _____
tests/reconciliation/test_bp_decoder.py:26: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_____ ERROR at setup of TestBeliefPropagationDecoder.test_decode_low_noise _____
tests/reconciliation/test_bp_decoder.py:26: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
__ ERROR at setup of TestBeliefPropagationDecoder.test_returns_decode_result ___
tests/reconciliation/test_bp_decoder.py:26: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
__________________ ERROR at setup of test_syndrome_linearity ___________________
tests/reconciliation/test_contracts.py:26: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_______ ERROR at setup of test_decoder_converged_implies_syndrome_match ________
tests/reconciliation/test_contracts.py:26: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_______________ ERROR at setup of test_edge_qber_does_not_crash ________________
tests/reconciliation/test_contracts.py:26: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
____________ ERROR at setup of test_leakage_budget_exceeded_aborts _____________
tests/reconciliation/test_contracts.py:26: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestEncoderDecoderIntegration.test_encode_decode_noiseless _
tests/reconciliation/test_integration.py:29: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestEncoderDecoderIntegration.test_encode_decode_with_noise _
tests/reconciliation/test_integration.py:29: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
________ ERROR at setup of TestHashVerification.test_hash_after_decode _________
tests/reconciliation/test_integration.py:29: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestRateSelectionFlow.test_rate_selection_guides_matrix_choice _
tests/reconciliation/test_integration.py:29: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestOrchestratorIntegration.test_orchestrator_initialization _
tests/reconciliation/test_integration.py:29: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestOrchestratorIntegration.test_single_block_reconciliation _
tests/reconciliation/test_integration.py:29: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestHighRatePatternBased.test_high_rate_with_pattern_rate_0_8 _
tests/reconciliation/test_integration.py:29: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
__ ERROR at setup of TestHighRatePatternBased.test_high_rate_stress_rate_0_9 ___
tests/reconciliation/test_integration.py:29: in matrix_manager
    return MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
___ ERROR at setup of test_orchestrator_reconcile_key_multiblock_happy_path ____
tests/reconciliation/test_orchestrator_multiblock.py:22: in matrix_manager
    yield MatrixManager.from_directory(constants.LDPC_MATRICES_PATH)
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
____ ERROR at setup of TestUntaintedProperty.test_untainted_property_holds _____
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
__ ERROR at setup of TestUntaintedProperty.test_forced_puncturing_is_minority __
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_____ ERROR at setup of TestPatternDeterminism.test_same_seed_same_pattern _____
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestPatternDeterminism.test_different_seed_different_pattern _
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
______ ERROR at setup of TestRecoverability.test_1step_recoverable_count _______
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestPatternProperties.test_pattern_size_matches_frame_size _
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestPatternProperties.test_puncture_count_matches_rate_difference _
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
____ ERROR at setup of TestPatternProperties.test_pattern_values_are_binary ____
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
___ ERROR at setup of TestEdgeCases.test_target_rate_must_exceed_mother_rate ___
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
____ ERROR at setup of TestEdgeCases.test_target_rate_equal_to_mother_rate _____
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestPatternStatistics.test_pattern_coverage_increases_with_rate _
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
_ ERROR at setup of TestPatternStatistics.test_forced_puncturing_increases_at_high_rates _
tests/reconciliation/test_puncture_patterns.py:29: in mother_code_matrix
    manager = MatrixManager.from_directory(
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
=================================== FAILURES ===================================
___________________ test_phase_e_end_to_end_ot_agreement[0] ____________________
tests/e2e/test_phase_e_protocol.py:122: in test_phase_e_end_to_end_ot_agreement
    ot, _raw = run_protocol(params, bob_choice_bit=choice_bit)
caligo/protocol/orchestrator.py:74: in run_protocol
    raw_results = squidasm_run(network_config, {"Alice": alice, "Bob": bob})
../../squidasm/squidasm/run/stack/run.py:126: in run
    results = _run(network)
../../squidasm/squidasm/run/stack/run.py:96: in _run
    ns.sim_run()
../../qia/lib/python3.10/site-packages/netsquid/util/simtools.py:279: in sim_run
    _simengine.run()
pydynaa/core.pyx:1477: in pydynaa.core.SimulationEngine.run
    ???
pydynaa/core.pyx:173: in pydynaa.core._call_obj_expr
    ???
pydynaa/core.pyx:170: in pydynaa.core._call_obj_expr
    ???
netsquid/protocols/protocol.pyx:290: in netsquid.protocols.protocol.Protocol._expression_callback
    ???
netsquid/protocols/protocol.pyx:301: in netsquid.protocols.protocol.Protocol._generator_step
    ???
../../squidasm/squidasm/sim/stack/host.py:170: in run
    result = yield from self._program.run(context)
caligo/protocol/base.py:134: in run
    result = yield from self._run_protocol(context)
caligo/protocol/alice.py:80: in _run_protocol
    reconciled_bits, total_syndrome_bits, verified_positions = yield from self._phase3_reconcile(
caligo/protocol/alice.py:366: in _phase3_reconcile
    result = yield from self._drive_alice_strategy(
caligo/protocol/alice.py:421: in _drive_alice_strategy
    outgoing = next(gen)
caligo/reconciliation/strategies/baseline.py:210: in alice_reconcile_block
    syndrome = self._codec.encode(frame, pattern)
caligo/reconciliation/strategies/codec.py:89: in encode
    packed_frame = self._bitpack(frame)
caligo/reconciliation/strategies/codec.py:245: in _bitpack
    packed[word_idx] |= (1 << bit_idx)
E   TypeError: ufunc 'bitwise_or' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
------------------------------ Captured log call -------------------------------
ERROR    caligo.protocol.alice:alice.py:206 DEBUG: Alice outcomes (first 20): [0 1 1 0 1 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1]
ERROR    caligo.protocol.alice:alice.py:207 DEBUG: Alice bases (first 20): [0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 0]
ERROR    caligo.protocol.alice:alice.py:208 DEBUG: Bob outcomes (first 20): [0 1 1 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1]
ERROR    caligo.protocol.alice:alice.py:209 DEBUG: Bob bases (first 20): [0 1 0 1 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1 0]
___________________ test_phase_e_end_to_end_ot_agreement[1] ____________________
tests/e2e/test_phase_e_protocol.py:122: in test_phase_e_end_to_end_ot_agreement
    ot, _raw = run_protocol(params, bob_choice_bit=choice_bit)
caligo/protocol/orchestrator.py:74: in run_protocol
    raw_results = squidasm_run(network_config, {"Alice": alice, "Bob": bob})
../../squidasm/squidasm/run/stack/run.py:126: in run
    results = _run(network)
../../squidasm/squidasm/run/stack/run.py:96: in _run
    ns.sim_run()
../../qia/lib/python3.10/site-packages/netsquid/util/simtools.py:279: in sim_run
    _simengine.run()
pydynaa/core.pyx:1477: in pydynaa.core.SimulationEngine.run
    ???
pydynaa/core.pyx:173: in pydynaa.core._call_obj_expr
    ???
pydynaa/core.pyx:170: in pydynaa.core._call_obj_expr
    ???
netsquid/protocols/protocol.pyx:290: in netsquid.protocols.protocol.Protocol._expression_callback
    ???
netsquid/protocols/protocol.pyx:301: in netsquid.protocols.protocol.Protocol._generator_step
    ???
../../squidasm/squidasm/sim/stack/host.py:170: in run
    result = yield from self._program.run(context)
caligo/protocol/base.py:134: in run
    result = yield from self._run_protocol(context)
caligo/protocol/alice.py:80: in _run_protocol
    reconciled_bits, total_syndrome_bits, verified_positions = yield from self._phase3_reconcile(
caligo/protocol/alice.py:366: in _phase3_reconcile
    result = yield from self._drive_alice_strategy(
caligo/protocol/alice.py:421: in _drive_alice_strategy
    outgoing = next(gen)
caligo/reconciliation/strategies/baseline.py:210: in alice_reconcile_block
    syndrome = self._codec.encode(frame, pattern)
caligo/reconciliation/strategies/codec.py:89: in encode
    packed_frame = self._bitpack(frame)
caligo/reconciliation/strategies/codec.py:245: in _bitpack
    packed[word_idx] |= (1 << bit_idx)
E   TypeError: ufunc 'bitwise_or' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
------------------------------ Captured log call -------------------------------
ERROR    caligo.protocol.alice:alice.py:206 DEBUG: Alice outcomes (first 20): [0 1 1 0 1 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1]
ERROR    caligo.protocol.alice:alice.py:207 DEBUG: Alice bases (first 20): [0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 0]
ERROR    caligo.protocol.alice:alice.py:208 DEBUG: Bob outcomes (first 20): [0 1 1 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1]
ERROR    caligo.protocol.alice:alice.py:209 DEBUG: Bob bases (first 20): [0 1 0 1 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1 0]
______________ test_phase_e_blind_reconciliation_ot_agreement[0] _______________
tests/e2e/test_phase_e_protocol.py:158: in test_phase_e_blind_reconciliation_ot_agreement
    ot, _raw = run_protocol(params, bob_choice_bit=choice_bit)
caligo/protocol/orchestrator.py:74: in run_protocol
    raw_results = squidasm_run(network_config, {"Alice": alice, "Bob": bob})
../../squidasm/squidasm/run/stack/run.py:126: in run
    results = _run(network)
../../squidasm/squidasm/run/stack/run.py:96: in _run
    ns.sim_run()
../../qia/lib/python3.10/site-packages/netsquid/util/simtools.py:279: in sim_run
    _simengine.run()
pydynaa/core.pyx:1477: in pydynaa.core.SimulationEngine.run
    ???
pydynaa/core.pyx:173: in pydynaa.core._call_obj_expr
    ???
pydynaa/core.pyx:170: in pydynaa.core._call_obj_expr
    ???
netsquid/protocols/protocol.pyx:290: in netsquid.protocols.protocol.Protocol._expression_callback
    ???
netsquid/protocols/protocol.pyx:301: in netsquid.protocols.protocol.Protocol._generator_step
    ???
../../squidasm/squidasm/sim/stack/host.py:170: in run
    result = yield from self._program.run(context)
caligo/protocol/base.py:134: in run
    result = yield from self._run_protocol(context)
caligo/protocol/alice.py:80: in _run_protocol
    reconciled_bits, total_syndrome_bits, verified_positions = yield from self._phase3_reconcile(
caligo/protocol/alice.py:366: in _phase3_reconcile
    result = yield from self._drive_alice_strategy(
caligo/protocol/alice.py:421: in _drive_alice_strategy
    outgoing = next(gen)
caligo/reconciliation/strategies/blind.py:193: in alice_reconcile_block
    syndrome = self._codec.encode(frame, mother_pattern)
caligo/reconciliation/strategies/codec.py:89: in encode
    packed_frame = self._bitpack(frame)
caligo/reconciliation/strategies/codec.py:245: in _bitpack
    packed[word_idx] |= (1 << bit_idx)
E   TypeError: ufunc 'bitwise_or' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
------------------------------ Captured log call -------------------------------
ERROR    caligo.protocol.alice:alice.py:206 DEBUG: Alice outcomes (first 20): [0 1 1 0 1 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1]
ERROR    caligo.protocol.alice:alice.py:207 DEBUG: Alice bases (first 20): [0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 0]
ERROR    caligo.protocol.alice:alice.py:208 DEBUG: Bob outcomes (first 20): [0 1 1 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1]
ERROR    caligo.protocol.alice:alice.py:209 DEBUG: Bob bases (first 20): [0 1 0 1 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1 0]
______________ test_phase_e_blind_reconciliation_ot_agreement[1] _______________
tests/e2e/test_phase_e_protocol.py:158: in test_phase_e_blind_reconciliation_ot_agreement
    ot, _raw = run_protocol(params, bob_choice_bit=choice_bit)
caligo/protocol/orchestrator.py:74: in run_protocol
    raw_results = squidasm_run(network_config, {"Alice": alice, "Bob": bob})
../../squidasm/squidasm/run/stack/run.py:126: in run
    results = _run(network)
../../squidasm/squidasm/run/stack/run.py:96: in _run
    ns.sim_run()
../../qia/lib/python3.10/site-packages/netsquid/util/simtools.py:279: in sim_run
    _simengine.run()
pydynaa/core.pyx:1477: in pydynaa.core.SimulationEngine.run
    ???
pydynaa/core.pyx:173: in pydynaa.core._call_obj_expr
    ???
pydynaa/core.pyx:170: in pydynaa.core._call_obj_expr
    ???
netsquid/protocols/protocol.pyx:290: in netsquid.protocols.protocol.Protocol._expression_callback
    ???
netsquid/protocols/protocol.pyx:301: in netsquid.protocols.protocol.Protocol._generator_step
    ???
../../squidasm/squidasm/sim/stack/host.py:170: in run
    result = yield from self._program.run(context)
caligo/protocol/base.py:134: in run
    result = yield from self._run_protocol(context)
caligo/protocol/alice.py:80: in _run_protocol
    reconciled_bits, total_syndrome_bits, verified_positions = yield from self._phase3_reconcile(
caligo/protocol/alice.py:366: in _phase3_reconcile
    result = yield from self._drive_alice_strategy(
caligo/protocol/alice.py:421: in _drive_alice_strategy
    outgoing = next(gen)
caligo/reconciliation/strategies/blind.py:193: in alice_reconcile_block
    syndrome = self._codec.encode(frame, mother_pattern)
caligo/reconciliation/strategies/codec.py:89: in encode
    packed_frame = self._bitpack(frame)
caligo/reconciliation/strategies/codec.py:245: in _bitpack
    packed[word_idx] |= (1 << bit_idx)
E   TypeError: ufunc 'bitwise_or' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
------------------------------ Captured log call -------------------------------
ERROR    caligo.protocol.alice:alice.py:206 DEBUG: Alice outcomes (first 20): [0 1 1 0 1 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1]
ERROR    caligo.protocol.alice:alice.py:207 DEBUG: Alice bases (first 20): [0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 0]
ERROR    caligo.protocol.alice:alice.py:208 DEBUG: Bob outcomes (first 20): [0 1 1 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1]
ERROR    caligo.protocol.alice:alice.py:209 DEBUG: Bob bases (first 20): [0 1 0 1 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1 0]
___ test_fint_001_blind_factory_reconciler_runs_one_block_when_assets_exist ____
tests/reconciliation/test_factory_integration.py:40: in test_fint_001_blind_factory_reconciler_runs_one_block_when_assets_exist
    corrected_bytes, meta = reconciler.reconcile(alice.tobytes(), bob.tobytes())
caligo/reconciliation/factory.py:640: in reconcile
    orchestrator = self._get_orchestrator()
caligo/reconciliation/factory.py:586: in _get_orchestrator
    matrix_manager = MatrixManager.from_directory(Path(matrix_path))
caligo/reconciliation/matrix_manager.py:153: in from_directory
    raise FileNotFoundError(
E   FileNotFoundError: Missing LDPC matrix for rate 0.50: /home/adaro/projects/qia_25/qia-challenge-2025/caligo/caligo/configs/ldpc_matrices/ldpc_4096_rate0.50.npz
=============================== warnings summary ===============================
../../qia/lib/python3.10/site-packages/netsquid/qubits/__init__.py:64
../../qia/lib/python3.10/site-packages/netsquid/qubits/__init__.py:64
../../qia/lib/python3.10/site-packages/netsquid/qubits/__init__.py:64
../../qia/lib/python3.10/site-packages/netsquid/qubits/__init__.py:64
  /home/adaro/projects/qia_25/qia/lib/python3.10/site-packages/netsquid/qubits/__init__.py:64: DeprecationWarning: Please import `csr_matrix` from the `scipy.sparse` namespace; the `scipy.sparse.csr` namespace is deprecated and will be removed in SciPy 2.0.0.
    from netsquid.qubits import qreprutil

<frozen importlib._bootstrap>:241: 14 warnings
  <frozen importlib._bootstrap>:241: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)

tests/integration/test_nsm_parameter_enforcement.py::TestDeviceNoiseInstallation::test_gate_depolar_installed
  /home/adaro/projects/qia_25/qia/lib/python3.10/site-packages/_pytest/python.py:194: DeprecationWarning: The PhysicalInstruction.q_noise_model is deprecated, use PhysicalInstruction.quantum_noise_model instead.
    result = testfunction(**testargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/e2e/test_phase_e_protocol.py::test_phase_e_end_to_end_ot_agreement[0]
FAILED tests/e2e/test_phase_e_protocol.py::test_phase_e_end_to_end_ot_agreement[1]
FAILED tests/e2e/test_phase_e_protocol.py::test_phase_e_blind_reconciliation_ot_agreement[0]
FAILED tests/e2e/test_phase_e_protocol.py::test_phase_e_blind_reconciliation_ot_agreement[1]
FAILED tests/reconciliation/test_factory_integration.py::test_fint_001_blind_factory_reconciler_runs_one_block_when_assets_exist
ERROR tests/reconciliation/test_bp_decoder.py::TestBeliefPropagationDecoder::test_decode_noiseless
ERROR tests/reconciliation/test_bp_decoder.py::TestBeliefPropagationDecoder::test_decode_low_noise
ERROR tests/reconciliation/test_bp_decoder.py::TestBeliefPropagationDecoder::test_returns_decode_result
ERROR tests/reconciliation/test_contracts.py::test_syndrome_linearity - FileN...
ERROR tests/reconciliation/test_contracts.py::test_decoder_converged_implies_syndrome_match
ERROR tests/reconciliation/test_contracts.py::test_edge_qber_does_not_crash
ERROR tests/reconciliation/test_contracts.py::test_leakage_budget_exceeded_aborts
ERROR tests/reconciliation/test_integration.py::TestEncoderDecoderIntegration::test_encode_decode_noiseless
ERROR tests/reconciliation/test_integration.py::TestEncoderDecoderIntegration::test_encode_decode_with_noise
ERROR tests/reconciliation/test_integration.py::TestHashVerification::test_hash_after_decode
ERROR tests/reconciliation/test_integration.py::TestRateSelectionFlow::test_rate_selection_guides_matrix_choice
ERROR tests/reconciliation/test_integration.py::TestOrchestratorIntegration::test_orchestrator_initialization
ERROR tests/reconciliation/test_integration.py::TestOrchestratorIntegration::test_single_block_reconciliation
ERROR tests/reconciliation/test_integration.py::TestHighRatePatternBased::test_high_rate_with_pattern_rate_0_8
ERROR tests/reconciliation/test_integration.py::TestHighRatePatternBased::test_high_rate_stress_rate_0_9
ERROR tests/reconciliation/test_orchestrator_multiblock.py::test_orchestrator_reconcile_key_multiblock_happy_path
ERROR tests/reconciliation/test_puncture_patterns.py::TestUntaintedProperty::test_untainted_property_holds
ERROR tests/reconciliation/test_puncture_patterns.py::TestUntaintedProperty::test_forced_puncturing_is_minority
ERROR tests/reconciliation/test_puncture_patterns.py::TestPatternDeterminism::test_same_seed_same_pattern
ERROR tests/reconciliation/test_puncture_patterns.py::TestPatternDeterminism::test_different_seed_different_pattern
ERROR tests/reconciliation/test_puncture_patterns.py::TestRecoverability::test_1step_recoverable_count
ERROR tests/reconciliation/test_puncture_patterns.py::TestPatternProperties::test_pattern_size_matches_frame_size
ERROR tests/reconciliation/test_puncture_patterns.py::TestPatternProperties::test_puncture_count_matches_rate_difference
ERROR tests/reconciliation/test_puncture_patterns.py::TestPatternProperties::test_pattern_values_are_binary
ERROR tests/reconciliation/test_puncture_patterns.py::TestEdgeCases::test_target_rate_must_exceed_mother_rate
ERROR tests/reconciliation/test_puncture_patterns.py::TestEdgeCases::test_target_rate_equal_to_mother_rate
ERROR tests/reconciliation/test_puncture_patterns.py::TestPatternStatistics::test_pattern_coverage_increases_with_rate
ERROR tests/reconciliation/test_puncture_patterns.py::TestPatternStatistics::test_forced_puncturing_increases_at_high_rates
====== 5 failed, 1080 passed, 1 skipped, 19 warnings, 28 errors in 5.36s =======

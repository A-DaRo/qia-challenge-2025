"""
Phase 0 Foundation Tests

Test suite for core data structures, abstract interfaces, exceptions,
constants, and logging infrastructure.

Test IDs:
- test_foundation::test_data_structures
- test_foundation::test_abstract_interfaces
- test_foundation::test_exception_hierarchy
- test_foundation::test_logging
"""

import pytest
import numpy as np
import logging
import inspect
import importlib
from pathlib import Path
import tempfile

from ehok import (
    ObliviousKey,
    MeasurementRecord,
    ProtocolResult,
    EHOKException,
    SecurityException,
    ProtocolError,
    QBERTooHighError,
    ReconciliationFailedError,
    CommitmentVerificationError,
)
from ehok.interfaces import ICommitmentScheme, IReconciliator, IPrivacyAmplifier
from ehok.core import constants
from ehok.utils import setup_ehok_logging, get_logger
from ehok.core.config import ProtocolConfig
from ehok.protocols.alice import AliceBaselineEHOK
from ehok.protocols.bob import BobBaselineEHOK


class TestDataStructures:
    """
    Test ID: test_foundation::test_data_structures
    Requirement: Data structures must enforce type constraints and value ranges.
    """

    def test_oblivious_key_construction_valid(self):
        """
        Test Case 2.1.1: ObliviousKey Construction
        
        Validates that ObliviousKey properly constructs with valid inputs
        and enforces all type and value constraints.
        """
        # Preconditions
        key_value = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        knowledge_mask = np.array([0, 0, 1, 1, 0], dtype=np.uint8)
        security_param = 1e-9
        qber = 0.03
        final_length = 5

        # Operation
        ok = ObliviousKey(
            key_value=key_value,
            knowledge_mask=knowledge_mask,
            security_param=security_param,
            qber=qber,
            final_length=final_length,
        )

        # Postconditions (∀ must hold)
        assert ok.key_value.dtype == np.uint8, "Key must be uint8"
        assert ok.knowledge_mask.dtype == np.uint8, "Mask must be uint8"
        assert len(ok.key_value) == final_length, "Key length must match final_length"
        assert len(ok.knowledge_mask) == final_length, "Mask length must match final_length"
        assert len(ok.key_value) == len(ok.knowledge_mask), "Key and mask must have same length"
        assert np.all((ok.key_value == 0) | (ok.key_value == 1)), "Key values must be 0 or 1"
        assert np.all((ok.knowledge_mask == 0) | (ok.knowledge_mask == 1)), "Mask values must be 0 or 1"
        assert 0 <= ok.qber <= 1, "QBER must be in [0, 1]"
        assert ok.security_param > 0, "Security parameter must be positive"

    def test_oblivious_key_invalid_key_value_type(self):
        """Failure Injection: key_value with wrong dtype should fail."""
        key_value = np.array([0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float64)  # Wrong dtype
        knowledge_mask = np.array([0, 0, 1, 1, 0], dtype=np.uint8)

        with pytest.raises(ValueError, match="Key must be uint8"):
            ObliviousKey(
                key_value=key_value,
                knowledge_mask=knowledge_mask,
                security_param=1e-9,
                qber=0.03,
                final_length=5,
            )

    def test_oblivious_key_invalid_key_value_range(self):
        """Failure Injection: key_value contains invalid values (not 0 or 1)."""
        key_value = np.array([0, 1, 2, 0, 1], dtype=np.uint8)  # Contains 2
        knowledge_mask = np.array([0, 0, 1, 1, 0], dtype=np.uint8)

        with pytest.raises(ValueError, match="Key values must be 0 or 1"):
            ObliviousKey(
                key_value=key_value,
                knowledge_mask=knowledge_mask,
                security_param=1e-9,
                qber=0.03,
                final_length=5,
            )

    def test_oblivious_key_length_mismatch(self):
        """Failure Injection: key_value and knowledge_mask have different lengths."""
        key_value = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        knowledge_mask = np.array([0, 0, 1], dtype=np.uint8)  # Different length

        with pytest.raises(ValueError, match="Key and mask must have same shape"):
            ObliviousKey(
                key_value=key_value,
                knowledge_mask=knowledge_mask,
                security_param=1e-9,
                qber=0.03,
                final_length=5,
            )

    def test_measurement_record_construction_valid(self):
        """Test that MeasurementRecord constructs correctly with valid inputs."""
        # Preconditions
        outcome = 1
        basis = 0
        timestamp = 456.78

        # Operation
        record = MeasurementRecord(outcome=outcome, basis=basis, timestamp=timestamp)

        # Postconditions
        assert record.outcome == 1
        assert record.basis == 0
        assert record.timestamp == 456.78

    def test_measurement_record_invalid_outcome(self):
        """Failure Injection: outcome not in {0, 1} should fail."""
        with pytest.raises(ValueError, match="Outcome must be 0 or 1"):
            MeasurementRecord(outcome=2, basis=0, timestamp=123.45)

    def test_measurement_record_invalid_basis(self):
        """Failure Injection: basis not in {0, 1} should fail."""
        with pytest.raises(ValueError, match="Basis must be 0 \\(Z\\) or 1 \\(X\\)"):
            MeasurementRecord(outcome=0, basis=2, timestamp=123.45)

    def test_protocol_result_construction_valid(self):
        """Test that ProtocolResult constructs correctly."""
        # Create a valid ObliviousKey
        key = ObliviousKey(
            key_value=np.array([1, 0, 1, 1, 0], dtype=np.uint8),
            knowledge_mask=np.array([0, 0, 1, 1, 0], dtype=np.uint8),
            security_param=1e-9,
            qber=0.08,
            final_length=5,
        )

        # Create ProtocolResult
        result = ProtocolResult(
            oblivious_key=key,
            success=True,
            abort_reason=None,
            raw_count=10000,
            sifted_count=5000,
            test_count=500,
            final_count=5,
            qber=0.08,
            execution_time_ms=2345.67,
        )

        # Postconditions
        assert result.success is True
        assert result.oblivious_key is not None
        assert result.abort_reason is None
        assert result.raw_count == 10000
        assert result.final_count == 5

    def test_protocol_result_with_abort(self):
        """Test ProtocolResult when protocol aborts."""
        result = ProtocolResult(
            oblivious_key=None,
            success=False,
            abort_reason="QBER too high",
            raw_count=10000,
            sifted_count=5000,
            test_count=500,
            final_count=0,
            qber=0.15,
            execution_time_ms=1234.56,
        )

        assert result.success is False
        assert result.oblivious_key is None
        assert result.abort_reason == "QBER too high"
        assert result.final_count == 0

    def test_protocol_result_invariants_fail_on_counts(self):
        """Ensure count relations are enforced."""
        with pytest.raises(ValueError, match=r"sifted_count must be >= test_count \+ final_count"):
            ProtocolResult(
                oblivious_key=None,
                success=False,
                abort_reason=None,
                raw_count=100,
                sifted_count=10,
                test_count=5,
                final_count=10,
                qber=0.1,
                execution_time_ms=1.0,
            )

    def test_protocol_result_invariants_fail_on_qber_and_key_length(self):
        """Ensure qber bounds and key length consistency are validated."""
        key = ObliviousKey(
            key_value=np.array([1, 0, 1], dtype=np.uint8),
            knowledge_mask=np.array([0, 0, 0], dtype=np.uint8),
            security_param=1e-9,
            qber=0.01,
            final_length=3,
        )

        with pytest.raises(ValueError, match=r"qber must be in \[0, 1\]"):
            ProtocolResult(
                oblivious_key=key,
                success=True,
                abort_reason=None,
                raw_count=10,
                sifted_count=9,
                test_count=1,
                final_count=3,
                qber=1.5,
                execution_time_ms=1.0,
            )

        with pytest.raises(ValueError, match="final_count must equal oblivious_key.final_length"):
            ProtocolResult(
                oblivious_key=key,
                success=True,
                abort_reason=None,
                raw_count=10,
                sifted_count=9,
                test_count=1,
                final_count=2,
                qber=0.1,
                execution_time_ms=1.0,
            )


class TestAbstractInterfaces:
    """
    Test ID: test_foundation::test_abstract_interfaces
    Requirement: Abstract base classes cannot be instantiated;
                 concrete classes must implement all methods.
    """

    def test_icommitment_scheme_not_instantiable(self):
        """
        Test Case 2.2.1: ICommitmentScheme cannot be instantiated.
        """
        with pytest.raises(TypeError):
            ICommitmentScheme()

    def test_ireconciliator_not_instantiable(self):
        """IReconciliator cannot be instantiated."""
        with pytest.raises(TypeError):
            IReconciliator()

    def test_iprivacy_amplifier_not_instantiable(self):
        """IPrivacyAmplifier cannot be instantiated."""
        with pytest.raises(TypeError):
            IPrivacyAmplifier()

    def test_icommitment_scheme_abstract_methods(self):
        """Verify ICommitmentScheme has required abstract methods."""
        assert hasattr(ICommitmentScheme, "commit")
        assert hasattr(ICommitmentScheme, "verify")
        assert hasattr(ICommitmentScheme, "open_subset")

    def test_ireconciliator_abstract_methods(self):
        """Verify IReconciliator has required abstract methods."""
        assert hasattr(IReconciliator, "select_rate")
        assert hasattr(IReconciliator, "compute_shortening")
        assert hasattr(IReconciliator, "reconcile_block")
        assert hasattr(IReconciliator, "compute_syndrome_block")
        assert hasattr(IReconciliator, "verify_block")
        assert hasattr(IReconciliator, "estimate_leakage_block")

    def test_iprivacy_amplifier_abstract_methods(self):
        """Verify IPrivacyAmplifier has required abstract methods."""
        assert hasattr(IPrivacyAmplifier, "generate_hash_seed")
        assert hasattr(IPrivacyAmplifier, "compress")
        assert hasattr(IPrivacyAmplifier, "compute_final_length")


class TestExceptionHierarchy:
    """
    Test ID: test_foundation::test_exception_hierarchy
    Requirement: Custom exceptions must form correct inheritance chain.
    """

    def test_ehok_exception_inherits_from_exception(self):
        """EHOKException must inherit from Exception."""
        assert issubclass(EHOKException, Exception)

    def test_security_exception_inherits_from_ehok_exception(self):
        """SecurityException must inherit from EHOKException."""
        assert issubclass(SecurityException, EHOKException)

    def test_protocol_error_inherits_from_ehok_exception(self):
        """ProtocolError must inherit from EHOKException."""
        assert issubclass(ProtocolError, EHOKException)

    def test_qber_too_high_error_inherits_from_security_exception(self):
        """QBERTooHighError must inherit from SecurityException."""
        assert issubclass(QBERTooHighError, SecurityException)

    def test_reconciliation_failed_error_inherits_from_protocol_error(self):
        """ReconciliationFailedError must inherit from ProtocolError."""
        assert issubclass(ReconciliationFailedError, ProtocolError)

    def test_matrix_synchronization_error_inherits_from_protocol_error(self):
        """MatrixSynchronizationError must inherit from ProtocolError."""
        from ehok.core.exceptions import MatrixSynchronizationError

        assert issubclass(MatrixSynchronizationError, ProtocolError)

    def test_commitment_verification_error_inherits_from_security_exception(self):
        """CommitmentVerificationError must inherit from SecurityException."""
        assert issubclass(CommitmentVerificationError, SecurityException)

    def test_qber_too_high_error_custom_attributes(self):
        """Test QBERTooHighError stores measured_qber and threshold."""
        measured_qber = 0.15
        threshold = 0.11

        try:
            raise QBERTooHighError(measured_qber, threshold)
        except QBERTooHighError as e:
            assert e.measured_qber == 0.15
            assert e.threshold == 0.11
            assert "0.1500" in str(e)
            assert "0.1100" in str(e)

    def test_all_exceptions_are_catchable_as_ehok_exception(self):
        """All custom exceptions can be caught as EHOKException."""
        exceptions_to_test = [
            SecurityException("test"),
            ProtocolError("test"),
            QBERTooHighError(0.15, 0.11),
            ReconciliationFailedError("test"),
            CommitmentVerificationError("test"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except EHOKException:
                pass  # Should catch all
            else:
                pytest.fail(f"{type(exc).__name__} not catchable as EHOKException")


class TestConstants:
    """
    Test that all constants are defined with correct values.
    """

    def test_protocol_parameters(self):
        """Verify protocol parameters are defined correctly."""
        assert constants.QBER_THRESHOLD == 0.11
        assert constants.TARGET_EPSILON_SEC == 1e-9
        assert constants.TEST_SET_FRACTION == 0.1

    def test_quantum_parameters(self):
        """Verify quantum generation parameters."""
        assert constants.TOTAL_EPR_PAIRS == 10_000
        assert constants.BATCH_SIZE == 5

    def test_network_configuration(self):
        """Verify network configuration constants."""
        assert constants.LINK_FIDELITY_MIN == 0.95
        assert constants.CLASSICAL_TIMEOUT_SEC == 30.0

    def test_ldpc_parameters(self):
        """Verify LDPC code parameters."""
        assert constants.LDPC_FRAME_SIZE == 4096
        assert constants.LDPC_CODE_RATES == (
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
        )
        assert constants.LDPC_DEFAULT_RATE == 0.50
        assert constants.LDPC_CRITICAL_EFFICIENCY == 1.22
        assert constants.LDPC_MAX_ITERATIONS == 60
        assert constants.LDPC_BP_THRESHOLD == 1e-6
        assert 0.50 in constants.LDPC_DEGREE_DISTRIBUTIONS

    def test_logging_configuration(self):
        """Verify logging configuration."""
        assert constants.LOG_LEVEL == "INFO"
        assert constants.LOG_TO_FILE is True


class TestProtocolConfigBinding:
    """Ensure protocol orchestration follows config injection rules."""

    def test_protocols_accept_protocol_config(self):
        for cls in (AliceBaselineEHOK, BobBaselineEHOK):
            sig = inspect.signature(cls.__init__)
            assert "config" in sig.parameters
            param = sig.parameters["config"]
            assert param.default is None or param.default == inspect._empty

    def test_protocols_do_not_instantiate_concretes(self):
        protocol_dir = Path(__file__).parent.parent / "protocols"
        forbidden_tokens = (
            "SHA256Commitment(",
            "LDPCReconciliator(",
            "ToeplitzAmplifier(",
        )
        for path in protocol_dir.glob("*.py"):
            text = path.read_text()
            for token in forbidden_tokens:
                assert (
                    token not in text
                ), f"Direct instantiation {token} found in {path.name}"


class TestDocstrings:
    """Validate presence of docstrings on public classes."""

    MODULES = [
        "ehok.core.config",
        "ehok.core.data_structures",
        "ehok.core.sifting",
        "ehok.interfaces.commitment",
        "ehok.interfaces.reconciliation",
        "ehok.interfaces.privacy_amplification",
        "ehok.interfaces.sampling_strategy",
        "ehok.interfaces.noise_estimator",
        "ehok.implementations.factories",
    ]

    def test_public_classes_have_docstrings(self):
        for module_name in self.MODULES:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module.__name__:
                    continue
                if name.startswith("_"):
                    continue
                assert obj.__doc__ and obj.__doc__.strip(), (
                    f"Missing docstring on class {module_name}.{name}"
                )


class TestLogging:
    """
    Test ID: test_foundation::test_logging
    Requirement: Logging must use SquidASM LogManager, never print().
    """

    def test_get_logger_returns_logger(self):
        """
        Test Case 2.4.1: Logger Creation
        Verify get_logger returns a valid logger instance.
        """
        logger = get_logger("test_module")

        # Postconditions
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert "ehok.test_module" in logger.name

    def test_setup_ehok_logging_console_only(self):
        """Test setup_ehok_logging with console output only."""
        logger = setup_ehok_logging(log_dir=None, log_level="WARNING")

        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.WARNING

    def test_setup_ehok_logging_with_file(self):
        """Test setup_ehok_logging with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = setup_ehok_logging(log_dir=log_dir, log_level="DEBUG")

            assert isinstance(logger, logging.Logger)
            assert logger.level == logging.DEBUG

            # Verify log file was created
            log_file = log_dir / "ehok_protocol.log"
            assert log_file.exists()

    def test_hierarchical_logger_names(self):
        """Test that hierarchical logger names are properly constructed."""
        quantum_logger = get_logger("quantum.measurement")
        protocols_logger = get_logger("protocols.alice")

        assert "ehok.quantum.measurement" in quantum_logger.name
        assert "ehok.protocols.alice" in protocols_logger.name

    def test_no_print_statements_in_production_code(self):
        """
        Test Case 2.4.2: No Print Statements in Codebase
        
        Verify that production code (non-test files) does not contain print().
        """
        import subprocess
        import os

        # Get the ehok directory
        ehok_dir = Path(__file__).parent.parent

        # Search for print statements, excluding test files and comments
        result = subprocess.run(
            [
                "grep",
                "-r",
                "print(",
                str(ehok_dir),
                "--include=*.py",
                "--exclude=test_*.py",
            ],
            capture_output=True,
            text=True,
        )

        # Filter out commented lines
        print_statements = [
            line
            for line in result.stdout.split("\n")
            if line and not line.strip().startswith("#")
        ]

        # Should find no print statements
        assert (
            len(print_statements) == 0
        ), f"Found print() statements in production code:\n" + "\n".join(print_statements)


class TestPhase0Integration:
    """
    Integration tests for Phase 0 components working together.
    """

    def test_data_structures_with_exceptions(self):
        """Test that data structures properly work with exception handling."""
        # Create an aborted protocol result
        try:
            # Simulate QBER check that fails
            measured_qber = 0.15
            if measured_qber > constants.QBER_THRESHOLD:
                raise QBERTooHighError(measured_qber, constants.QBER_THRESHOLD)
        except QBERTooHighError as e:
            result = ProtocolResult(
                oblivious_key=None,
                success=False,
                abort_reason=str(e),
                raw_count=10000,
                sifted_count=5000,
                test_count=500,
                final_count=0,
                qber=measured_qber,
                execution_time_ms=1000.0,
            )

            assert result.success is False
            assert "0.1500 exceeds threshold 0.1100" in result.abort_reason

    def test_logging_with_exceptions(self):
        """Test that exceptions are properly logged."""
        logger = get_logger("test.integration")

        try:
            raise QBERTooHighError(0.15, 0.11)
        except QBERTooHighError as e:
            # This should not raise an exception
            logger.error(f"Protocol aborted: {e}")
            # Success if no exception raised


# Mark this module as containing tests
__all__ = [
    "TestDataStructures",
    "TestAbstractInterfaces",
    "TestExceptionHierarchy",
    "TestConstants",
    "TestLogging",
    "TestPhase0Integration",
]

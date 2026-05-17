"""
Tests for custom exception classes.
"""

import pytest

from incerto.exceptions import (
    CalibrationError,
    ConfigurationError,
    DataError,
    IncertoError,
    NotFittedError,
    SerializationError,
)


def test_incerto_error():
    """Test base IncertoError exception."""
    with pytest.raises(IncertoError):
        raise IncertoError("Test error")


def test_incerto_error_inheritance():
    """Test that IncertoError inherits from Exception."""
    assert issubclass(IncertoError, Exception)


def test_not_fitted_error():
    """Test NotFittedError exception."""
    with pytest.raises(NotFittedError):
        raise NotFittedError("Model not fitted")


def test_not_fitted_error_inheritance():
    """Test that NotFittedError inherits from IncertoError."""
    assert issubclass(NotFittedError, IncertoError)
    assert issubclass(NotFittedError, Exception)


def test_calibration_error():
    """Test CalibrationError exception."""
    with pytest.raises(CalibrationError):
        raise CalibrationError("Calibration failed")


def test_calibration_error_inheritance():
    """Test that CalibrationError inherits from IncertoError."""
    assert issubclass(CalibrationError, IncertoError)


def test_serialization_error():
    """Test SerializationError exception."""
    with pytest.raises(SerializationError):
        raise SerializationError("Failed to serialize")


def test_serialization_error_inheritance():
    """Test that SerializationError inherits from IncertoError."""
    assert issubclass(SerializationError, IncertoError)


def test_configuration_error():
    """Test ConfigurationError exception."""
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Invalid configuration")


def test_configuration_error_inheritance():
    """Test that ConfigurationError inherits from IncertoError."""
    assert issubclass(ConfigurationError, IncertoError)


def test_data_error():
    """Test DataError exception."""
    with pytest.raises(DataError):
        raise DataError("Invalid data")


def test_data_error_inheritance():
    """Test that DataError inherits from IncertoError."""
    assert issubclass(DataError, IncertoError)


def test_exception_catching():
    """Test that specific exceptions can be caught as IncertoError."""
    try:
        raise NotFittedError("Test")
    except IncertoError as e:
        assert isinstance(e, NotFittedError)
        assert isinstance(e, IncertoError)


def test_exception_messages():
    """Test that exception messages are preserved."""
    message = "This is a test error message"

    try:
        raise CalibrationError(message)
    except CalibrationError as e:
        assert str(e) == message


def test_exception_chaining():
    """Test exception chaining with from clause."""
    original_error = ValueError("Original error")

    try:
        raise SerializationError("Serialization failed") from original_error
    except SerializationError as e:
        assert e.__cause__ is original_error
        assert isinstance(e.__cause__, ValueError)

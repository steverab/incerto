"""
Custom exceptions for the incerto library.

This module defines library-specific exceptions for better error handling
and more informative error messages.
"""


class IncertoError(Exception):
    """
    Base exception class for all incerto-specific errors.

    All custom exceptions in the incerto library inherit from this class,
    making it easy to catch any incerto-related error.

    Example:
        >>> try:
        ...     calibrator.predict(logits)  # Without fitting first
        ... except IncertoError as e:
        ...     print(f"Incerto error: {e}")
    """

    pass


class NotFittedError(IncertoError):
    """
    Raised when calling predict/score/transform before fit.

    This error is raised when attempting to use a model or estimator
    that has not been fitted yet.

    Example:
        >>> calibrator = TemperatureScaling()
        >>> calibrator.predict(logits)  # Raises NotFittedError
    """

    pass


class CalibrationError(IncertoError):
    """
    Raised when calibration fails.

    This can happen when optimization fails, data is invalid, or
    calibration parameters cannot be found.

    Example:
        >>> calibrator.fit(invalid_logits, labels)  # Raises CalibrationError
    """

    pass


class SerializationError(IncertoError):
    """
    Raised when saving or loading state fails.

    This can happen when the state dict is corrupted, incompatible,
    or the file path is invalid.

    Example:
        >>> calibrator.load_state_dict(corrupted_state)  # Raises SerializationError
    """

    pass


class ConfigurationError(IncertoError):
    """
    Raised when configuration parameters are invalid.

    This can happen when incompatible parameters are provided or
    required parameters are missing.

    Example:
        >>> calibrator = TemperatureScaling(temperature=-1.0)  # Raises ConfigurationError
    """

    pass


class DataError(IncertoError):
    """
    Raised when input data is invalid.

    This can happen when data shapes are incompatible, values are
    out of range, or data types are incorrect.

    Example:
        >>> calibrator.fit(wrong_shape_logits, labels)  # Raises DataError
    """

    pass


__all__ = [
    "IncertoError",
    "NotFittedError",
    "CalibrationError",
    "SerializationError",
    "ConfigurationError",
    "DataError",
]

"""Errors raised by ChemEAGLE vision backends."""


class VisionBackendError(RuntimeError):
    """Base class for local and remote vision failures."""


class VisionConfigurationError(VisionBackendError):
    """The selected vision backend is not configured correctly."""


class VisionProcessError(VisionBackendError):
    """A remote worker could not be started or exited unexpectedly."""


class VisionProtocolError(VisionBackendError):
    """The remote worker emitted an invalid or mismatched response."""


class VisionTimeoutError(VisionBackendError):
    """A remote worker request exceeded its configured timeout."""


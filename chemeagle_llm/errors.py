"""Domain exceptions shared by all ChemEAGLE LLM backends."""


class BackendError(RuntimeError):
    """Base class for backend failures."""


class BackendConfigurationError(BackendError):
    """The selected backend is missing required configuration."""


class AuthenticationError(BackendError):
    """Authentication is absent, expired, or incompatible with the route."""


class BackendTimeoutError(BackendError):
    """A backend request exceeded its configured deadline."""


class BackendCancelledError(BackendError):
    """A backend request was explicitly cancelled by its caller."""


class BackendProcessError(BackendError):
    """A managed backend process exited or violated its protocol."""


class BackendRateLimitError(BackendError):
    """The selected account or service reached a usage limit."""


class InvalidResponseError(BackendError):
    """The model returned data that could not be validated."""


class UnsupportedCapabilityError(BackendError):
    """A requested capability is not implemented by this backend."""


class ToolExecutionError(BackendError):
    """A model-selected, explicitly registered tool failed."""


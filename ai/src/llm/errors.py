class LLMError(Exception):
    """Base exception for the LLM abstraction layer."""


class LLMConfigurationError(LLMError):
    """Raised when a provider is missing required configuration."""


class LLMProviderError(LLMError):
    """Raised when an upstream provider request fails."""

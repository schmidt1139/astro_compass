"""Custom exceptions for the astro_compass core module."""


class FirstGuessException(Exception):
    """Exception raised when initial co-state guess cannot be found."""
    pass

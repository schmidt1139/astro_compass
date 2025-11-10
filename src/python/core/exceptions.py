"""Custom exceptions for the astro_compass core module."""


class FirstGuessException(Exception):
    """Exception raised when initial co-state guess cannot be found."""
    pass


class SpacecraftCollisionException(Exception):
    """Exception raised when spacecraft gets too close to central body."""
    pass


class LowMassException(Exception):
    """Exception raised when spacecraft mass gets too low during integration."""
    pass

class TimeoutException(Exception):
    """Exception raised when a trajectory computation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Trajectory computation timed out")

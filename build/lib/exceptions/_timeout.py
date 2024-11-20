# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "TimeoutError",
    "timeout",
]

import signal


class TimeoutError(Exception):
    """Custom exception to be raised when a timeout occurs."""

    pass


def timeout(seconds=10, error_message="Function call timed out"):
    """
    Decorator that limits the execution time of a function.

    Parameters:
    -----------
    seconds : int
        The number of seconds before the function is terminated with a timeout error.
    error_message : str
        The message to include in the TimeoutError raised when the timeout occurs.
    """

    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            # Set the signal handler for SIGALRM to our custom handler
            signal.signal(signal.SIGALRM, _handle_timeout)
            # Schedule the SIGALRM signal to be sent after `seconds` seconds
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result

        return wrapper

    return decorator

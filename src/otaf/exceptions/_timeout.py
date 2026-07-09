from __future__ import annotations

__author__ = "Kramer84"
__all__ = ["timeout"]
import signal


def timeout(seconds=10, error_message="Function call timed out"):
    """Limit the execution time of a function using a decorator.

    Parameters
    ----------
    seconds : int
        The number of seconds before the function is terminated with a timeout error.
    error_message : str
        The message to include in the TimeoutError raised when the timeout occurs.
    """

    def decorator(func):

        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

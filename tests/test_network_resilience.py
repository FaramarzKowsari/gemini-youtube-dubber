from dubber.rate_limit import is_retryable_gemini_error

def test_connection_reset_by_peer():
    assert is_retryable_gemini_error(RuntimeError("APIConnectionError: [Errno 104] Connection reset by peer"))

def test_timeout():
    assert is_retryable_gemini_error(TimeoutError("timed out"))

def test_nested_errno_104():
    try:
        try:
            raise OSError(104, "Connection reset by peer")
        except OSError as inner:
            raise RuntimeError("SDK wrapper") from inner
    except RuntimeError as outer:
        assert is_retryable_gemini_error(outer)

def test_400_is_not_retryable():
    assert not is_retryable_gemini_error(RuntimeError("Error code: 400 invalid request"))

import time


def test_should_timeout():
    """This test intentionally sleeps longer than the TEST_TIMEOUT to verify the alarm."""
    # Sleep for 10 seconds; our TEST_TIMEOUT will be set to a smaller value when running.
    time.sleep(10)
    assert True

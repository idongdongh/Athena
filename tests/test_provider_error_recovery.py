"""Provider/API 错误恢复测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.provider_error_recovery import (
    ProviderRequestFailed,
    ProviderRequestInterrupted,
    _interruptible_wait,
    classify_provider_error,
    request_with_retries,
)


class FakeAPIError(Exception):
    def __init__(self, message, status_code=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


class APITimeoutError(Exception):
    pass


class ProviderErrorRecoveryTests(unittest.TestCase):
    def test_classifies_retryable_and_permanent_errors(self):
        self.assertEqual(classify_provider_error(APITimeoutError("slow")).kind, "timeout")
        self.assertTrue(classify_provider_error(FakeAPIError("busy", 503)).retryable)
        self.assertEqual(
            classify_provider_error(FakeAPIError("conflict", 409)).kind,
            "transient_conflict",
        )
        self.assertFalse(classify_provider_error(FakeAPIError("bad key", 401)).retryable)

    def test_retries_until_success_without_real_sleep(self):
        attempts = 0
        retries = []

        def request():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise FakeAPIError("busy", 503)
            return "ok"

        result = request_with_retries(
            request,
            sleep=lambda _seconds: None,
            on_retry=lambda error, retry, wait: retries.append((error.kind, retry, wait)),
        )
        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual([item[:2] for item in retries], [("server_error", 1), ("server_error", 2)])

    def test_retry_after_header_controls_wait(self):
        waits = []
        attempts = 0

        def request():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise FakeAPIError("limited", 429, {"Retry-After": "2"})
            return "ok"

        with patch("agent.provider_error_recovery._interruptible_wait") as wait:
            request_with_retries(
                request,
                on_retry=lambda _error, _retry, seconds: waits.append(seconds),
            )
        self.assertEqual(waits, [2.0])
        self.assertEqual(wait.call_args.args[0], 2.0)

    def test_permanent_error_is_not_retried(self):
        attempts = 0

        def request():
            nonlocal attempts
            attempts += 1
            raise FakeAPIError("bad key", 401)

        with self.assertRaises(ProviderRequestFailed) as raised:
            request_with_retries(request, sleep=lambda _seconds: None)
        self.assertEqual(attempts, 1)
        self.assertEqual(raised.exception.error.kind, "authentication")

    def test_retry_exhaustion_raises_classified_error(self):
        attempts = 0

        def request():
            nonlocal attempts
            attempts += 1
            raise FakeAPIError("busy", 503)

        with self.assertRaises(ProviderRequestFailed):
            request_with_retries(request, max_retries=2, sleep=lambda _seconds: None)
        self.assertEqual(attempts, 3)

    def test_interrupt_stops_backoff(self):
        with self.assertRaises(ProviderRequestInterrupted):
            request_with_retries(
                lambda: (_ for _ in ()).throw(FakeAPIError("busy", 503)),
                is_interrupted=lambda: True,
                sleep=lambda _seconds: None,
            )

    def test_zero_wait_still_honors_interrupt(self):
        with self.assertRaises(ProviderRequestInterrupted):
            _interruptible_wait(
                0,
                is_interrupted=lambda: True,
                sleep=lambda _seconds: None,
            )

    def test_interrupt_during_final_sleep_stops_retry(self):
        interrupted = False

        def sleep(_seconds):
            nonlocal interrupted
            interrupted = True

        with self.assertRaises(ProviderRequestInterrupted):
            _interruptible_wait(
                0.1,
                is_interrupted=lambda: interrupted,
                sleep=sleep,
            )


if __name__ == "__main__":
    unittest.main()

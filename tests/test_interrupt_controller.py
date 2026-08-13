"""InterruptController 的线程可见性测试。"""

import threading
import unittest

from agent.interrupt_controller import (
    InterruptController,
    ToolExecutionCancelled,
    interrupt_controller,
    run_interruptible,
)


class InterruptControllerTests(unittest.TestCase):
    def test_request_is_visible_to_waiting_thread_and_clear_resets_state(self):
        controller = InterruptController()
        observed = []

        waiter = threading.Thread(target=lambda: observed.append(controller.wait(1)))
        waiter.start()
        controller.request()
        waiter.join(1)

        self.assertEqual(observed, [True])
        self.assertTrue(controller.is_requested())
        controller.clear()
        self.assertFalse(controller.is_requested())

    def test_run_interruptible_calls_cancel_callback_once(self):
        release_worker = threading.Event()
        cancel_calls = []
        interrupt_controller.clear()
        timer = threading.Timer(0.02, interrupt_controller.request)
        timer.start()
        try:
            with self.assertRaises(ToolExecutionCancelled):
                run_interruptible(
                    lambda: release_worker.wait(1),
                    on_cancel=lambda: (
                        cancel_calls.append("cancelled"),
                        release_worker.set(),
                    ),
                    poll_interval=0.005,
                )
        finally:
            timer.cancel()
            release_worker.set()
            interrupt_controller.clear()

        self.assertEqual(cancel_calls, ["cancelled"])


if __name__ == "__main__":
    unittest.main()

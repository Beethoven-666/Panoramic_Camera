"""CLI-only signal handlers request graceful capture stop and restore handlers."""

from contextlib import contextmanager
import signal
import threading


@contextmanager
def cooperative_capture_signals(args):
    event = getattr(args, "cancel_event", None) or threading.Event()
    args.cancel_event = event
    previous = {}

    def stop(number, _frame):
        if not event.is_set():
            args.stop_reason = signal.Signals(number).name
            event.set()

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, stop)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)

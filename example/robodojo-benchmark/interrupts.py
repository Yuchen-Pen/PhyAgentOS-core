"""Translate terminal interrupts into a single authoritative Core stop request."""

import asyncio
from contextlib import contextmanager
import signal


@contextmanager
def graceful_signals():
    loop = asyncio.get_running_loop()
    event = asyncio.Event()
    previous = {}

    def request():
        if not event.is_set():
            print('Stop requested. Waiting for the native episode boundary; the service remains managed.', flush=True)
            event.set()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            loop.add_signal_handler(signum, request)
        yield event
    finally:
        for signum, handler in previous.items():
            loop.remove_signal_handler(signum)
            signal.signal(signum, handler)

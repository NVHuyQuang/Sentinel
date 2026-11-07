"""Virtual-time asyncio loop: sleeps advance a simulated clock, so a 25 minute run takes seconds.

Same coroutines run on the real loop in live mode; only the clock differs.
"""

import asyncio


class VirtualLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self._vt = 0.0
        self._clock_resolution = 1e-9  # Windows monotonic resolution (15 ms) would blur latencies
        real_select = self._selector.select

        def select(timeout=None):
            if timeout:
                self._vt += timeout  # jump to the next scheduled handle instead of sleeping
            return real_select(0)

        self._selector.select = select

    def time(self) -> float:
        return self._vt


def run_virtual(coro):
    loop = VirtualLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

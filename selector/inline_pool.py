"""An in-process stand-in for ProcessPoolExecutor, shared by the map builder
and the runtime verifier so both see the same execution."""

from __future__ import annotations


class InlineProcessPool:
    """Stand-in for ProcessPoolExecutor that runs each task at submit time.

    Arguments and results take a pickle round trip, as they would through a
    spawned worker, so a task cannot mutate the caller's objects. Coverage of
    the task then lands on the test that submitted it, and no interpreter is
    spawned.
    """

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        self._initializer = initializer
        self._initargs = initargs
        self._ready = initializer is None

    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures
        import pickle as _pickle

        future = concurrent.futures.Future()
        try:
            if not self._ready:
                self._ready = True
                self._initializer(*_pickle.loads(_pickle.dumps(self._initargs)))
            args, kwargs = _pickle.loads(_pickle.dumps((args, kwargs)))
            future.set_result(_pickle.loads(_pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:  # noqa: BLE001 - mirrors a worker's failure
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        return (f.result() for f in futures)

    def shutdown(self, wait=True, *, cancel_futures=False):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def install_inline_pool() -> None:
    import concurrent.futures
    import concurrent.futures.process

    concurrent.futures.ProcessPoolExecutor = InlineProcessPool
    concurrent.futures.process.ProcessPoolExecutor = InlineProcessPool

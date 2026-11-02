"""Opt-in fresh500 limits with the historical credit-trace-v1 collector unchanged."""
from contextlib import contextmanager

from src.tasks.hammer.rl.credit_training_trace import _CreditTrace, _need


_LIMIT = 160 * 1024**2


class _LongCreditTrace(_CreditTrace):
    def __init__(self, runner, path, world_ids, max_bytes, max_updates):
        _need(type(max_updates) is int and max_updates == 500,
              'fresh500 trace requires exactly500 updates')
        _need(type(max_bytes) is int and 0 < max_bytes <= _LIMIT,
              'fresh500 trace byte cap must be1..160MiB')
        # Admit the complete historical validation, then extend only this instance.
        super().__init__(runner, path, world_ids, min(max_bytes, 32 * 1024**2), 100)
        self.max_updates, self.cap = max_updates, max_bytes


@contextmanager
def attach_credit_training_trace(runner, output_path, world_ids=(0, 1),
                                 max_bytes=_LIMIT, max_updates=500):
    """Retain partial evidence and restore lookup semantics on every exit."""
    trace = _LongCreditTrace(runner, output_path, world_ids, max_bytes, max_updates)
    try:
        trace.start()
        yield trace
        _need(not trace.buffer, 'trace context ended with an unjoined partial rollout')
    finally:
        trace.close()

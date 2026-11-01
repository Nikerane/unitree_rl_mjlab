"""Pure tensor episode ledger for the opt-in 031 native depth rewards.

Amounts are raw event units, before reward-manager dt scaling. The caller owns
the physics clock and must supply the genuine 200ms native horizon mask.
"""

import torch


def _potential(depth_m: torch.Tensor) -> torch.Tensor:
    # Set the endpoint explicitly: float32 subtraction/division can otherwise
    # leave the nominal 30mm threshold one ULP short of its full unit budget.
    scaled = ((depth_m - .004) / (.030 - .004)).clamp(0., 1.)
    return torch.where(depth_m >= .030, 1., scaled)


class DepthCreditLedger031:
    """Observe every substep, collect once per policy boundary, reset explicitly.

    ``observe`` checks tensor metadata only; nonfinite samples are latched on
    device and rejected at ``collect``. Finite depths follow the native clamp
    to [0, .032] metres. No simulator, policy, RNG or host scalar reads occur in
    that substep path. Returned dictionaries and ``last_snapshot`` contain
    defensive tensor clones, so readers cannot mutate ledger state or receipts
    held by another reader. ``last_snapshot`` survives episode resets.

    Collections are cached until an observation or nonempty reset. Duplicate
    readers must pass the same masks; changing masks within a cached boundary
    is rejected. A native terminal closes both additions until explicit reset.
    """

    def __init__(self, num_envs, device, *, progress_enabled, completion_enabled):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if type(progress_enabled) is not bool or type(completion_enabled) is not bool:
            raise ValueError("component switches must be booleans")
        self.num_envs = num_envs
        self.progress_enabled = progress_enabled
        self.completion_enabled = completion_enabled
        self._current = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.device = self._current.device
        self._peak = torch.zeros_like(self._current)
        self._credited = torch.zeros_like(self._current)
        self._initial_potential = torch.zeros_like(self._current)
        self._progress_total = torch.zeros_like(self._current)
        self._completion_total = torch.zeros_like(self._current)
        self._forfeited_total = torch.zeros_like(self._current)
        self._external_forfeited = torch.zeros_like(self._current)
        self._attained = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._closed = torch.zeros_like(self._attained)
        self._invalid = torch.zeros_like(self._attained)
        self._generation = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self._resets = torch.zeros_like(self._generation)
        self._snapshot = None
        self._cache_valid = False

    def _check_tensor(self, value, name, shape, *, boolean=False):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.device != self.device
                or (value.dtype != torch.bool if boolean else not value.is_floating_point())):
            kind = "boolean" if boolean else "floating"
            raise ValueError(f"{name} must be a {kind} tensor of shape {shape} on {self.device}")

    @property
    def generation(self):
        return self._generation.clone()

    @property
    def last_snapshot(self):
        if self._snapshot is None:
            return None
        return {key: value.clone() for key, value in self._snapshot.items()}

    @torch.no_grad()
    def observe(self, depth_m):
        """Consume one floating sample per world without synchronizing the host."""
        self._check_tensor(depth_m, "depth_m", (self.num_envs,))
        self._invalid.logical_or_(~torch.isfinite(depth_m))
        depth = depth_m.clamp(0., .032).to(dtype=self._current.dtype)
        self._current.copy_(torch.where(self._closed, self._current, depth))
        self._peak.copy_(torch.maximum(self._peak, self._current))
        self._attained.logical_or_(self._peak >= .030)
        self._cache_valid = False

    @torch.no_grad()
    def collect(self, *, horizon_complete, native_reset):
        """Return cloned raw-unit accounting at a native policy boundary.

        ``horizon_complete`` and ``native_reset`` are boolean (num_envs,)
        tensors on the ledger device. A horizon without native reset is an
        invalid boundary. Timeout is ``native_reset & ~horizon_complete``;
        simultaneous timeout and horizon therefore pays completion.
        """
        for name, value in (("horizon_complete", horizon_complete), ("native_reset", native_reset)):
            self._check_tensor(value, name, (self.num_envs,), boolean=True)
        if (horizon_complete & ~native_reset).any().item():
            raise ValueError("horizon_complete requires native_reset")
        if self._invalid.any().item():
            raise ValueError("nonfinite depth observed; reset affected worlds before collecting")
        if self._cache_valid:
            if (not torch.equal(horizon_complete, self._snapshot["horizon_complete"])
                    or not torch.equal(native_reset, self._snapshot["native_reset"])):
                raise ValueError("boundary masks changed without an observation or reset")
            return self.last_snapshot

        potential = _potential(self._peak)
        progress = (potential - self._credited).clamp_min(0.)
        progress = torch.where(~self._closed & self.progress_enabled, progress, 0.)
        self._credited.copy_(potential)
        self._progress_total.add_(progress)
        pending = self._attained & ~self._closed & self.completion_enabled
        due = (pending & horizon_complete).to(self._current.dtype)
        forfeited = (pending & native_reset & ~horizon_complete).to(self._current.dtype)
        self._completion_total.add_(due)
        self._forfeited_total.add_(forfeited)
        self._closed.logical_or_(native_reset)

        snapshot = dict(
            current_depth_m=self._current,
            peak_depth_m=self._peak,
            attained=self._attained,
            progress_potential=potential,
            progress_initial_potential=self._initial_potential,
            progress_paid=progress,
            progress_paid_total=self._progress_total,
            completion_due=due,
            completion_paid=due,
            completion_paid_total=self._completion_total,
            completion_forfeited=forfeited,
            completion_forfeited_total=self._forfeited_total,
            completion_pending=(self._attained & ~self._closed & self.completion_enabled).to(self._current.dtype),
            completion_closed=self._closed,
            horizon_complete=horizon_complete,
            native_reset=native_reset,
            generation=self._generation,
            externally_forfeited_completion=self._external_forfeited,
            resets_since_collect=self._resets,
        )
        self._snapshot = {key: value.clone() for key, value in snapshot.items()}
        self._external_forfeited.zero_()
        self._resets.zero_()
        self._cache_valid = True
        return self.last_snapshot

    @torch.no_grad()
    def reset(self, env_ids=None, *, initial_depth_m=None):
        """Reset selected episodes using the NEW scene's optional depth sample.

        ``env_ids`` is None (all worlds) or a unique 1D int64 device tensor.
        ``initial_depth_m`` is a floating device tensor of shape (num_selected,)
        in the same order as env_ids, defaulting to zero. It must be finite and
        is clamped to [0, .032]. Initial potential is already credited, without
        paying progress; initial geometric attainment remains completion-eligible.
        External reset discards pending completion and never pays either term.
        """
        if env_ids is None:
            ids = slice(None)
            count = self.num_envs
        else:
            if (not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1
                    or env_ids.dtype != torch.int64 or env_ids.device != self.device):
                raise ValueError("env_ids must be a 1D int64 tensor on the ledger device")
            if ((env_ids < 0) | (env_ids >= self.num_envs)).any().item():
                raise ValueError("env_ids out of range")
            if env_ids.unique().numel() != env_ids.numel():
                raise ValueError("env_ids must be unique")
            ids, count = env_ids, env_ids.numel()
        if initial_depth_m is None:
            initial = torch.zeros(count, device=self.device, dtype=self._current.dtype)
        else:
            self._check_tensor(initial_depth_m, "initial_depth_m", (count,))
            if not torch.isfinite(initial_depth_m).all().item():
                raise ValueError("initial_depth_m must be finite")
            initial = initial_depth_m.clamp(0., .032).to(dtype=self._current.dtype)
        if count == 0:
            return

        self._external_forfeited[ids] += (
            self._attained[ids] & ~self._closed[ids] & self.completion_enabled
        ).to(self._current.dtype)
        self._current[ids] = initial
        self._peak[ids] = initial
        self._credited[ids] = _potential(initial)
        self._initial_potential[ids] = self._credited[ids]
        self._attained[ids] = initial >= .030
        self._closed[ids] = False
        self._invalid[ids] = False
        self._progress_total[ids] = 0.
        self._completion_total[ids] = 0.
        self._forfeited_total[ids] = 0.
        self._generation[ids] += 1
        self._resets[ids] += 1
        self._cache_valid = False

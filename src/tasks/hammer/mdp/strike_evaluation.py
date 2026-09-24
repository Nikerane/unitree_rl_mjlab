"""Evaluation-only, physics-sample evidence; never a reward or constraint source."""

from __future__ import annotations

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.metrics_manager import MetricsTermCfg


LOAD_SIGNAL_SEMANTICS = (
    "per-physics SubstepImpulseAccumulator.impulse: "
    "max(current rolling Lambda, current control-step peak)"
)


class StrikeEvaluationCollector:
    """Independent onset window and path/load counters, with subset-safe reset.

    The load input is the authoritative per-physics max(current rolling Lambda,
    current control-step peak), including pre-onset ring history and any retained
    pre-onset control-step peak. Exceedance durations describe this peak-held
    signal, not instantaneous rolling-Lambda exceedance durations.
    The onset boolean is the native tracker's accepted, latched onset identity.
    """

    def __init__(self, num_envs, device, physics_dt, caps, *, window_samples=None):
        self.num_envs, self.dt = num_envs, physics_dt
        self.window = round(.05 / physics_dt) if window_samples is None else window_samples
        if self.window < 1 or not math.isclose(self.window * physics_dt, .05):
            raise ValueError("common window must span exactly 50 ms")
        self.caps = torch.as_tensor(caps, device=device, dtype=torch.float64)
        if self.caps.shape != (6,) or not torch.isfinite(self.caps).all() or not (self.caps > 0).all():
            raise ValueError("require six positive finite load caps")
        def zeros(*shape, dtype=torch.float64):
            return torch.zeros(num_envs, *shape, device=device, dtype=dtype)
        self.samples = zeros(dtype=torch.long)
        self.onset = zeros(dtype=torch.long)
        self.window_count = zeros(dtype=torch.long)
        self.integrals = zeros(3)
        self.load_peaks = zeros(2, 6)
        self.load_exceedances = zeros(2, 6, dtype=torch.long)
        self.path_count = zeros(2, dtype=torch.long)
        self.path_squares = zeros(2)
        self.path_max = zeros(2)
        self.bin_count = zeros(16, dtype=torch.long)
        self.bin_squares = zeros(16)
        self.bin_max = zeros(16)
        self.approach_height_min = zeros()
        self.approach_height_min.fill_(1.)
        self.approach_height_max = zeros()
        self.success_sample = zeros(dtype=torch.long)

    def reset(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        for name in ("samples", "onset", "window_count", "integrals", "load_peaks",
                     "load_exceedances", "path_count", "path_squares", "path_max",
                     "bin_count", "bin_squares", "bin_max", "approach_height_max", "success_sample"):
            getattr(self, name)[ids] = 0
        self.approach_height_min[ids] = 1

    def sample(self, *, started, contact, force_world, axis, load, approach_height, error_m):
        self.samples += 1
        self.onset.copy_(torch.where(started & (self.onset == 0), self.samples, self.onset))
        active = (self.onset > 0) & (self.window_count < self.window)
        self.window_count += active.long()
        signed = (force_world * axis).sum(-1)
        transverse = torch.linalg.vector_norm(force_world - signed[:, None] * axis, dim=-1)
        force = torch.stack((signed.clamp_min(0), signed, transverse), dim=-1)
        self.integrals += torch.where((active & contact)[:, None], force * self.dt, 0.)
        for index, mask in enumerate((torch.ones_like(active), active)):
            self.load_peaks[:, index] = torch.maximum(self.load_peaks[:, index],
                                                      torch.where(mask[:, None], load, 0.))
            self.load_exceedances[:, index] += ((load > self.caps) & mask[:, None]).long()
        precontact = self.onset == 0
        for index, mask in enumerate((precontact, ~precontact)):
            self.path_count[:, index] += mask.long()
            self.path_squares[:, index] += torch.where(mask, error_m.double().square(), 0.)
            self.path_max[:, index] = torch.maximum(self.path_max[:, index], torch.where(mask, error_m, 0.))
        approach_height = approach_height.clamp(0, 1)
        self.approach_height_min = torch.minimum(self.approach_height_min, torch.where(precontact, approach_height, 1.))
        self.approach_height_max = torch.maximum(self.approach_height_max, torch.where(precontact, approach_height, 0.))
        bins = (approach_height * 16).long().clamp_max(15)
        worlds = torch.arange(self.num_envs, device=self.samples.device)
        self.bin_count[worlds, bins] += precontact.long()
        self.bin_squares[worlds, bins] += torch.where(precontact, error_m.double().square(), 0.)
        self.bin_max[worlds, bins] = torch.maximum(self.bin_max[worlds, bins], torch.where(precontact, error_m, 0.))

    def success_termination(self, native_success):
        """Latch success at its native policy check, then defer only its reset."""
        self.success_sample.copy_(torch.where(native_success & (self.success_sample == 0),
                                              self.samples, self.success_sample))
        # A success without accepted onset cannot produce a window; do not hang it.
        return (self.success_sample > 0) & ((self.window_count >= self.window) | (self.onset == 0))

    def records(self, stop_reasons=None):
        """Return detached JSON evidence. Unfinished observations are censored."""
        def path(n, squares, peak):
            n = int(n)
            return dict(sample_count=n, rms_m=math.sqrt(float(squares) / n) if n else None,
                        max_m=float(peak) if n else None)
        records = []
        for i in range(self.num_envs):
            count = int(self.window_count[i])
            complete = count == self.window
            pre = path(self.path_count[i, 0], self.path_squares[i, 0], self.path_max[i, 0])
            bins = [path(self.bin_count[i, b], self.bin_squares[i, b], self.bin_max[i, b]) for b in range(16)]
            skipped = [b for b, entry in enumerate(bins) if not entry["sample_count"]]
            visited = [entry for entry in bins if entry["sample_count"]]
            pre.update(bins=bins, skipped_bins=skipped, coverage=(16 - len(skipped)) / 16,
                       observed_bin_rms_m=math.sqrt(sum(entry["rms_m"] ** 2 for entry in visited) / len(visited)) if visited else None,
                       observed_bin_max_m=max(entry["max_m"] for entry in visited) if visited else None,
                       height_coordinate="clamped start-to-nail vertical fraction; excludes follow-through",
                       approach_height_extent=[float(self.approach_height_min[i]), float(self.approach_height_max[i])] if pre["sample_count"] else None)
            loads = {"cap_nms": self.caps.tolist(), "cadence_s": self.dt,
                     "signal": LOAD_SIGNAL_SEMANTICS,
                     "common_window_semantics": (
                         "peak of max(current rolling Lambda, current control-step peak) during the onset window; "
                         "includes pre-onset ring history and any retained pre-onset control-step peak"),
                     "exceedance_duration_semantics": (
                         "physics-sample duration above cap for the authoritative peak-held signal, "
                         "not instantaneous rolling-Lambda exceedance duration")}
            for index, name in enumerate(("episode", "common_window")):
                loads[name] = dict(peak_nms=self.load_peaks[i, index].tolist(),
                                   utilization=(self.load_peaks[i, index] / self.caps).tolist(),
                                   exceeded=(self.load_peaks[i, index] > self.caps).tolist(),
                                   exceedance_samples=self.load_exceedances[i, index].tolist(),
                                   exceedance_duration_s=(self.load_exceedances[i, index] * self.dt).tolist(),
                                   observed_samples=int(self.samples[i]) if index == 0 else count)
            records.append(dict(
                miss=not bool(self.onset[i]), observed_samples=int(self.samples[i]),
                observed_duration_s=int(self.samples[i]) * self.dt,
                time_to_contact_s=(int(self.onset[i]) * self.dt if self.onset[i] else None),
                native_success_sample=int(self.success_sample[i]) or None,
                native_success_time_s=int(self.success_sample[i]) * self.dt if self.success_sample[i] else None,
                precontact_path=pre,
                postcontact_path=path(self.path_count[i, 1], self.path_squares[i, 1], self.path_max[i, 1]),
                loads=loads,
                common_window=dict(window_samples=count, window_complete=complete,
                    observed_duration_s=count * self.dt, required_samples=self.window,
                    censored=not complete,
                    censor_reason=None if complete else (stop_reasons[i] if stop_reasons else "evaluation_budget"),
                    axial_positive_ns=float(self.integrals[i, 0]), axial_signed_ns=float(self.integrals[i, 1]),
                    transverse_magnitude_ns=float(self.integrals[i, 2]))))
        return records


class StrikeEvaluationMetric(ManagerTermBase):
    """Adapter ordered last among per-substep metrics; reads, never forwards."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self.collector = StrikeEvaluationCollector(env.num_envs, env.device, env.physics_dt,
            env.cfg.metrics["cat_soft"].params["imp_limit"])
        env._strike_evaluation = self.collector

    def reset(self, env_ids):
        self.collector.reset(env_ids)

    def __call__(self, env, **params):
        tracker = env._hammer_first_strike
        head = tracker._robot.data.site_pos_w[:, tracker._site_ids].squeeze(1)
        phase, _, error = env._strike_reference.diagnose_current_height(head)
        approach_height = env._strike_reference.approach_height_coordinate(head)
        self.collector.sample(started=tracker.started,
            contact=(tracker._contact_sensor.data.found > 0).any(-1),
            force_world=tracker._impulse_sensor.data.force.sum(-2), axis=tracker._axis,
            load=env._hammer_substep_impulse.impulse, approach_height=approach_height, error_m=error)
        return torch.zeros_like(phase)


def configure_strike_evaluation(cfg):
    """Evaluation configs only; append after the two authoritative sources."""
    for name in ("first_strike", "substep_impulse"):
        if name not in cfg.metrics or not cfg.metrics[name].per_substep:
            raise ValueError(f"evaluation requires per-substep {name}")
    cfg.metrics["strike_evaluation"] = MetricsTermCfg(func=StrikeEvaluationMetric, per_substep=True, reduce="last")


class MeasurementProtocol:
    """Wrap only success after checkpoint validation, restoring the exact term."""

    def __init__(self, env, protocol):
        if protocol not in ("native", "common-window"):
            raise ValueError("unknown measurement protocol")
        self.env, self.protocol = env, protocol

    def __enter__(self):
        manager = self.env.termination_manager
        index = manager._term_names.index("nail_driven")
        self.cfg = manager._term_cfgs[index]
        self.original = self.cfg.func
        def success(env, **params):
            native = self.original(env, **params)
            continued = env._strike_evaluation.success_termination(native)
            return continued if self.protocol == "common-window" else native
        self.cfg.func = success
        return self

    def __exit__(self, *exc):
        self.cfg.func = self.original


def evaluation_snapshot(env):
    """Called before autoreset, after termination/reward/metrics computations."""
    manager = env.termination_manager
    reasons = ["+".join(name for name, values in manager._term_dones.items() if bool(values[i]))
               or "evaluation_budget" for i in range(env.num_envs)]
    records = env._strike_evaluation.records(reasons)
    tracker = env._hammer_first_strike
    for i, record in enumerate(records):
        record["native"] = dict(delivered_impulse_ns=float(tracker.delivered[i]),
            transverse_impulse_ns=float(tracker.delivered_transverse[i]),
            finalized=bool(tracker.finalized[i]), stop_reason=int(tracker.reason[i]))
        record["termination_reasons"] = reasons[i]
    return records


def summarize_strike_evaluations(records):
    """All observed attempts are denominators, including censored misses/failures."""
    summary = dict(attempts=len(records), misses=sum(r["miss"] for r in records),
                   complete_windows=sum(r["common_window"]["window_complete"] for r in records),
                   censored_windows=sum(r["common_window"]["censored"] for r in records))
    for interval in ("episode", "common_window"):
        summary[interval + "_joint_exceedance_attempts"] = [
            sum(r["loads"][interval]["exceeded"][j] for r in records) for j in range(6)]
    return summary

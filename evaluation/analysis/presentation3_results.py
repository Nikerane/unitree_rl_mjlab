"""Build the compact, provenance-bound Presentation3 result package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from evaluation.analysis.first_strike_campaign import (
    EXPECTED_EPISODES_PER_ENV,
    EXPECTED_EPISODES_PER_SEED,
    EXPECTED_IMPULSE_LIMITS_N_M_S,
    EXPECTED_NUM_ENVS,
    aggregate_episode_metrics,
    summarize_episode,
)


VIDEO_ARM_BY_TREATMENT = {
    "M": "P+D4",
    "V+D0": "P+V+D0",
    "V": "P+V",
    "V+M": "P+V+D4",
}
DELIVERED_DOSE_BY_TREATMENT = {"M": 4.0, "V+D0": 0.0, "V": 2.0, "V+M": 4.0}
EXPECTED_ASSET_REVISION = "b58ccd2f81fd246f27c1e8d88cf86484cd888703"
EXPECTED_RESET_DIGEST = "bde511ec2adc42e5365e1e46f45ff1fb43223a93c31c6fbfb4580352e445e319"
EXPECTED_EXISTING_MANIFEST = "dab2f08e66dae9390e59764189a08f356fbe4718e36927644a29ee897a223c89"
EXPECTED_D0_MANIFEST = "f55f20c80774916950f56252efa5fddebecf475de2408b221eb44273a5b7efd7"
EXPECTED_SAMPLED_INPUT_INVENTORY_SHA256 = (
    "2c1e1cf3711ffc8ef57c8aa078688b712f5719994b5ac5fb5549e4a369ca9ec2"
)
EXPECTED_EXISTING_TRAINING = "ba6119c767fe92a8eb4b6131e0c0b0d3c120f0fe"
EXPECTED_D0_TRAINING = "db48032449cb77184579f8cf62b9a5a228cfa77f"
EXPECTED_EXISTING_EVALUATION = "2b045371704604367b7c16b87e90457d20db30f7"
EXPECTED_D0_EVALUATION = EXPECTED_D0_TRAINING
EXPECTED_EXISTING_RENDERER = "268502be41ae2caf20d5a147edfa90a0f1cda766"
EXPECTED_D0_RENDERER = "3435dc8357ff63b6dc24f8dee153b9b950c5b71e"
EXPECTED_IMPULSE_ANALYSIS_BY_POLICY = {
    **{("M", seed): EXPECTED_EXISTING_RENDERER for seed in (2, 3, 4)},
    **{("M", seed): EXPECTED_EXISTING_TRAINING for seed in (5, 6, 7)},
    **{("V+D0", seed): EXPECTED_D0_RENDERER for seed in range(2, 8)},
    **{("V", seed): EXPECTED_EXISTING_RENDERER for seed in range(2, 8)},
    **{("V+M", seed): EXPECTED_EXISTING_TRAINING for seed in range(2, 8)},
}
EXPECTED_CONFIG_IDENTITY_BY_TREATMENT = {
    "M": {
        "campaign_config_sha256": (
            "1b100978f6658ea45b5c06a56a1a50a1efd39691b038bafcc2980764d57379a3"
        ),
        "treatment_config_sha256": (
            "5d78b234089983296f79e444f48182f8bc89cb253c53cc61087e1f421d00c1ac"
        ),
        "nail_asset_sha256": (
            "47b0986d74675673a8c7e03a7b6d7012790d7cb10c1622bf699179f0046c3e00"
        ),
    },
    "V+D0": {
        "campaign_config_sha256": (
            "370ef93158ce3f77fc27a6b796a49453ef2eee2779b11791b06f654ecd252a6d"
        ),
        "treatment_config_sha256": (
            "a7bcaac571ad6bb11239a0be30a6e3d374457c35b2e7dc1d36d095cbc59379e3"
        ),
        "nail_asset_sha256": (
            "47b0986d74675673a8c7e03a7b6d7012790d7cb10c1622bf699179f0046c3e00"
        ),
    },
    "V": {
        "campaign_config_sha256": (
            "5cb484bb432f84d767bb41737b094f8eccd65a9ee133d5d43a4c32bd8975f31f"
        ),
        "treatment_config_sha256": (
            "14876e175f6ea4cca5ee65fc1bcb5acf9822a99767b12290f3c58a0d928b8ad5"
        ),
        "nail_asset_sha256": (
            "47b0986d74675673a8c7e03a7b6d7012790d7cb10c1622bf699179f0046c3e00"
        ),
    },
    "V+M": {
        "campaign_config_sha256": (
            "1b100978f6658ea45b5c06a56a1a50a1efd39691b038bafcc2980764d57379a3"
        ),
        "treatment_config_sha256": (
            "c59ffdb85516909f05aa37a20742e00917b46f2fffa6273fd1108815ebffee62"
        ),
        "nail_asset_sha256": (
            "47b0986d74675673a8c7e03a7b6d7012790d7cb10c1622bf699179f0046c3e00"
        ),
    },
}
FROZEN_MANIFEST_FIELDS = (
    "array_index",
    "arm",
    "name",
    "task",
    "training_seed",
    "training_code_revision",
    "training_asset_revision",
    "checkpoint_path",
    "checkpoint_sha256",
)
EXPECTED_SELECTION = "first two completed episodes from each of 256 environments"
EXPECTED_RNG_STREAMS = {
    "reset": 2036072919,
    "observation": 2046072933,
    "action": 2056072941,
}
EXPECTED_EVALUATION_CONTRACT = {
    "base_rng_seed": 2026072900,
    "num_envs": EXPECTED_NUM_ENVS,
    "episodes_per_env": EXPECTED_EPISODES_PER_ENV,
    "episode_len_s": 4.0,
    "mean_nsteps": 400,
    "completion_rule": "first_two_completions_per_environment",
    "stochastic_actions": True,
    "reset_position_noise_rad": [0.0, 0.0],
    "actor_observation_corruption": True,
    "critic_observation_corruption": False,
    "physics_dt_s": 0.002,
    "control_decimation": 10,
    "fixed_impedance_signature_sha256": (
        "a8252c853dd0059c89cff357e8e542fc6d097ecc9ef2652768ca0ef83d8aa269"
    ),
    "fixed_action_signature_sha256": (
        "56e59da46050a16c2005cb1872632ed81d41f82b48ae5c94beaca8fea44790ec"
    ),
    "strict_config_identities": {},
}


def load_frozen_training_manifest(path, *, expected_sha256, expected_keys):
    """Read one hash-pinned training manifest and require its exact arm/seed matrix."""
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"frozen manifest SHA-256 mismatch: {manifest_path}")
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
        if tuple(reader.fieldnames or ()) != FROZEN_MANIFEST_FIELDS:
            raise ValueError("frozen manifest header drift")
        rows = list(reader)
    except UnicodeDecodeError as error:
        raise ValueError("frozen manifest is not UTF-8") from error

    expected = set(expected_keys)
    if len(rows) != len(expected):
        raise ValueError("frozen manifest row-count drift")
    keyed = {}
    array_indices = set()
    for index, row in enumerate(rows):
        if None in row or any(row.get(field, "") == "" for field in FROZEN_MANIFEST_FIELDS):
            raise ValueError(f"frozen manifest row {index} is malformed")
        try:
            key = (str(row["arm"]), int(row["training_seed"]))
            array_index = int(row["array_index"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"frozen manifest row {index} identity is malformed") from error
        if key in keyed:
            raise ValueError(f"frozen manifest duplicate policy identity: {key}")
        if len(row["training_code_revision"]) != 40 or len(
            row["training_asset_revision"]
        ) != 40:
            raise ValueError(f"frozen manifest row {index} revision is malformed")
        if len(row["checkpoint_sha256"]) != 64:
            raise ValueError(f"frozen manifest row {index} checkpoint is malformed")
        keyed[key] = row
        array_indices.add(array_index)
    if set(keyed) != expected:
        raise ValueError("frozen manifest policy matrix drift")
    if array_indices != set(range(len(rows))):
        raise ValueError("frozen manifest array-index drift")
    return keyed


def validate_sampled_input_inventory(records, *, expected_sha256):
    """Bind the exact accepted summary/trace bytes, including the r4 evaluation."""
    arm_order = {arm: index for index, arm in enumerate(VIDEO_ARM_BY_TREATMENT)}
    expected_keys = {
        (arm, seed) for arm in VIDEO_ARM_BY_TREATMENT for seed in range(2, 8)
    }
    keyed = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"sampled input inventory row {index} is malformed")
        try:
            key = (str(record["arm"]), int(record["training_seed"]))
            summary_sha = str(record["summary_sha256"])
            artifact_sha = str(record["artifact_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"sampled input inventory row {index} is malformed"
            ) from error
        if key in keyed:
            raise ValueError(f"sampled input inventory duplicate policy: {key}")
        try:
            valid_hashes = (
                len(summary_sha) == 64
                and len(artifact_sha) == 64
                and int(summary_sha, 16) >= 0
                and int(artifact_sha, 16) >= 0
            )
        except ValueError:
            valid_hashes = False
        if not valid_hashes:
            raise ValueError(f"sampled input inventory row {index} hash is malformed")
        keyed[key] = (summary_sha, artifact_sha)
    if set(keyed) != expected_keys:
        raise ValueError("sampled input inventory policy matrix drift")
    canonical = "".join(
        f"{arm}\t{seed}\t{keyed[(arm, seed)][0]}\t{keyed[(arm, seed)][1]}\n"
        for arm, seed in sorted(
            keyed, key=lambda key: (arm_order[key[0]], key[1])
        )
    ).encode("utf-8")
    actual_sha256 = hashlib.sha256(canonical).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("sampled input inventory SHA-256 mismatch")
    return actual_sha256


def _strict_json_equal(actual, expected):
    """Compare decoded JSON without Python's bool/int or int/float coercions."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_sampled_population_contract(
    summary,
    payload,
    *,
    manifest,
    training_revision,
    evaluation_revision,
    asset_revision,
    manifest_row,
    config_identity,
):
    """Bind one raw population to the frozen Presentation3 sampling contract."""
    if not isinstance(summary, Mapping) or not isinstance(payload, Mapping):
        raise ValueError("sampled contract inputs must be mappings")
    treatment = str(summary.get("treatment", ""))
    if treatment not in VIDEO_ARM_BY_TREATMENT:
        raise ValueError("unknown Presentation3 treatment")
    try:
        seed = int(summary["training_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("sampled training seed is malformed") from error
    task = str(summary.get("task", ""))
    checkpoint = str(summary.get("checkpoint_sha256", ""))
    if not isinstance(manifest_row, Mapping):
        raise ValueError("frozen manifest row is missing")
    expected_manifest_identity = {
        "arm": treatment,
        "name": summary.get("name"),
        "task": task,
        "training_seed": str(seed),
        "training_code_revision": training_revision,
        "training_asset_revision": asset_revision,
        "checkpoint_sha256": checkpoint,
    }
    for field, expected in expected_manifest_identity.items():
        if manifest_row.get(field) != expected:
            raise ValueError(f"frozen manifest {field} drift")
    if not isinstance(config_identity, Mapping):
        raise ValueError("frozen config identity is missing")
    for field in (
        "campaign_config_sha256",
        "treatment_config_sha256",
        "nail_asset_sha256",
    ):
        if summary.get(field) != config_identity.get(field):
            raise ValueError(f"sampled config identity {field} drift")
    expected_dose = DELIVERED_DOSE_BY_TREATMENT[treatment]
    expected_summary = {
        "accepted_checkpoint_sha256": checkpoint,
        "accepted_manifest_sha256": manifest,
        "training_code_revision": training_revision,
        "training_asset_revision": asset_revision,
        "git_revision": evaluation_revision,
        "asset_git_revision": asset_revision,
        "n_episodes_sampled": str(EXPECTED_EPISODES_PER_SEED),
        "seed": "2026072900",
        "num_envs": str(EXPECTED_NUM_ENVS),
        "episodes_per_env_sampled": str(EXPECTED_EPISODES_PER_ENV),
        "episode_len_s": "4.0",
        "nsteps": "400",
        "sampled_completion_rule": "first_two_completions_per_environment",
        "sampled_actions_stochastic": "True",
        "reset_position_noise_min_rad": "0.0",
        "reset_position_noise_max_rad": "0.0",
        "actor_observation_corruption": "True",
        "critic_observation_corruption": "False",
        "physics_dt_s": "0.002",
        "control_decimation": "10",
        "fixed_impedance_signature_sha256": EXPECTED_EVALUATION_CONTRACT[
            "fixed_impedance_signature_sha256"
        ],
        "fixed_action_signature_sha256": EXPECTED_EVALUATION_CONTRACT[
            "fixed_action_signature_sha256"
        ],
        "reset_rng_seed": str(EXPECTED_RNG_STREAMS["reset"]),
        "observation_rng_seed": str(EXPECTED_RNG_STREAMS["observation"]),
        "action_rng_seed": str(EXPECTED_RNG_STREAMS["action"]),
        "impact_weight": "8.0",
        "delivered_weight": str(float(expected_dose)),
        "r_waypoint_progress_weight": "8.0",
        "event_i_ref_n_s": "0.3088",
        "imp_max_p": "0.0",
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"sampled summary {field} drift")
    if str(summary.get("git_dirty", "")).lower() != "false" or str(
        summary.get("asset_git_dirty", "")
    ).lower() != "false":
        raise ValueError("sampled summary dirty provenance")

    scalar_bindings = {
        "schema_version": 3,
        "selection": EXPECTED_SELECTION,
        "expected_episode_count": EXPECTED_EPISODES_PER_SEED,
        "task": task,
        "treatment": treatment,
        "training_seed": seed,
        "event_i_ref_n_s": float(summary["event_i_ref_n_s"]),
        "imp_max_p": 0.0,
    }
    for field, expected in scalar_bindings.items():
        if not _strict_json_equal(payload.get(field), expected):
            label = "selection" if field == "selection" else field
            raise ValueError(f"sampled payload {label} drift")
    if not _strict_json_equal(payload.get("rng_streams"), EXPECTED_RNG_STREAMS):
        raise ValueError("sampled payload RNG streams drift")
    if not _strict_json_equal(
        payload.get("evaluation_contract"), EXPECTED_EVALUATION_CONTRACT
    ):
        raise ValueError("sampled payload evaluation contract drift")

    if not _strict_json_equal(payload.get("weights"), {
        "impact_progress": 8.0,
        "delivered_impulse": expected_dose,
        "r_waypoint_progress": 8.0,
    }):
        raise ValueError("sampled payload treatment weights drift")
    try:
        caps = np.asarray(payload.get("impulse_limits_n_m_s"), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("sampled payload impulse caps drift") from error
    expected_caps = np.asarray(EXPECTED_IMPULSE_LIMITS_N_M_S, dtype=float)
    if caps.shape != expected_caps.shape or not np.array_equal(caps, expected_caps):
        raise ValueError("sampled payload impulse caps drift")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("sampled payload provenance is missing")
    expected_provenance = {
        "checkpoint_sha256": checkpoint,
        "accepted_checkpoint_sha256": checkpoint,
        "accepted_manifest_sha256": manifest,
        "campaign_config_sha256": config_identity.get("campaign_config_sha256"),
        "treatment_config_sha256": config_identity.get("treatment_config_sha256"),
        "nail_asset_sha256": config_identity.get("nail_asset_sha256"),
        "training_code_revision": training_revision,
        "training_asset_revision": asset_revision,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(f"sampled payload provenance {field} drift")
    for field, expected_revision in (
        ("code_git", evaluation_revision),
        ("asset_git", asset_revision),
    ):
        binding = provenance.get(field)
        if (
            not isinstance(binding, Mapping)
            or binding.get("revision") != expected_revision
            or binding.get("dirty") is not False
        ):
            raise ValueError(f"sampled payload provenance {field} drift")

    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != EXPECTED_EPISODES_PER_SEED:
        raise ValueError("sampled payload episode count drift")
    coordinates = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"sampled episode {index} is malformed")
        try:
            env_id = episode["env_id"]
            ordinal = episode["episode_ordinal"]
            if type(env_id) is not int or type(ordinal) is not int:
                raise TypeError
            coordinate = (env_id, ordinal)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"sampled episode {index} environment/ordinal drift") from error
        if (
            episode.get("arm") != treatment
            or episode.get("task") != task
            or episode.get("episode_id")
            != f"{treatment}-env{coordinate[0]}-episode{coordinate[1]}"
        ):
            raise ValueError(f"sampled episode {index} episode identity drift")
        episode_caps = np.asarray(episode.get("impulse_limits_n_m_s"), dtype=float)
        if (
            episode_caps.shape != expected_caps.shape
            or not np.array_equal(episode_caps, expected_caps)
        ):
            raise ValueError(f"sampled episode {index} episode impulse caps drift")
        coordinates.add(coordinate)
    expected_coordinates = {
        (env_id, ordinal)
        for env_id in range(EXPECTED_NUM_ENVS)
        for ordinal in range(EXPECTED_EPISODES_PER_ENV)
    }
    if coordinates != expected_coordinates or len(coordinates) != len(episodes):
        raise ValueError("sampled environment/ordinal population drift")


def write_result_package(rows, traces, output_dir):
    """Write the one compact table and four presentation figures."""
    ordered = canonical_policy_rows(rows)
    output = Path(output_dir)
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "presentation3_24_policy_comparison.csv"
    fieldnames = list(ordered[0])
    if any(list(row) != fieldnames for row in ordered):
        raise ValueError("result rows must share one ordered schema")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)

    figures = {
        "dose_response": (
            dose_response_figure(ordered),
            figure_dir / "presentation3_dose_first_event.png",
        ),
        "velocity_cat": (
            velocity_cat_figure(ordered),
            figure_dir / "presentation3_velocity_cat.png",
        ),
        "cap_utilization": (
            cap_utilization_figure(ordered),
            figure_dir / "presentation3_impulse_cap_nonbinding.png",
        ),
        "trajectory_grid": (
            trajectory_grid_figure(ordered, traces),
            figure_dir / "presentation3_fixed_reset_4x6_xz.png",
        ),
    }
    paths = {"table": table_path}
    for name, (figure, path) in figures.items():
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths[name] = path
    return paths


def assemble_result_row(
    sampled_summary,
    sampled_payload,
    video_metadata,
    video_trace,
    impulse_metadata,
    impulse_trace,
):
    """Join one sampled population with one identical-reset CPU trajectory."""
    mappings = (
        sampled_summary,
        sampled_payload,
        video_metadata,
        video_trace,
        impulse_metadata,
        impulse_trace,
    )
    if not all(isinstance(value, Mapping) for value in mappings):
        raise ValueError("result inputs must be mappings")
    treatment = str(sampled_summary.get("treatment", ""))
    if treatment not in VIDEO_ARM_BY_TREATMENT:
        raise ValueError("unknown Presentation3 treatment")
    try:
        seed = int(sampled_summary["training_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("sampled training seed is malformed") from error
    task = str(sampled_summary.get("task", ""))
    checkpoint = str(sampled_summary.get("checkpoint_sha256", ""))
    sampled_identity = {
        "checkpoint_sha256": checkpoint,
        "reset_state_digest": sampled_summary.get("reset_digest"),
        "training_revision": sampled_summary.get("training_code_revision"),
        "asset_revision": sampled_summary.get("training_asset_revision"),
        "training_seed": seed,
    }
    fixed_identity = {
        "checkpoint_sha256": video_metadata.get("checkpoint_sha256"),
        "reset_state_digest": video_metadata.get("reset_state_digest"),
        "training_revision": video_metadata.get("training_revision"),
        "asset_revision": video_metadata.get("asset_revision"),
        "training_seed": video_metadata.get("training_seed"),
    }
    validate_artifact_join(sampled_identity, fixed_identity)
    expected_video_arm = VIDEO_ARM_BY_TREATMENT[treatment]
    identity_checks = {
        "sampled payload task": sampled_payload.get("task") == task,
        "sampled payload treatment": sampled_payload.get("treatment") == treatment,
        "sampled payload seed": sampled_payload.get("training_seed") == seed,
        "sampled payload checkpoint": sampled_payload.get("provenance", {}).get(
            "checkpoint_sha256"
        ) == checkpoint,
        "video campaign": video_metadata.get("campaign") == "presentation3",
        "video arm": video_metadata.get("arm") == expected_video_arm,
        "video task": video_metadata.get("task") == task,
        "impulse campaign": impulse_metadata.get("campaign") == "presentation3",
        "impulse arm": impulse_metadata.get("arm") == expected_video_arm,
        "impulse task": impulse_metadata.get("task") == task,
        "impulse seed": impulse_metadata.get("training_seed") == seed,
        "impulse checkpoint": impulse_metadata.get("checkpoint_sha256") == checkpoint,
        "impulse asset": impulse_metadata.get("asset_revision")
        == sampled_identity["asset_revision"],
        "impulse reset": impulse_metadata.get("reset_state_digest")
        == sampled_identity["reset_state_digest"],
    }
    failed = [name for name, passed in identity_checks.items() if not passed]
    if failed:
        raise ValueError(f"artifact identity mismatch: {', '.join(failed)}")
    expected_impulse_revision = EXPECTED_IMPULSE_ANALYSIS_BY_POLICY.get(
        (treatment, seed)
    )
    if impulse_metadata.get("code_revision") != expected_impulse_revision:
        raise ValueError("impulse analysis revision drift")
    try:
        fixed_caps = np.asarray(impulse_metadata["j_limit_n_m_s"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fixed-reset impulse caps are malformed") from error
    expected_caps = np.asarray(EXPECTED_IMPULSE_LIMITS_N_M_S, dtype=float)
    if fixed_caps.shape != expected_caps.shape or not np.array_equal(
        fixed_caps, expected_caps
    ):
        raise ValueError("fixed-reset impulse caps drift")
    if sampled_summary.get("accepted_checkpoint_sha256") != checkpoint:
        raise ValueError("accepted checkpoint mismatch")
    if str(sampled_summary.get("git_dirty")).lower() != "false" or str(
        sampled_summary.get("asset_git_dirty")
    ).lower() != "false":
        raise ValueError("dirty sampled evaluation provenance")
    weights = sampled_payload.get("weights", {})
    expected_dose = DELIVERED_DOSE_BY_TREATMENT[treatment]
    if (
        float(weights.get("impact_progress", float("nan"))) != 8.0
        or float(weights.get("r_waypoint_progress", float("nan"))) != 8.0
        or float(weights.get("delivered_impulse", float("nan"))) != expected_dose
        or float(sampled_payload.get("imp_max_p", float("nan"))) != 0.0
    ):
        raise ValueError("sampled treatment weights drift")

    try:
        sampled_n = int(sampled_summary["n_episodes_sampled"])
        raw_episodes = sampled_payload["episodes"]
        nail_geometry = sampled_payload["nail_geometry"]
        if sampled_n <= 0 or not isinstance(raw_episodes, list) or not raw_episodes:
            raise ValueError("sampled population must be non-empty")
        raw_episode_metrics = [
            summarize_episode(
                episode,
                nail_geometry=nail_geometry,
                qvel_limit_rad_s=3.1415,
            )
            for episode in raw_episodes
        ]
        raw_aggregate = aggregate_episode_metrics(
            raw_episode_metrics, expected_episode_count=sampled_n
        )
        sampled_caps = np.asarray(
            sampled_payload["impulse_limits_n_m_s"], dtype=np.float32
        )
        if (
            sampled_caps.shape != (6,)
            or not np.isfinite(sampled_caps).all()
            or np.any(sampled_caps <= 0.0)
        ):
            raise ValueError("sampled impulse caps must be six finite positive values")
        sampled_peak_ratios = []
        for index, episode in enumerate(raw_episodes):
            peak_lambda = np.asarray(
                episode["episode_peak_lambda"], dtype=np.float32
            )
            if (
                peak_lambda.shape != sampled_caps.shape
                or not np.isfinite(peak_lambda).all()
                or np.any(peak_lambda < 0.0)
            ):
                raise ValueError(
                    f"sampled episode {index} has malformed peak Lambda"
                )
            sampled_peak_ratios.append(
                float(np.max(peak_lambda / sampled_caps))
            )
        sampled_max_lambda_cap_ratio = max(sampled_peak_ratios)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"sampled raw metric recomputation failed: {error}") from error

    recomputed_summary = {
        "qvel_violation_rate_sampled": raw_aggregate[
            "qvel_violation_rate_sampled"
        ],
        "qvel_max_abs_rad_s_sampled": raw_aggregate[
            "qvel_max_abs_rad_s_sampled"
        ],
        "worst_ratio_max_sampled": sampled_max_lambda_cap_ratio,
        "recontact_rate_sampled": raw_aggregate["recontact_rate_sampled"],
        "tail_fraction_mean_sampled": raw_aggregate[
            "tail_fraction_mean_sampled"
        ],
        "first_strike_v_precontact_mean_sampled": raw_aggregate[
            "first_strike_v_precontact_mean_sampled"
        ],
    }
    for field, recomputed in recomputed_summary.items():
        try:
            recorded = float(sampled_summary[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"sampled summary {field} is malformed") from error
        if not np.isfinite(recorded) or not np.isclose(
            recorded, float(recomputed), rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"sampled summary {field} mismatch")

    video_head = np.asarray(video_trace["substep_head_position_m"])
    diagnostic_head = np.asarray(impulse_trace["head_position_m"])
    video_contact = np.asarray(video_trace["substep_contact"], dtype=bool)
    diagnostic_contact = np.asarray(impulse_trace["contact"], dtype=bool)
    video_qvel = np.asarray(video_trace["substep_arm_qvel_rad_s"])
    diagnostic_qvel = np.asarray(
        impulse_trace["joint_velocity_post_integration_rad_s"]
    )
    video_depth = np.minimum(
        np.asarray(video_trace["substep_nail_depth_m"], dtype=float), 0.032
    )
    diagnostic_depth = np.asarray(impulse_trace["nail_depth_post_integration_m"])
    shared = {
        "head position": (video_head, diagnostic_head),
        "contact": (video_contact, diagnostic_contact),
        "post-integration qvel": (video_qvel, diagnostic_qvel),
        "clamped nail depth": (video_depth, diagnostic_depth),
    }
    for label, (left, right) in shared.items():
        if left.shape != right.shape or not np.array_equal(
            left.astype(np.float32, copy=False), right.astype(np.float32, copy=False)
        ):
            raise ValueError(f"renderer/diagnostic {label} mismatch")

    sampled_event = sampled_first_event_metrics(sampled_payload)
    sampled_waypoints = sampled_waypoints_by_contact(sampled_payload)
    fixed_impulse = fixed_impulse_metrics(
        impulse_trace, caps=fixed_caps
    )
    fixed_peak, fixed_legal = peak_post_integration_qvel(
        diagnostic_qvel, limit=3.1415
    )
    fixed_waypoints = waypoints_reached_by_contact(
        video_trace["substep_gate_index"], video_contact
    )
    fixed_geometry = precontact_geometry_metrics(
        video_head,
        video_trace["substep_perpendicular_error_m"],
        video_contact,
        entry=video_trace["guideline_entry_m"],
    )
    if sampled_event["episode_n"] != sampled_n:
        raise ValueError("sampled episode count mismatch")
    sampled_qvel_legal_n = sum(
        not bool(episode["qvel_violation"]) for episode in raw_episode_metrics
    )

    row = {
        "arm": treatment,
        "training_seed": seed,
        "task": task,
        "checkpoint_sha256": checkpoint,
        "training_revision": sampled_identity["training_revision"],
        "asset_revision": sampled_identity["asset_revision"],
        "sampled_evaluation_revision": sampled_summary.get("git_revision"),
        "renderer_analysis_revision": video_metadata.get("analysis_revision"),
        "impulse_analysis_revision": impulse_metadata.get("code_revision"),
        "reset_state_digest": sampled_identity["reset_state_digest"],
        "sampled_manifest_sha256": sampled_summary.get("accepted_manifest_sha256"),
        "sampled_payload_digest": sampled_summary.get("sampled_trace_digest"),
        "sampled_artifact_sha256": sampled_summary.get(
            "sampled_trace_artifact_sha256"
        ),
        "delivered_reward_dose": expected_dose,
        "velocity_cat_enabled": treatment != "M",
        "impulse_cat_max_p": 0.0,
        "waypoint_progress_weight": 8.0,
        "sampled_episode_n": sampled_n,
        "sampled_overall_success_n": sampled_event["overall_success_n"],
        "sampled_productive_n": sampled_event["productive_n"],
        "sampled_success_finalized_n": sampled_event["success_finalized_n"],
        "sampled_window_finalized_n": sampled_event["window_finalized_n"],
        "sampled_first_event_impulse_mean_n_s": sampled_event[
            "first_event_impulse_mean_n_s"
        ],
        "sampled_cumulative_impulse_mean_n_s": sampled_event[
            "cumulative_impulse_mean_n_s"
        ],
        "sampled_post_event_tail_mean_n_s": sampled_event[
            "post_event_tail_mean_n_s"
        ],
        "sampled_post_finalization_contact_rate": float(
            raw_aggregate["recontact_rate_sampled"]
        ),
        "sampled_tail_fraction_mean": float(
            raw_aggregate["tail_fraction_mean_sampled"]
        ),
        "sampled_precontact_axial_speed_mean_m_s": float(
            raw_aggregate["first_strike_v_precontact_mean_sampled"]
        ),
        "sampled_all_six_by_contact_n": sampled_waypoints[
            "all_six_by_contact_n"
        ],
        "sampled_all_six_by_contact_rate": sampled_waypoints[
            "all_six_by_contact_rate"
        ],
        "sampled_qvel_legal_n": sampled_qvel_legal_n,
        "sampled_qvel_max_rad_s": float(
            raw_aggregate["qvel_max_abs_rad_s_sampled"]
        ),
        "sampled_max_lambda_cap_ratio": sampled_max_lambda_cap_ratio,
        "fixed_first_event_impulse_n_s": fixed_impulse[
            "first_event_impulse_n_s"
        ],
        "fixed_cumulative_impulse_n_s": fixed_impulse[
            "cumulative_impulse_n_s"
        ],
        "fixed_post_event_tail_n_s": fixed_impulse["post_event_tail_n_s"],
        "fixed_precontact_axial_speed_m_s": fixed_impulse[
            "precontact_axial_speed_m_s"
        ],
        "fixed_max_lambda_cap_ratio": fixed_impulse["max_lambda_cap_ratio"],
        "fixed_qvel_max_rad_s": fixed_peak,
        "fixed_qvel_legal": fixed_legal,
        "fixed_all_six_by_contact": fixed_waypoints >= 6,
        "fixed_waypoints_by_contact": fixed_waypoints,
        "fixed_success": bool(float(np.max(diagnostic_depth)) >= 0.032 - 1e-7),
        "fixed_clamped_depth_m": float(np.max(diagnostic_depth)),
        "fixed_precontact_window_substeps": fixed_geometry["window_substeps"],
        "fixed_precontact_perpendicular_max_m": fixed_geometry[
            "perpendicular_max_m"
        ],
        "fixed_precontact_perpendicular_rms_m": fixed_geometry[
            "perpendicular_rms_m"
        ],
        "fixed_precontact_path_ratio": fixed_geometry["path_ratio"],
        "fixed_strict_straight": bool(
            fixed_waypoints >= 6
            and fixed_geometry["perpendicular_max_m"] <= 0.015
            and fixed_geometry["path_ratio"] <= 1.15
        ),
    }
    trajectory = {
        "head_position_m": video_head,
        "entry_m": np.asarray(video_trace["guideline_entry_m"], dtype=float),
        "nail_m": np.asarray(video_trace["guideline_nail_m"], dtype=float),
        "waypoints_m": np.asarray(
            video_trace["guideline_gate_centers_m"], dtype=float
        ),
    }
    return row, trajectory


def dose_response_figure(rows):
    """Plot matched delivered-reward doses without pooling sampled and fixed resets."""
    ordered = canonical_policy_rows(rows)
    dose_arms = ("V+D0", "V", "V+M")
    x = np.asarray([0.0, 2.0, 4.0])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=False)
    panels = (
        (
            "sampled_first_event_impulse_mean_n_s",
            "sampled_cumulative_impulse_mean_n_s",
            "Strict sampled CUDA (512 episodes/checkpoint)",
        ),
        (
            "fixed_first_event_impulse_n_s",
            "fixed_cumulative_impulse_n_s",
            "Identical fixed reset (one CPU rollout/checkpoint)",
        ),
    )
    keyed = {(row["arm"], int(row["training_seed"])): row for row in ordered}
    for axis, (first_field, cumulative_field, title) in zip(axes, panels, strict=True):
        first_matrix = np.asarray(
            [[float(keyed[(arm, seed)][first_field]) for arm in dose_arms]
             for seed in TRAINING_SEEDS]
        )
        cumulative_matrix = np.asarray(
            [[float(keyed[(arm, seed)][cumulative_field]) for arm in dose_arms]
             for seed in TRAINING_SEEDS]
        )
        for values in first_matrix:
            axis.plot(x, values, color="#3b82f6", alpha=0.28, linewidth=1.0)
        for values in cumulative_matrix:
            axis.plot(x, values, color="#9ca3af", alpha=0.20, linewidth=0.8)
        axis.plot(
            x,
            np.median(first_matrix, axis=0),
            "o-",
            color="#075985",
            linewidth=2.5,
            label="median first event",
        )
        axis.plot(
            x,
            np.median(cumulative_matrix, axis=0),
            "s--",
            color="#4b5563",
            linewidth=1.8,
            label="median cumulative",
        )
        axis.set_xticks(x, ("D0", "D2", "D4"))
        axis.set_xlabel("Delivered-reward dose (velocity CaT active)")
        axis.set_ylabel("Axial hammer–nail impulse (N·s)")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Delivered-reward dose: productive first event vs episode cumulative")
    fig.tight_layout()
    return fig


def velocity_cat_figure(rows):
    """Show the matched D4 velocity-CaT contrast and retained first-event impulse."""
    ordered = canonical_policy_rows(rows)
    keyed = {(row["arm"], int(row["training_seed"])): row for row in ordered}
    arms = ("M", "V+M")
    x = np.asarray([0.0, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    qvel = np.asarray(
        [[float(keyed[(arm, seed)]["sampled_qvel_max_rad_s"]) for arm in arms]
         for seed in TRAINING_SEEDS]
    )
    impulse = np.asarray(
        [[float(keyed[(arm, seed)]["sampled_first_event_impulse_mean_n_s"])
          for arm in arms] for seed in TRAINING_SEEDS]
    )
    for values in qvel:
        axes[0].plot(x, values, "o-", color="#7c3aed", alpha=0.45)
    axes[0].plot(x, np.median(qvel, axis=0), "o-", color="#4c1d95", linewidth=2.8)
    axes[0].axhline(3.1415, color="#dc2626", linestyle="--", label="3.1415 rad/s limit")
    axes[0].set_ylabel("Worst observed 500 Hz |qvel| (rad/s)")
    axes[0].set_title("Observed velocity legality at D4")
    axes[0].legend(frameon=False)
    for values in impulse:
        axes[1].plot(x, values, "o-", color="#0f766e", alpha=0.45)
    axes[1].plot(x, np.median(impulse, axis=0), "o-", color="#134e4a", linewidth=2.8)
    axes[1].set_ylabel("Productive first-event impulse (N·s)")
    axes[1].set_title("Matched impact performance at D4")
    for axis in axes:
        axis.set_xticks(x, ("M\nno velocity CaT", "V+M\n500 Hz velocity CaT"))
        axis.grid(alpha=0.22)
    fig.suptitle("Velocity CaT at the same delivered-reward dose")
    fig.tight_layout()
    return fig


def cap_utilization_figure(rows):
    """Show that unchanged project-defined impulse caps never became active."""
    ordered = canonical_policy_rows(rows)
    keyed = {(row["arm"], int(row["training_seed"])): row for row in ordered}
    x = np.arange(len(ARM_ORDER), dtype=float)
    values = np.asarray(
        [[float(keyed[(arm, seed)]["sampled_max_lambda_cap_ratio"])
          for arm in ARM_ORDER] for seed in TRAINING_SEEDS]
    )
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for seed_values in values:
        axis.plot(x, seed_values, "o-", color="#d97706", alpha=0.38)
    axis.plot(x, np.median(values, axis=0), "o-", color="#92400e", linewidth=2.8)
    axis.axhline(1.0, color="#dc2626", linestyle="--", label="project cap (binding at 1.0)")
    axis.set_xticks(x, ARM_ORDER)
    axis.set_ylabel("Maximum per-joint Λ / project-defined cap")
    axis.set_title("Impulse CaT was non-binding at unchanged caps")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def trajectory_grid_figure(rows, traces):
    """Build a canonical 4x6 XZ trajectory grid with waypoints but no gate disks."""
    ordered = canonical_policy_rows(rows)
    required = {(row["arm"], int(row["training_seed"])) for row in ordered}
    if set(traces) != required:
        raise ValueError("trajectory traces must match the exact 4x6 policy matrix")
    arrays = []
    for key in [(arm, seed) for arm in ARM_ORDER for seed in TRAINING_SEEDS]:
        trace = traces[key]
        position = np.asarray(trace["head_position_m"], dtype=float)
        entry = np.asarray(trace["entry_m"], dtype=float)
        nail = np.asarray(trace["nail_m"], dtype=float)
        waypoints = np.asarray(trace["waypoints_m"], dtype=float)
        if (
            position.ndim != 2
            or position.shape[1] != 3
            or position.shape[0] == 0
            or entry.shape != (3,)
            or nail.shape != (3,)
            or waypoints.shape != (6, 3)
            or not all(np.isfinite(value).all() for value in (position, entry, nail, waypoints))
        ):
            raise ValueError(f"malformed trajectory trace for {key}")
        arrays.append((key, position, entry, nail, waypoints))
    all_x = np.concatenate(
        [np.concatenate((position[:, 0], entry[:1], nail[:1], waypoints[:, 0]))
         for _, position, entry, nail, waypoints in arrays]
    )
    all_z = np.concatenate(
        [np.concatenate((position[:, 2], entry[2:], nail[2:], waypoints[:, 2]))
         for _, position, entry, nail, waypoints in arrays]
    )
    margin = 0.005
    x_limits = (float(np.min(all_x) - margin), float(np.max(all_x) + margin))
    z_limits = (float(np.min(all_z) - margin), float(np.max(all_z) + margin))
    fig, axes = plt.subplots(4, 6, figsize=(18, 11), sharex=True, sharey=True)
    color = {"M": "#be123c", "V+D0": "#2563eb", "V": "#059669", "V+M": "#7c3aed"}
    description = {
        "M": "D4, no velocity CaT",
        "V+D0": "D0 + 500 Hz velocity CaT",
        "V": "D2 + 500 Hz velocity CaT",
        "V+M": "D4 + 500 Hz velocity CaT",
    }
    for axis, (key, position, entry, nail, waypoints) in zip(axes.flat, arrays, strict=True):
        arm, seed = key
        axis.plot(position[:, 0], position[:, 2], color=color[arm], linewidth=2.0)
        axis.plot(
            [entry[0], nail[0]], [entry[2], nail[2]],
            color="black", linestyle="--", linewidth=1.0,
        )
        axis.scatter(
            waypoints[:, 0], waypoints[:, 2], marker="D", s=16,
            facecolors="white", edgecolors="black", linewidths=0.7,
        )
        axis.scatter([nail[0]], [nail[2]], marker="x", s=22, color="black")
        axis.set_title(f"{description[arm]} | seed {seed}", fontsize=9)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    for row_axes in axes:
        row_axes[0].set_ylabel("Z (m)")
    for axis in axes[-1]:
        axis.set_xlabel("X (m)")
    fig.suptitle(
        "Fixed-reset hammer-head trajectories: waypoint guideline (dashed) and ordered waypoints (diamonds)"
    )
    fig.tight_layout()
    return fig


ARM_ORDER = ("M", "V+D0", "V", "V+M")
TRAINING_SEEDS = tuple(range(2, 8))


def sampled_first_event_metrics(payload):
    """Summarize every sampled first event without outcome filtering."""
    if not isinstance(payload, Mapping):
        raise ValueError("sampled payload must be a mapping")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("sampled payload has no episodes")

    delivered = []
    cumulative = []
    overall_success_n = finalized_n = productive_n = 0
    success_finalized_n = window_finalized_n = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping) or not isinstance(
            episode.get("first_strike"), Mapping
        ):
            raise ValueError(f"sampled episode {index} is malformed")
        first = episode["first_strike"]
        value = first.get("delivered_n_s")
        accumulated = episode.get("episode_delivered_accumulator_n_s")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"sampled episode {index} has invalid first-event impulse")
        if (
            isinstance(accumulated, bool)
            or not isinstance(accumulated, (int, float))
            or not np.isfinite(accumulated)
            or accumulated < 0.0
            or accumulated + 1e-9 < float(value)
        ):
            raise ValueError(f"sampled episode {index} has invalid cumulative impulse")
        finalized = first.get("finalized")
        productive = first.get("productive")
        overall_success = episode.get("overall_success")
        if not all(type(flag) is bool for flag in (finalized, productive, overall_success)):
            raise ValueError(f"sampled episode {index} has invalid Boolean state")
        reason = first.get("reason")
        if reason not in {"success", "window", "none"}:
            raise ValueError(f"sampled episode {index} has invalid finalization reason")
        delivered.append(float(value))
        cumulative.append(float(accumulated))
        overall_success_n += int(overall_success)
        finalized_n += int(finalized)
        productive_n += int(productive)
        success_finalized_n += int(finalized and reason == "success")
        window_finalized_n += int(finalized and reason == "window")

    return {
        "episode_n": len(episodes),
        "overall_success_n": overall_success_n,
        "finalized_n": finalized_n,
        "productive_n": productive_n,
        "success_finalized_n": success_finalized_n,
        "window_finalized_n": window_finalized_n,
        "first_event_impulse_mean_n_s": float(np.mean(delivered)),
        "cumulative_impulse_mean_n_s": float(np.mean(cumulative)),
        "post_event_tail_mean_n_s": float(
            np.mean(np.asarray(cumulative) - np.asarray(delivered))
        ),
    }


def sampled_waypoints_by_contact(payload):
    """Count episodes that reached all six ordered waypoints by contact onset."""
    if not isinstance(payload, Mapping):
        raise ValueError("sampled payload must be a mapping")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("sampled payload has no episodes")
    completed = 0
    for index, episode in enumerate(episodes):
        try:
            onset_raw = episode["first_strike"]["accepted_onset_index"]
            progress = np.asarray(episode["guideline"]["next_gate"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"sampled episode {index} has malformed waypoint data") from error
        if progress.ndim != 1 or progress.size == 0 or not np.isfinite(progress).all():
            raise ValueError(f"sampled episode {index} has malformed waypoint data")
        if onset_raw is None:
            continue
        if isinstance(onset_raw, bool):
            raise ValueError(f"sampled episode {index} has invalid contact onset")
        try:
            onset = int(onset_raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"sampled episode {index} has invalid contact onset") from error
        if onset != onset_raw or not 0 <= onset < progress.size:
            raise ValueError(f"sampled episode {index} has invalid contact onset")
        completed += int(float(np.max(progress[: onset + 1])) >= 6.0)
    return {
        "all_six_by_contact_n": completed,
        "all_six_by_contact_rate": completed / len(episodes),
    }


def canonical_policy_rows(rows):
    """Require the frozen 4x6 matrix and return treatment/seed order."""
    rows = list(rows)
    if len(rows) != len(ARM_ORDER) * len(TRAINING_SEEDS):
        raise ValueError("Presentation3 requires exactly 24 policy rows")
    expected = {(arm, seed) for arm in ARM_ORDER for seed in TRAINING_SEEDS}
    keyed = {}
    checkpoint_hashes = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("policy row must be a mapping")
        try:
            key = (str(row["arm"]), int(row["training_seed"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("policy row identity is malformed") from error
        if key in keyed:
            raise ValueError(f"duplicate policy identity: {key}")
        keyed[key] = row
        checkpoint_hashes.append(str(row.get("checkpoint_sha256", "")))
    if set(keyed) != expected:
        raise ValueError("policy matrix differs from the frozen four-by-six identity")
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("policy checkpoint hashes must be unique")
    return [keyed[(arm, seed)] for arm in ARM_ORDER for seed in TRAINING_SEEDS]


def fixed_impulse_metrics(diagnostic, *, caps):
    """Separate finalized productive first-event impulse from episode accumulation."""
    finalized = np.asarray(diagnostic["tracker_finalized"], dtype=bool)
    productive = np.asarray(diagnostic["tracker_productive"], dtype=bool)
    first_event = np.asarray(diagnostic["tracker_delivered_n_s"], dtype=float)
    cumulative = np.asarray(diagnostic["delivered_impulse_n_s"], dtype=float)
    impulse = np.asarray(
        diagnostic["lambda_windowed_constraint_read_n_m_s"], dtype=float
    )
    limits = np.asarray(caps, dtype=float)
    try:
        precontact_speed = np.asarray(
            diagnostic["tracker_v_precontact_m_s"], dtype=float
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("malformed fixed-reset tracker speed") from error
    if (
        precontact_speed.shape != finalized.shape
        or not np.all(np.isfinite(precontact_speed))
        or np.any(precontact_speed < 0.0)
    ):
        raise ValueError("malformed fixed-reset tracker speed")
    if not (
        finalized.ndim == productive.ndim == first_event.ndim == cumulative.ndim == 1
        and len(finalized) == len(productive) == len(first_event) == len(cumulative)
        and len(finalized) > 0
        and impulse.ndim == 2
        and impulse.shape[0] == len(finalized)
        and limits.shape == (impulse.shape[1],)
        and np.all(np.isfinite(first_event))
        and np.all(np.isfinite(cumulative))
        and np.all(np.isfinite(impulse))
        and np.all(np.isfinite(limits))
        and np.all(limits > 0.0)
    ):
        raise ValueError("malformed fixed-reset impulse trace")
    if not finalized[-1] or not productive[-1]:
        raise ValueError("first strike did not finalize as productive")
    first_finalization = int(np.flatnonzero(finalized)[0])
    if (
        not productive[first_finalization]
        or not np.all(finalized[first_finalization:])
        or not np.all(first_event[first_finalization:] == first_event[first_finalization])
        or not np.all(
            precontact_speed[first_finalization:]
            == precontact_speed[first_finalization]
        )
    ):
        if not np.all(
            first_event[first_finalization:] == first_event[first_finalization]
        ):
            raise ValueError("fixed-reset first-event latch drift")
        raise ValueError("fixed-reset tracker state is not latched at first finalization")
    first_value = float(first_event[first_finalization])
    cumulative_value = float(cumulative[-1])
    if first_value < 0.0 or cumulative_value + 1e-9 < first_value:
        raise ValueError("inconsistent delivered-impulse accumulators")
    return {
        "first_event_impulse_n_s": first_value,
        "cumulative_impulse_n_s": cumulative_value,
        "post_event_tail_n_s": cumulative_value - first_value,
        "precontact_axial_speed_m_s": float(
            precontact_speed[first_finalization]
        ),
        "max_lambda_cap_ratio": float(np.max(np.abs(impulse) / limits)),
    }


def waypoints_reached_by_contact(next_waypoint, contact):
    """Return ordered progress at first contact, inclusive of its substep."""
    progress = np.asarray(next_waypoint)
    contact_mask = np.asarray(contact, dtype=bool)
    if (
        progress.ndim != 1
        or contact_mask.ndim != 1
        or len(progress) != len(contact_mask)
        or len(progress) == 0
        or not np.all(np.isfinite(progress))
    ):
        raise ValueError("malformed waypoint/contact trace")
    if not np.any(contact_mask):
        return 0
    onset = int(np.flatnonzero(contact_mask)[0])
    return int(np.max(progress[: onset + 1]))


def precontact_geometry_metrics(positions, recorded_segment_error, contact, *, entry):
    """Measure the reset-to-contact path using the recorded finite-segment error."""
    position = np.asarray(positions, dtype=float)
    error = np.asarray(recorded_segment_error, dtype=float)
    contact_mask = np.asarray(contact, dtype=bool)
    start = np.asarray(entry, dtype=float)
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or error.shape != (position.shape[0],)
        or contact_mask.shape != (position.shape[0],)
        or start.shape != (3,)
        or position.shape[0] == 0
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(error))
        or not np.all(np.isfinite(start))
        or np.any(error < 0.0)
    ):
        raise ValueError("malformed pre-contact geometry trace")
    onset = int(np.flatnonzero(contact_mask)[0]) if np.any(contact_mask) else len(error) - 1
    position_pre = position[: onset + 1]
    error_pre = error[: onset + 1]
    path_length = float(np.linalg.norm(np.diff(position_pre, axis=0), axis=1).sum())
    direct = float(np.linalg.norm(position_pre[-1] - start))
    return {
        "window_substeps": onset + 1,
        "perpendicular_max_m": float(np.max(error_pre)),
        "perpendicular_rms_m": float(np.sqrt(np.mean(np.square(error_pre)))),
        "path_ratio": path_length / direct if direct > 0.0 else float("nan"),
    }


def peak_post_integration_qvel(post_integration, *, limit):
    """Apply the hardware rail to the same 500 Hz post-integration signal as CaT."""
    values = np.asarray(post_integration, dtype=float)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("malformed post-integration qvel trace")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("invalid qvel limit")
    peak = float(np.max(np.abs(values)))
    return peak, bool(peak <= float(limit))


def validate_artifact_join(sampled, fixed):
    """Fail closed unless sampled and fixed-reset evidence identify one policy."""
    if not isinstance(sampled, Mapping) or not isinstance(fixed, Mapping):
        raise ValueError("artifact identity must be a mapping")
    for field in (
        "checkpoint_sha256",
        "reset_state_digest",
        "training_revision",
        "asset_revision",
        "training_seed",
    ):
        if sampled.get(field) != fixed.get(field):
            raise ValueError(f"artifact join {field} mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the provenance-bound 24-policy Presentation3 result package."
    )
    parser.add_argument("--existing-fixed-root", type=Path, required=True)
    parser.add_argument("--d0-fixed-root", type=Path, required=True)
    parser.add_argument("--existing-sampled-root", type=Path, required=True)
    parser.add_argument("--d0-sampled-root", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--d0-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from evaluation.analysis.fixed_reset_video_library import (
        validate_wave1_policy_artifacts,
    )
    from scripts.diag_impulse_trace import validate_trace_leaf
    from scripts.eval_impulse import (
        _canonical_digest,
        _validated_physical_trace_digest,
        _validated_reset_state_digest,
    )

    existing_manifest_rows = load_frozen_training_manifest(
        args.existing_manifest,
        expected_sha256=EXPECTED_EXISTING_MANIFEST,
        expected_keys={
            (arm, seed)
            for arm in ("M", "V", "V+M")
            for seed in range(2, 8)
        },
    )
    d0_manifest_rows = load_frozen_training_manifest(
        args.d0_manifest,
        expected_sha256=EXPECTED_D0_MANIFEST,
        expected_keys={("V+D0", seed) for seed in range(2, 8)},
    )

    rows = []
    trajectories = {}
    sampled_input_records = []
    for sampled_root in (args.existing_sampled_root, args.d0_sampled_root):
        for summary_path in sorted(sampled_root.glob("*/summary.csv")):
            summary_bytes = summary_path.read_bytes()
            try:
                summary_rows = list(
                    csv.DictReader(io.StringIO(summary_bytes.decode("utf-8")))
                )
            except UnicodeDecodeError as error:
                raise ValueError(f"summary is not UTF-8: {summary_path}") from error
            if len(summary_rows) != 1:
                raise ValueError(f"expected one summary row: {summary_path}")
            summary = summary_rows[0]
            treatment = str(summary.get("treatment", ""))
            if treatment not in VIDEO_ARM_BY_TREATMENT:
                raise ValueError(f"unknown treatment in {summary_path}")
            trace_paths = list(summary_path.parent.glob("*_sampled_traces.npz"))
            if len(trace_paths) != 1:
                raise ValueError(f"expected one sampled trace: {summary_path.parent}")
            sampled_trace_path = trace_paths[0]
            artifact_sha = hashlib.sha256(sampled_trace_path.read_bytes()).hexdigest()
            if artifact_sha != summary.get("sampled_trace_artifact_sha256"):
                raise ValueError(f"sampled artifact SHA-256 mismatch: {sampled_trace_path}")
            with np.load(sampled_trace_path, allow_pickle=False) as archive:
                if archive.files != ["payload_json"]:
                    raise ValueError(f"sampled artifact schema mismatch: {sampled_trace_path}")
                encoded = archive["payload_json"]
                if encoded.dtype != np.uint8 or encoded.ndim != 1:
                    raise ValueError(f"sampled payload encoding mismatch: {sampled_trace_path}")
                payload = json.loads(encoded.tobytes().decode("utf-8"))
            recorded_payload_digest = payload.pop("payload_digest", None)
            recomputed_payload_digest = _canonical_digest(payload)
            payload["payload_digest"] = recorded_payload_digest
            if (
                recorded_payload_digest != recomputed_payload_digest
                or recorded_payload_digest != summary.get("sampled_trace_digest")
            ):
                raise ValueError(f"sampled payload digest mismatch: {sampled_trace_path}")
            is_d0 = treatment == "V+D0"
            expected_manifest = EXPECTED_D0_MANIFEST if is_d0 else EXPECTED_EXISTING_MANIFEST
            expected_training = EXPECTED_D0_TRAINING if is_d0 else EXPECTED_EXISTING_TRAINING
            expected_evaluation = (
                EXPECTED_D0_EVALUATION if is_d0 else EXPECTED_EXISTING_EVALUATION
            )
            expected_renderer = EXPECTED_D0_RENDERER if is_d0 else EXPECTED_EXISTING_RENDERER
            manifest_rows = d0_manifest_rows if is_d0 else existing_manifest_rows
            try:
                manifest_row = manifest_rows[
                    (treatment, int(summary["training_seed"]))
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{summary.get('name')}: not in frozen manifest") from error
            exact_summary = {
                "accepted_manifest_sha256": expected_manifest,
                "training_code_revision": expected_training,
                "training_asset_revision": EXPECTED_ASSET_REVISION,
                "git_revision": expected_evaluation,
                "reset_digest": EXPECTED_RESET_DIGEST,
                "n_episodes_sampled": "512",
            }
            for field, expected in exact_summary.items():
                if summary.get(field) != expected:
                    raise ValueError(f"{summary['name']}: {field} drift")
            validate_sampled_population_contract(
                summary,
                payload,
                manifest=expected_manifest,
                training_revision=expected_training,
                evaluation_revision=expected_evaluation,
                asset_revision=EXPECTED_ASSET_REVISION,
                manifest_row=manifest_row,
                config_identity=EXPECTED_CONFIG_IDENTITY_BY_TREATMENT[treatment],
            )
            sampled_input_records.append(
                {
                    "arm": treatment,
                    "training_seed": int(summary["training_seed"]),
                    "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                    "artifact_sha256": artifact_sha,
                }
            )
            for episode in payload["episodes"]:
                _validated_physical_trace_digest(
                    episode, require_recorded_digest=True
                )
                _validated_reset_state_digest(episode, require_recorded_digest=True)

            fixed_root = args.d0_fixed_root if is_d0 else args.existing_fixed_root
            video_leaf = fixed_root / "videos" / summary["name"]
            impulse_leaf = fixed_root / "impulse" / summary["name"]
            video_metadata = json.loads(
                (video_leaf / "metadata.json").read_text(encoding="utf-8")
            )
            expectations = {
                "arm": VIDEO_ARM_BY_TREATMENT[treatment],
                "training_seed": int(summary["training_seed"]),
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "reset_state_digest": EXPECTED_RESET_DIGEST,
                "training_revision": expected_training,
                "asset_revision": EXPECTED_ASSET_REVISION,
                "analysis_revision": expected_renderer,
            }
            validate_wave1_policy_artifacts(
                video_leaf, expectations, campaign="presentation3"
            )
            impulse_metadata = validate_trace_leaf(impulse_leaf)
            with np.load(video_leaf / "trace.npz", allow_pickle=False) as archive:
                video_trace = {key: archive[key] for key in archive.files}
            with np.load(impulse_leaf / "trace.npz", allow_pickle=False) as archive:
                impulse_trace = {key: archive[key] for key in archive.files}
            row, trajectory = assemble_result_row(
                summary,
                payload,
                video_metadata,
                video_trace,
                impulse_metadata,
                impulse_trace,
            )
            row.update(
                {
                    "sampled_summary_path": str(summary_path.resolve()),
                    "sampled_trace_path": str(sampled_trace_path.resolve()),
                    "fixed_video_path": str((video_leaf / "policy.mp4").resolve()),
                    "fixed_trajectory_path": str(
                        (video_leaf / "trajectory.png").resolve()
                    ),
                }
            )
            rows.append(row)
            trajectories[(treatment, int(summary["training_seed"]))] = trajectory

    validate_sampled_input_inventory(
        sampled_input_records,
        expected_sha256=EXPECTED_SAMPLED_INPUT_INVENTORY_SHA256,
    )
    outputs = write_result_package(rows, trajectories, args.output_dir)
    for label, path in outputs.items():
        print(f"{label}: {path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()

"""Root-owned metadata routing checks on tensor-only manager fixtures."""
from copy import deepcopy
import pytest
from tests.test_split_event036 import event_env
from tests.test_spline_exploration_pilot_contract import models


def test_shared_observation_schema_preserves_actor43_and_critic47():
    from src.tasks.hammer.rl.impact_speed_contract import _observation_metadata
    metadata=_observation_metadata(event_env())
    assert sum(metadata['observation_term_widths'])==43
    assert sum(metadata['critic_observation_term_widths'])==47
    assert metadata['observation_normalization']['critic_fixed_tail'][-1]=='credit_time_to_payout'


def test_shared_reward_profile_uses_truthful_two_tranche_timing():
    from src.tasks.hammer.rl.impact_speed_contract import _spline_reward_profile
    profile=_spline_reward_profile(event_env())
    assert profile['credit_timing']['early_fraction']==.5
    assert profile['credit_timing']['reserve_fraction']==.5
    assert profile['credit_timing']['observability']=='critic_only_raw_split_contact_age_balance_next_payout'


def test_shared_models_dispatch_new_marker_and_load_validation_rejects_drift():
    from src.tasks.hammer.rl.spline_split_event036_contract import split_event_metadata
    from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models,validate_impact_speed_metadata
    metadata={'schema_version':'impact_speed_v1','spline_split_event036':split_event_metadata(event_env())}
    a,c=models(mode='event')
    validate_impact_speed_models(a,c,metadata)
    validate_impact_speed_metadata(deepcopy(metadata),metadata)
    drift=deepcopy(metadata);drift['spline_split_event036']['critic_width']=43
    with pytest.raises(ValueError):validate_impact_speed_metadata(drift,drift)

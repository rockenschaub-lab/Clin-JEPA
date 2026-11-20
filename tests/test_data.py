import numpy as np
import pandas as pd

from clin_jepa.data.helpers.observation_filter import filter_obs_events
from clin_jepa.data.step06_trajectories import NO_OBSERVATIONS, hourly_sofa, hourly_texts, slice_windows

WINDOWS = {"t_max": 72, "min_len": 6, "stride": 12}


def test_window_slicing():
    assert slice_windows(5, WINDOWS) == []
    assert slice_windows(40, WINDOWS) == [(0, 40)]
    assert slice_windows(72, WINDOWS) == [(0, 72)]
    assert slice_windows(100, WINDOWS) == [(0, 72), (12, 72), (24, 72)]


def test_hour_without_events_gets_the_placeholder_text():
    events = pd.DataFrame(columns=["step_idx", "source_table", "variable_name", "numeric_value", "text_value", "unit"])
    assert hourly_texts(events, 2, NO_OBSERVATIONS) == [f"t=0h | {NO_OBSERVATIONS}", f"t=1h | {NO_OBSERVATIONS}"]


def test_hourly_sofa_keeps_the_first_value_of_an_hour():
    events = pd.DataFrame({
        "step_idx": [0, 0, 2, 2], "source_table": ["sofa", "sofa", "sofa", "vitalsign"],
        "variable_name": ["sofa_24hours", "sofa_24hours", "sofa_24hours", "heart_rate"],
        "numeric_value": [3.0, 5.0, 7.0, 90.0],
    })
    sofa = hourly_sofa(events, 4, {"variable_name": "sofa_24hours", "source_table": "sofa"})
    assert sofa.dtype == np.float32
    np.testing.assert_array_equal(sofa, np.array([3.0, np.nan, 7.0, np.nan], dtype=np.float32))


def test_state_text_filter_drops_categories_and_variables():
    events = pd.DataFrame({"category": ["vitals", "diagnosis", "kidney"],
                           "variable_name": ["heart_rate", "I10", "aki_stage_uo"]})
    kept = filter_obs_events(events, {"exclude_categories": ["diagnosis"], "exclude_variables": ["aki_stage_uo"]})
    assert kept["variable_name"].tolist() == ["heart_rate"]

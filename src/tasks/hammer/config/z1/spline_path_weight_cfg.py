"""Task identities for the isolated spline-path reward-weight screen."""

BASE_TASK = "Unitree-Z1-Hammer-Quintic-ThreeStarts-SplineSpeedCurriculum-Observation"
WEIGHTS = {"1x": -0.2, "5x": -1.0, "20x": -4.0}
TASKS = {
    arm: f"{BASE_TASK}-PathWeight-{arm}"
    for arm in WEIGHTS
}

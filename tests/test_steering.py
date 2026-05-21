from pathlib import Path

from asne.steering import apply_prompt_prefix, load_steering_conditions


def test_load_steering_conditions_from_example_config():
    conditions = load_steering_conditions(Path("configs/steering_conditions.example.yaml"))

    ids = {condition.id for condition in conditions}
    assert "neutral" in ids
    assert "threat" in ids
    assert all(condition.prompt_prefix for condition in conditions)


def test_apply_prompt_prefix_includes_prefix_and_stimulus():
    condition = load_steering_conditions(Path("configs/steering_conditions.example.yaml"))[0]
    result = apply_prompt_prefix(condition, "A short test stimulus.")

    assert condition.prompt_prefix in result
    assert "A short test stimulus." in result

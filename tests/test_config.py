from md_emg.config import apply_overrides


def test_repeats_is_cli_override_and_updates_run_name():
    cfg = {
        "experiment": {"name": "final", "seed": 2026},
        "data": {},
        "scenario": {"channel": "ch2", "phoneme": "all"},
        "cv": {"repeats": 1},
        "models": {"enabled": []},
        "compute": {},
        "output": {},
        "balance": {},
    }
    out = apply_overrides(cfg, repeats=10, channel="ch3", phoneme="theta")
    assert out["cv"]["repeats"] == 10
    assert out["experiment"]["name"] == "ph_theta_ch3_r10"

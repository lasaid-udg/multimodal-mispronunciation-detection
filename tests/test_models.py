from md_emg.models import SUPPORTED_MODELS, build_model, selected_class_weight


def _cfg():
    return {
        "balance": {"class_weight": "tune", "options": [None, "balanced"]},
        "models": {
            "svm": {"grid": [{"kernel": ["linear"], "C": [1.0]}]},
            "rf": {"grid": {"n_estimators": [10], "max_depth": [None]}},
            "dt": {"grid": {"max_depth": [None]}},
            "knn": {"grid": {"n_neighbors": [3]}},
            "gnb": {"grid": {"var_smoothing": [1e-9]}},
            "mlp": {"max_iter": 10, "grid": {"hidden_layer_sizes": [[5]], "activation": ["relu"]}},
            "lr": {"max_iter": 100, "grid": {"C": [1.0]}},
        },
    }


def test_all_models_build():
    cfg = _cfg()
    for name in SUPPORTED_MODELS:
        pipe, grid = build_model(name, cfg, seed=1)
        assert pipe is not None
        assert grid is not None


def test_class_weight_only_supported_models():
    cfg = _cfg()
    for name in ["svm", "rf", "dt", "lr"]:
        _, grid = build_model(name, cfg, seed=1)
        grids = grid if isinstance(grid, list) else [grid]
        assert all("clf__class_weight" in g for g in grids)
    for name in ["knn", "gnb", "mlp"]:
        _, grid = build_model(name, cfg, seed=1)
        grids = grid if isinstance(grid, list) else [grid]
        assert all("clf__class_weight" not in g for g in grids)


def test_selected_class_weight():
    assert selected_class_weight({"clf__class_weight": "balanced"}) == "balanced"
    assert selected_class_weight({}) == "none"


def test_mlp_v31_diagnostics_config():
    cfg = _cfg()
    cfg["models"]["mlp"].update({
        "max_iter": 3000,
        "early_stopping": True,
        "validation_fraction": 0.10,
        "n_iter_no_change": 20,
    })
    pipe, _ = build_model("mlp", cfg, seed=1)
    clf = pipe.named_steps["clf"]
    assert clf.max_iter == 3000
    assert clf.early_stopping is True
    assert clf.n_iter_no_change == 20

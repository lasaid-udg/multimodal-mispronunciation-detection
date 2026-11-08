from __future__ import annotations

from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SUPPORTED_MODELS = ("svm", "rf", "dt", "knn", "gnb", "mlp", "lr")
CLASS_WEIGHT_MODELS = {"svm", "rf", "dt", "lr"}


def _base_steps(scale: bool):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return steps


def _prefix_grid(raw):
    """Prefix classifier parameters for Pipeline/GridSearchCV."""
    if isinstance(raw, list):
        return [{f"clf__{k}": v for k, v in g.items()} for g in raw]
    return {f"clf__{k}": v for k, v in raw.items()}


def _apply_class_weight(grid, name: str, cfg: dict):
    balance = cfg.get("balance", {})
    mode = balance.get("class_weight", "tune")
    if name not in CLASS_WEIGHT_MODELS:
        return grid

    if mode == "tune":
        values = balance.get("options", [None, "balanced"])
    elif mode == "balanced":
        values = ["balanced"]
    elif mode in ("none", None, False):
        values = [None]
    else:
        raise ValueError(f"Unknown balance.class_weight mode: {mode}")

    grids = grid if isinstance(grid, list) else [grid]
    out = []
    for g in grids:
        h = deepcopy(g)
        h["clf__class_weight"] = values
        out.append(h)
    return out if isinstance(grid, list) else out[0]


def build_model(name: str, cfg: dict, seed: int):
    name = name.lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {name}. Supported: {', '.join(SUPPORTED_MODELS)}")

    mcfg = cfg["models"][name]

    if name == "svm":
        clf = SVC(probability=False, random_state=seed)
        pipe = Pipeline(_base_steps(scale=True) + [("clf", clf)])
    elif name == "rf":
        clf = RandomForestClassifier(random_state=seed, n_jobs=1)
        pipe = Pipeline(_base_steps(scale=False) + [("clf", clf)])
    elif name == "dt":
        clf = DecisionTreeClassifier(random_state=seed)
        pipe = Pipeline(_base_steps(scale=False) + [("clf", clf)])
    elif name == "knn":
        clf = KNeighborsClassifier(n_jobs=1)
        pipe = Pipeline(_base_steps(scale=True) + [("clf", clf)])
    elif name == "gnb":
        clf = GaussianNB()
        pipe = Pipeline(_base_steps(scale=False) + [("clf", clf)])
    elif name == "mlp":
        clf = MLPClassifier(
            random_state=seed,
            max_iter=int(mcfg.get("max_iter", 3000)),
            early_stopping=bool(mcfg.get("early_stopping", True)),
            validation_fraction=float(mcfg.get("validation_fraction", 0.10)),
            n_iter_no_change=int(mcfg.get("n_iter_no_change", 20)),
        )
        pipe = Pipeline(_base_steps(scale=True) + [("clf", clf)])
    elif name == "lr":
        clf = LogisticRegression(random_state=seed, max_iter=int(mcfg.get("max_iter", 2000)), solver="lbfgs")
        pipe = Pipeline(_base_steps(scale=True) + [("clf", clf)])

    grid = _prefix_grid(mcfg.get("grid", {}))
    grid = _apply_class_weight(grid, name, cfg)
    return pipe, grid


def selected_class_weight(best_params: dict) -> str:
    value = best_params.get("clf__class_weight", None)
    return "balanced" if value == "balanced" else "none"


def estimator_diagnostics(estimator) -> dict:
    """Return lightweight diagnostics for the selected/refit classifier.

    For iterative estimators this records the actual iteration count and whether
    the configured maximum was reached.  This is especially useful for MLP,
    where reaching max_iter is a practical convergence flag.
    """
    clf = estimator.named_steps.get("clf", estimator) if hasattr(estimator, "named_steps") else estimator
    out = {
        "estimator_class": clf.__class__.__name__,
        "n_iter": None,
        "max_iter": getattr(clf, "max_iter", None),
        "reached_max_iter": None,
        "best_validation_score": getattr(clf, "best_validation_score_", None),
    }
    n_iter = getattr(clf, "n_iter_", None)
    if n_iter is not None:
        try:
            # LogisticRegression may expose one value per class.
            if hasattr(n_iter, "__len__") and not isinstance(n_iter, (str, bytes)):
                n = int(max(n_iter))
            else:
                n = int(n_iter)
            out["n_iter"] = n
            # A negative max_iter (e.g. SVC max_iter=-1) means unlimited,
            # so it must not be interpreted as hitting an iteration limit.
            if out["max_iter"] is not None and int(out["max_iter"]) >= 0:
                out["reached_max_iter"] = bool(n >= int(out["max_iter"]))
        except Exception:
            out["n_iter"] = str(n_iter)
    return out


def continuous_score(estimator, X):
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X), "decision_function"
    if hasattr(estimator, "predict_proba"):
        classes = list(estimator.classes_)
        idx = classes.index(1)
        return estimator.predict_proba(X)[:, idx], "probability"
    return None, None

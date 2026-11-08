from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def apply_overrides(cfg: dict, *, n_jobs=None, seed=None, outer_folds=None, inner_folds=None,
                    repeats=None, models=None, data_path=None, output_root=None, save_models=None,
                    class_weight=None, verbose=None, channel=None, phoneme=None) -> dict:
    c = deepcopy(cfg)
    if n_jobs is not None: c["compute"]["n_jobs"] = int(n_jobs)
    if seed is not None: c["experiment"]["seed"] = int(seed)
    if outer_folds is not None: c["cv"]["outer_folds"] = int(outer_folds)
    if inner_folds is not None: c["cv"]["inner_folds"] = int(inner_folds)
    if repeats is not None: c["cv"]["repeats"] = int(repeats)
    if models: c["models"]["enabled"] = list(models)
    if data_path is not None: c["data"]["path"] = str(data_path)
    if output_root is not None: c["output"]["root"] = str(output_root)
    if save_models is not None: c["compute"]["save_models"] = bool(save_models)
    if class_weight is not None: c.setdefault("balance", {})["class_weight"] = class_weight
    if verbose is not None: c["compute"]["verbose"] = bool(verbose)
    if channel is not None: c["scenario"]["channel"] = str(channel).lower()
    if phoneme is not None: c["scenario"]["phoneme"] = str(phoneme).lower()

    # When scenario dimensions are supplied on the CLI, make run names
    # self-describing so channel/phoneme outputs cannot be confused.
    if channel is not None or phoneme is not None or repeats is not None:
        ch = str(c["scenario"].get("channel", "ch2")).lower()
        ph = str(c["scenario"].get("phoneme", "all")).lower()
        base = f"general_{ch}" if ph == "all" else f"ph_{ph}_{ch}"
        nrep = int(c.get("cv", {}).get("repeats", 1))
        c["experiment"]["name"] = f"{base}_r{nrep}" if nrep > 1 else base
    return c

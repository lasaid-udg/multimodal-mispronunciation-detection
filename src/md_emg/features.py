from __future__ import annotations

import re

AUDIO_START = "duration"
AUDIO_END = "zero_crossing_rate"


def build_feature_registry(columns: list[str]) -> dict:
    cols = list(columns)
    if AUDIO_START not in cols or AUDIO_END not in cols:
        raise ValueError("Could not identify acoustic feature boundaries duration..zero_crossing_rate")
    i0, i1 = cols.index(AUDIO_START), cols.index(AUDIO_END)
    audio = cols[i0:i1+1]
    channels = {}
    for ch in range(1, 5):
        patt = re.compile(rf"(^CH{ch}_|_CH{ch}$|_CH{ch}_)" , re.IGNORECASE)
        channels[f"ch{ch}"] = [c for c in cols if patt.search(c)]
    return {"audio": audio, **channels}


def predictor_columns(registry: dict, channel: str, exclude_order_time: bool = True) -> tuple[list[str], list[str]]:
    audio = list(registry["audio"])
    emg = list(registry[channel.lower()])
    if exclude_order_time:
        emg = [c for c in emg if not c.lower().startswith("order_ch") and not c.lower().startswith("time_ch")]
    return audio, emg

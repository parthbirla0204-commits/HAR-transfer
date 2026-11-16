"""Label mapping from every dataset's native labels to the 5 common activities."""
import pandas as pd
from .config import COMMON_ACTIVITIES

LABEL_MAP = {
    "uci_har":     {1: "walking", 2: "upstairs", 3: "downstairs", 4: "sitting", 5: "standing", 6: None},
    "pamap2":      {4: "walking", 12: "upstairs", 13: "downstairs", 2: "sitting", 3: "standing"},
    "wisdm":       {"Walking": "walking", "Upstairs": "upstairs", "Downstairs": "downstairs",
                    "Sitting": "sitting", "Standing": "standing", "Jogging": None},
    "motionsense": {"wlk": "walking", "ups": "upstairs", "dws": "downstairs",
                    "sit": "sitting", "std": "standing", "jog": None},
    "hhar":        {"walk": "walking", "stairsup": "upstairs", "stairsdown": "downstairs",
                    "sit": "sitting", "stand": "standing", "bike": None},
}


def mapping_table() -> pd.DataFrame:
    """Table for the paper: common activity -> native label(s) per dataset (None = dropped)."""
    rows = []
    for act in COMMON_ACTIVITIES:
        row = {"common activity": act}
        for ds, m in LABEL_MAP.items():
            row[ds] = ", ".join(str(k) for k, v in m.items() if v == act)
        rows.append(row)
    dropped = {"common activity": "(dropped)"}
    for ds, m in LABEL_MAP.items():
        dropped[ds] = ", ".join(str(k) for k, v in m.items() if v is None) or "other classes"
    rows.append(dropped)
    return pd.DataFrame(rows)


def to_index(name):
    return COMMON_ACTIVITIES.index(name) if name in COMMON_ACTIVITIES else None

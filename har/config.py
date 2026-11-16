"""Central configuration: protocol constants and the hyper-parameter table."""
from dataclasses import dataclass, asdict

FS = 50                 # Hz, every dataset is resampled to this rate
WIN = 128               # samples per window (2.56 s)
STEP = 64               # 50 % overlap
G = 9.80665             # m/s^2 -> g conversion
COMMON_ACTIVITIES = ["walking", "upstairs", "downstairs", "sitting", "standing"]
FRACTIONS = [0.01, 0.05, 0.10, 0.20, 1.0]
N_FOLDS = 5
N_SEEDS = 5


@dataclass
class TrainCfg:
    # identical for every method
    optimizer: str = "Adam"
    max_epochs: int = 40
    batch_size: int = 64
    patience: int = 8            # early stopping on validation macro-F1
    weight_decay: float = 1e-4
    # method-specific learning rates (documented in the hyper-parameter table)
    lr_pretrain: float = 1e-3    # source pre-training
    lr_scratch: float = 1e-3     # scratch / published models
    lr_head: float = 1e-3        # frozen backbone + new head
    lr_finetune: float = 1e-4    # full fine-tuning (low LR)
    lr_max_adapt: float = 5e-4   # proposed method: max per-layer LR
    lr_floor: float = 0.2        # proposed: LR of a selected layer = lr_max*(floor+(1-floor)*r_l)
    rho: float = 0.3             # proposed: layer trainable iff r_l = d_l/max(d) >= rho
    lr_decay: float = 0.5        # layer-wise LR decay baseline
    mmd_samples: int = 512
    val_frac_subjects: float = 0.2

    def table(self):
        return [{"hyper-parameter": k, "value": str(v)} for k, v in asdict(self).items()]

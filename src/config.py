from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StudyAreaConfig:
    bbox: Tuple[float, float, float, float] = (-97.7, 32.4, -96.4, 33.5)


@dataclass(frozen=True)
class DataConfig:
    start_date: str = "2016-01-01"
    end_date: str = "2024-12-31"
    sample_per_month: int = 3000
    scale_m: int = 1000
    seed: int = 42


@dataclass(frozen=True)
class Paths:
    data_dir: str = "data"
    raw_samples: str = "data/dfw_samples.parquet"
    cleaned_samples: str = "data/dfw_samples_clean.parquet"
    figures_dir: str = "outputs/figures"
    models_dir: str = "outputs/models"
    tables_dir: str = "outputs/tables"


STUDY_AREA = StudyAreaConfig()
DATA = DataConfig()
PATHS = Paths()

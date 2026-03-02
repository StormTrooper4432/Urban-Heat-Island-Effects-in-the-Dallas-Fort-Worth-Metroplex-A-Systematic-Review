# Urban Heat Island Severity Modeling (DFW)

This project accompanies the paper "Urban Heat Island Effects in the Dallas-Fort Worth Metroplex: A Systematic Review" submitted to the 2026 Modeling the Future Challenge.

## Setup
1. Create and activate a Python environment.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Authenticate Earth Engine:

```bash
earthengine authenticate
```

## Run
```bash
python -m src.run_pipeline
```

If you want to skip fetching:

```bash
python -m src.run_pipeline --skip-fetch
```

## Outputs
- Raw samples: `data/dfw_samples.parquet`
- Cleaned samples: `data/dfw_samples_clean.parquet`
- Figures: `outputs/figures/*.png`
- Tables: `outputs/tables/*.csv`

# Urban Heat Island Severity Modeling (DFW)

This project builds a full, reproducible pipeline to predict Urban Heat Island (UHI) severity in the Dallas-Fort Worth metroplex using Google Earth Engine datasets. It includes data extraction, cleaning, feature selection, EDA, multiple ML models, deep learning, evaluation, and ARIMA forecasting. This project accompanies the paper "Urban Heat Island Effects in the Dallas-Fort Worth Metroplex: A Systematic Review" submitted to the 2026 Modeling the Future Challenge.

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

If you have already downloaded the data and want to skip fetching:

```bash
python -m src.run_pipeline --skip-fetch
```

## Outputs
- Raw samples: `data/dfw_samples.parquet`
- Cleaned samples: `data/dfw_samples_clean.parquet`
- Figures: `outputs/figures/*.png`
- Tables: `outputs/tables/*.csv`
- Report: `paper.md`

## Notes
- The DFW study area is defined as a bounding box in `src/config.py`.
- You can adjust the date range, sample size, and resolution in `src/config.py`.
- Data extraction uses Google Earth Engine datasets and may require quota considerations.

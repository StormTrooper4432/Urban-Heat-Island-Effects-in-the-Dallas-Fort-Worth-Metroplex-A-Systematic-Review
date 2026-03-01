from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from .config import STUDY_AREA, DATA, PATHS
from .utils import ensure_dirs, ts_print


def initialize_ee() -> None:
    import ee

    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='urban-heat-islands-67')


def get_dfw_geometry():
    import ee

    west, south, east, north = STUDY_AREA.bbox
    return ee.Geometry.Rectangle([west, south, east, north])


def _prep_modis_lst(img):
    import ee

    lst_day = img.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("lst_day_c")
    lst_night = img.select("LST_Night_1km").multiply(0.02).subtract(273.15).rename("lst_night_c")

    qc_day = img.select("QC_Day").bitwiseAnd(3).lte(2)
    qc_night = img.select("QC_Night").bitwiseAnd(3).lte(2)
    valid_mask = (
        img.select("LST_Day_1km").gt(0)
        .And(img.select("LST_Night_1km").gt(0))
        .And(qc_day)
        .And(qc_night)
    )
    return ee.Image.cat([lst_day, lst_night]).updateMask(valid_mask)


def _prep_modis_vi(img):
    import ee

    ndvi = img.select("NDVI").multiply(0.0001).rename("ndvi")
    evi = img.select("EVI").multiply(0.0001).rename("evi")
    valid_mask = img.select("NDVI").gte(-0.2).And(img.select("NDVI").lte(1.0))
    return ee.Image.cat([ndvi, evi]).updateMask(valid_mask)


def _prep_viirs(img):
    return img.select("avg_rad").rename("night_lights")


def _prep_era5(img):
    return img.select("temperature_2m").subtract(273.15).rename("t2m_c")


def _prep_modis_albedo(img):
    return img.select("Albedo_WSA_shortwave").multiply(0.001).rename("albedo_wsa_sw")


def build_monthly_composite(start: str, end: str):
    import ee

    lst_ic = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start, end).map(_prep_modis_lst)
    vi_ic = ee.ImageCollection("MODIS/061/MOD13A2").filterDate(start, end).map(_prep_modis_vi)
    albedo_ic = ee.ImageCollection("MODIS/061/MCD43A3").filterDate(start, end).map(_prep_modis_albedo)
    viirs_ic = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").filterDate(start, end).map(_prep_viirs)
    era5_ic = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(start, end).map(_prep_era5)

    lst = lst_ic.mean()
    vi = vi_ic.mean()
    albedo = albedo_ic.mean()
    viirs = viirs_ic.mean()
    era5 = era5_ic.mean()

    nlcd = (
        ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")
        .filter(ee.Filter.eq("system:index", "2019"))
        .first()
    )
    impervious = nlcd.select("impervious").rename("impervious")
    landcover = nlcd.select("landcover").rename("landcover")

    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation")
    terrain = ee.Terrain.products(elevation)
    slope = terrain.select("slope").rename("slope")
    aspect = terrain.select("aspect").rename("aspect")

    tcc = (
        ee.ImageCollection("USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5")
        .filter(ee.Filter.eq("year", 2019))
        .first()
    )
    tree_cover = tcc.select("NLCD_Percent_Tree_Canopy_Cover").rename("tree_cover")

    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").unmask(0)
    water_dist = water.distance(ee.Kernel.euclidean(100000, "meters")).rename("dist_to_water_m")

    latlon = ee.Image.pixelLonLat().rename(["lon", "lat"])

    composite = ee.Image.cat([
        lst,
        vi,
        albedo,
        viirs,
        era5,
        impervious,
        landcover,
        elevation,
        slope,
        aspect,
        tree_cover,
        water_dist,
        latlon,
    ])

    rural_mask = impervious.lt(20).And(vi.select("ndvi").gt(0.4))
    rural_mean = (
        composite.select("lst_day_c")
        .updateMask(rural_mask)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=get_dfw_geometry(),
            scale=DATA.scale_m,
            bestEffort=True,
        )
        .get("lst_day_c")
    )

    fallback_mean = (
        composite.select("lst_day_c")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=get_dfw_geometry(),
            scale=DATA.scale_m,
            bestEffort=True,
        )
        .get("lst_day_c")
    )

    rural_mean_num = ee.Number(
        ee.Algorithms.If(
            rural_mean,
            rural_mean,
            ee.Algorithms.If(fallback_mean, fallback_mean, 0),
        )
    )
    uhi = composite.select("lst_day_c").subtract(ee.Image.constant(rural_mean_num)).rename("uhi_c")
    composite = composite.addBands([uhi, ee.Image.constant(rural_mean_num).rename("rural_mean_c")])

    return composite


def sample_month(month_start: pd.Timestamp) -> pd.DataFrame:
    import ee
    import geemap

    month_end = (month_start + pd.offsets.MonthEnd(1)).to_pydatetime()
    start = month_start.to_pydatetime()

    comp = build_monthly_composite(start.isoformat(), month_end.isoformat())

    samples = comp.sample(
        region=get_dfw_geometry(),
        scale=DATA.scale_m,
        numPixels=DATA.sample_per_month,
        seed=DATA.seed,
        dropNulls=False,
        tileScale=4,
        geometries=False,
    )

    samples = samples.map(
        lambda f: f.set({
            "date": month_start.strftime("%Y-%m-%d"),
            "year": int(month_start.year),
            "month": int(month_start.month),
        })
    )

    if hasattr(geemap, "ee_to_df"):
        df = geemap.ee_to_df(samples)
    else:
        df = geemap.ee_to_pandas(samples)
    ts_print(f"Month {month_start.strftime('%Y-%m')} samples: {len(df)}")
    return df


def fetch_monthly_samples() -> pd.DataFrame:
    initialize_ee()

    months = pd.date_range(DATA.start_date, DATA.end_date, freq="MS")
    frames = []
    for month_start in tqdm(months, desc="Sampling monthly data"):
        frames.append(sample_month(month_start))

    df = pd.concat(frames, ignore_index=True)
    ts_print(f"Total samples collected: {len(df)}")
    ensure_dirs([PATHS.data_dir])
    df.to_parquet(PATHS.raw_samples, index=False)
    return df


def load_or_fetch_samples() -> pd.DataFrame:
    import pandas as pd

    try:
        try:
            return pd.read_parquet(PATHS.raw_samples)
        except Exception:
            return pd.read_parquet(PATHS.raw_samples, engine="pyarrow", use_legacy_dataset=True)
    except Exception:
        return fetch_monthly_samples()

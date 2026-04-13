"""
master_merge.py
===============
Builds master_final_v3.csv from 10 primary sources.
DBA Capstone — TCFD Framework | Karthikeyan Venkatesan
Walsh College & Deakin University, 2026

Usage:
    python src/data_pipeline/master_merge.py --raw_dir data/raw/ --out data/master_final_v3.csv

Raw data files needed in data/raw/:
    master_final_v2.csv             (NFHS-5 base — 707 × 29)
    ssrn_datasheet.xls              (Full NFHS-5 109 variables)
    25982521.zip                    (Mukherji et al. 2024 PDF — UHCd scores)
    IHME-GBD_2023_DATA-9b13d1c4.zip (IHME GBD state-level DALYs)
    A-1_Census_2011.xlsx            (Census A-1 district population/area)
    NFHS_Policy_Tracker_A.xlsx      (Boundary-changed districts flag)
    PMJAY_UHC_India_Data_Validated.xlsx
    NHA_District_DIU_Cards_Hospitals.xlsx
    gadm41_IND_shp.zip              (GADM v4.1 district shapefiles)
"""

import argparse
import logging
import zipfile
import io
import re
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import pdfplumber
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Normalise district/state names for consistent joins ──────────────
def norm(s):
    return str(s).upper().strip().replace("  ", " ")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ── State capital coordinates ─────────────────────────────────────────
STATE_CAPITALS = {
    "ANDHRA PRADESH": (17.366, 78.476), "ARUNACHAL PRADESH": (27.083, 93.617),
    "ASSAM": (26.144, 91.736), "BIHAR": (25.594, 85.137),
    "CHANDIGARH": (30.733, 76.779), "CHHATTISGARH": (21.250, 81.630),
    "DELHI": (28.613, 77.209), "GOA": (15.499, 73.824),
    "GUJARAT": (23.022, 72.571), "HARYANA": (30.733, 76.779),
    "HIMACHAL PRADESH": (31.104, 77.173), "JAMMU & KASHMIR": (34.083, 74.797),
    "JHARKHAND": (23.344, 85.309), "KARNATAKA": (12.971, 77.594),
    "KERALA": (8.524, 76.936), "LADAKH": (34.166, 77.584),
    "LAKSHADWEEP": (10.564, 72.638), "MADHYA PRADESH": (23.259, 77.412),
    "MAHARASHTRA": (19.076, 72.877), "MANIPUR": (24.817, 93.944),
    "MEGHALAYA": (25.576, 91.883), "MIZORAM": (23.729, 92.717),
    "NAGALAND": (25.674, 94.110), "ODISHA": (20.296, 85.825),
    "PUDUCHERRY": (11.934, 79.830), "PUNJAB": (30.900, 75.857),
    "RAJASTHAN": (26.912, 75.787), "SIKKIM": (27.332, 88.612),
    "TAMIL NADU": (13.082, 80.270), "TELANGANA": (17.366, 78.476),
    "TRIPURA": (23.831, 91.286), "UTTAR PRADESH": (26.846, 80.946),
    "UTTARAKHAND": (30.316, 78.032), "WEST BENGAL": (22.572, 88.363),
    "ANDAMAN & NICOBAR ISLANDS": (11.623, 92.726),
}

# ── UHCd extraction from Mukherji et al. 2024 PDF ────────────────────
UHCD_STATE_MAP = {
    "Andaman": "ANDAMAN & NICOBAR ISLANDS", "Andhra": "ANDHRA PRADESH",
    "Arunachal": "ARUNACHAL PRADESH", "Himachal": "HIMACHAL PRADESH",
    "Jammu": "JAMMU & KASHMIR", "Andhra Pradesh": "ANDHRA PRADESH",
    "Assam": "ASSAM", "Bihar": "BIHAR", "Chandigarh": "CHANDIGARH",
    "Chhattisgarh": "CHHATTISGARH", "Goa": "GOA", "Gujarat": "GUJARAT",
    "Haryana": "HARYANA", "Jharkhand": "JHARKHAND", "Karnataka": "KARNATAKA",
    "Kerala": "KERALA", "Ladakh": "LADAKH", "Madhya Pradesh": "MADHYA PRADESH",
    "Maharashtra": "MAHARASHTRA", "Manipur": "MANIPUR", "Meghalaya": "MEGHALAYA",
    "Mizoram": "MIZORAM", "Nagaland": "NAGALAND", "Odisha": "ODISHA",
    "Puducherry": "PUDUCHERRY", "Punjab": "PUNJAB", "Rajasthan": "RAJASTHAN",
    "Sikkim": "SIKKIM", "Tamil Nadu": "TAMIL NADU", "Telangana": "TELANGANA",
    "Tripura": "TRIPURA", "Uttar Pradesh": "UTTAR PRADESH",
    "Uttarakhand": "UTTARAKHAND", "West Bengal": "WEST BENGAL",
}

def extract_uhcd_from_pdf(zip_path: str) -> pd.DataFrame:
    """Extract UHCd scores from Mukherji et al. 2024 Technical Report PDF."""
    pdf_name = "BLT.23.290854  Mukherji Technical report.pdf"
    records = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(pdf_name) as f:
            data = f.read()
    pattern = re.compile(
        r"^(.+?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+"
        r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(Low|Medium|High)\s+UHC"
    )
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[30:50]:
            for line in (page.extract_text() or "").split("\n"):
                m = pattern.match(line.strip())
                if m:
                    raw = m.group(1).strip()
                    state = next(
                        (UHCD_STATE_MAP[k] for k in UHCD_STATE_MAP if raw.startswith(k)),
                        raw.split()[0].upper()
                    )
                    district = raw[len(next((k for k in UHCD_STATE_MAP if raw.startswith(k)), raw.split()[0])):].strip()
                    records.append({
                        "_key": norm(district) + "||" + state,
                        "RMNCH": float(m.group(2)), "ID": float(m.group(3)),
                        "NCD": float(m.group(4)), "SCA": float(m.group(5)),
                        "FRP": float(m.group(6)), "UHCd": float(m.group(7)),
                        "UHCd_Tercile": m.group(8),
                    })
    df = pd.DataFrame(records).groupby("_key").first().reset_index()
    log.info(f"UHCd extracted: {len(df)} districts")
    return df


def compute_geo_features(shp_zip: str) -> pd.DataFrame:
    """Compute geographic variables from GADM v4.1 district shapefiles."""
    shp_dir = "/tmp/gadm_shp"
    os.makedirs(shp_dir, exist_ok=True)
    with zipfile.ZipFile(shp_zip) as z:
        z.extractall(shp_dir)
    gdf = gpd.read_file(f"{shp_dir}/gadm41_IND_2.shp")
    gdf_utm = gdf.to_crs("EPSG:32644")
    gdf["geo_area_km2"] = (gdf_utm.geometry.area / 1e6).round(1)
    gdf["geo_perimeter_km"] = (gdf_utm.geometry.length / 1000).round(2)
    gdf["geo_compactness"] = (
        4 * 3.14159 * gdf["geo_area_km2"] / (gdf["geo_perimeter_km"] ** 2)
    ).round(4)
    centroids = gdf_utm.geometry.centroid.to_crs("EPSG:4326")
    gdf["geo_centroid_lat"] = centroids.y.round(4)
    gdf["geo_centroid_lon"] = centroids.x.round(4)
    # Distance to state capital
    state_fix = {
        "ANDAMAN AND NICOBAR": "ANDAMAN & NICOBAR ISLANDS",
        "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
    }
    gdf["_state_norm"] = gdf["NAME_1"].apply(
        lambda x: state_fix.get(x.upper(), x.upper())
    )
    gdf["geo_dist_to_capital_km"] = gdf.apply(
        lambda r: round(haversine(
            r["geo_centroid_lat"], r["geo_centroid_lon"],
            *STATE_CAPITALS.get(r["_state_norm"], (0, 0))
        ), 1) if r["_state_norm"] in STATE_CAPITALS else np.nan, axis=1
    )
    # Geography tax composite
    valid = gdf[["geo_area_km2", "geo_dist_to_capital_km", "geo_compactness"]].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(valid)
    geo_tax = (scaled[:, 0] * 0.3 + scaled[:, 1] * 0.5 + (1 - scaled[:, 2]) * 0.2).round(4)
    gdf.loc[valid.index, "geo_tax_index"] = geo_tax
    gdf["geo_tax_quintile"] = pd.qcut(
        gdf["geo_tax_index"].dropna(), 5,
        labels=["Q1_low_tax", "Q2", "Q3", "Q4", "Q5_high_tax"]
    ).reindex(gdf.index)
    gdf["_key_g"] = gdf["NAME_2"].apply(norm) + "||" + gdf["_state_norm"]
    cols = ["_key_g", "geo_area_km2", "geo_perimeter_km", "geo_compactness",
            "geo_centroid_lat", "geo_centroid_lon", "geo_dist_to_capital_km",
            "geo_tax_index", "geo_tax_quintile", "GID_2"]
    log.info(f"Geo features computed for {len(gdf)} GADM districts")
    return gdf[cols].copy()


def build_master(raw_dir: str, out_path: str) -> pd.DataFrame:
    """Full merge pipeline — builds master_final_v3.csv from all sources."""
    log.info("=== Building master_final_v3.csv ===")

    # S1: Base NFHS-5
    master = pd.read_csv(f"{raw_dir}/master_final_v2.csv")
    master["_key"] = master["District_norm"].apply(norm) + "||" + master["State_norm"].apply(norm)
    log.info(f"Base: {master.shape}")

    # S1b: Full NFHS-5 (SSRN)
    ssrn = pd.read_excel(f"{raw_dir}/ssrn_datasheet.xls", engine="xlrd")
    ssrn["_key"] = ssrn["District Names"].apply(norm) + "||" + ssrn["State/UT"].apply(norm)
    skip = {c.strip().lower() for c in master.columns}
    new_cols = [c for c in ssrn.columns if c not in ("District Names", "State/UT")
                and c.strip().lower() not in skip]
    ssrn_dd = ssrn.groupby("_key").first().reset_index()
    master = master.merge(ssrn_dd[["_key"] + new_cols], on="_key", how="left")
    log.info(f"After SSRN: {master.shape}, matched {master[new_cols[0]].notna().sum()}/707")

    # S2: UHCd
    uhcd = extract_uhcd_from_pdf(f"{raw_dir}/25982521.zip")
    master = master.merge(
        uhcd[["_key", "RMNCH", "ID", "NCD", "SCA", "FRP", "UHCd", "UHCd_Tercile"]],
        on="_key", how="left"
    )
    log.info(f"UHCd matched: {master['UHCd'].notna().sum()}/707")

    # S4: IHME GBD
    with zipfile.ZipFile(f"{raw_dir}/IHME-GBD_2023_DATA-9b13d1c4.zip") as z:
        with z.open(z.namelist()[0]) as f:
            gbd = pd.read_csv(f)
    gbd_f = gbd[
        (gbd.measure_name == "DALYs (Disability-Adjusted Life Years)") &
        (gbd.metric_name == "Rate") & (gbd.sex_name == "Both") &
        (gbd.age_name == "All ages") & (gbd.cause_name == "All causes") &
        (gbd.year == 2021)
    ][["location_name", "val", "lower", "upper"]].copy()
    gbd_f.columns = ["loc", "gbd_daly_rate_2021", "gbd_daly_lower", "gbd_daly_upper"]
    gbd_f["_sg"] = gbd_f["loc"].apply(
        lambda x: {"Jammu & Kashmir and Ladakh": "JAMMU & KASHMIR"}.get(x, x.upper())
    )
    master = master.merge(
        gbd_f[gbd_f["_sg"].str.len() > 3][["_sg", "gbd_daly_rate_2021", "gbd_daly_lower", "gbd_daly_upper"]],
        left_on="State_norm", right_on="_sg", how="left"
    ).drop(columns=["_sg"])
    log.info(f"GBD matched: {master['gbd_daly_rate_2021'].notna().sum()}/707")

    # S5: Census A-1
    census = pd.read_excel(f"{raw_dir}/A-1_Census_2011.xlsx", header=None)
    data = census.iloc[4:].copy()
    data.columns = ["sc", "dc", "subd", "nc", "name", "ru", "vi", "vu", "t", "hh",
                    "persons", "m", "f", "area", "pd_c"]
    dist = data[
        (data["subd"].astype(str).str.strip() == "00000") &
        (data["dc"].astype(str).str.strip() != "000") &
        (data["ru"].astype(str).str.strip() == "Total")
    ][["name", "persons", "area", "pd_c"]].copy()
    dist.columns = ["_n", "census_pop_2011", "census_area_sqkm", "census_pop_density"]
    dist["_kc"] = dist["_n"].apply(norm)
    for c in ["census_pop_2011", "census_area_sqkm", "census_pop_density"]:
        dist[c] = pd.to_numeric(dist[c], errors="coerce")
    master = master.merge(
        dist.groupby("_kc").first().reset_index()[["_kc", "census_pop_2011", "census_area_sqkm", "census_pop_density"]],
        left_on="District_norm", right_on="_kc", how="left"
    ).drop(columns=["_kc"])
    log.info(f"Census matched: {master['census_area_sqkm'].notna().sum()}/707")

    # S8: Boundary flag
    tracker = pd.read_excel(f"{raw_dir}/NFHS_Policy_Tracker_A.xlsx")
    changed = set(
        (tracker["District Name"].apply(norm) + "||" + tracker["State Name"].apply(norm)).values
    )
    master["nfhs5_boundary_changed"] = master["_key"].isin(changed).astype(int)
    log.info(f"Boundary-changed flagged: {master['nfhs5_boundary_changed'].sum()}")

    # S3: PM-JAY
    pmjay = pd.read_excel(f"{raw_dir}/PMJAY_UHC_India_Data_Validated.xlsx",
                          sheet_name="PM_JAY_State", header=1)
    pmjay.columns = [str(c).strip() for c in pmjay.columns]
    pmjay["_sn"] = pmjay["State UT"].apply(norm)
    pmjay2 = pmjay.rename(columns={c: c.lower().replace(" ", "_") for c in pmjay.columns})
    pmjay2["_sn"] = pmjay["_sn"]
    master = master.merge(
        pmjay2.add_prefix("pmjay_").rename(columns={"pmjay__sn": "_sn_p"}),
        left_on="State_norm", right_on="_sn_p", how="left"
    ).drop(columns=["_sn_p"], errors="ignore")
    log.info(f"PM-JAY matched: {master['pmjay_treatment_flag'].notna().sum()}/707")

    # S7: NHA
    nha = pd.read_excel(f"{raw_dir}/NHA_District_DIU_Cards_Hospitals.xlsx",
                        sheet_name="Combined_State_Master", header=1)
    nha.columns = [str(c).strip() for c in nha.columns]
    nha["_sn"] = nha["State_UT"].apply(norm)
    rename_nha = {
        "DIU_Established (derived)": "diu_established",
        "Eligible_Families_Lakh": "ayushman_eligible_lakh",
        "Cards_Issued_Lakh": "ayushman_cards_lakh",
        "Card_Penetration_%": "ayushman_card_pct",
        "Hospitals_Total_2020": "hospitals_total_2020",
        "Hospitals_Public_2020": "hospitals_public_2020",
        "Hospitals_Private_2020": "hospitals_private_2020",
        "Hospitals_Est_2022": "hospitals_est_2022",
    }
    avail = {k: v for k, v in rename_nha.items() if k in nha.columns}
    nha2 = nha[["_sn"] + list(avail.keys())].rename(columns=avail)
    master = master.merge(nha2, left_on="State_norm", right_on="_sn", how="left").drop(
        columns=["_sn"], errors="ignore"
    )
    log.info(f"NHA matched: {master['diu_established'].notna().sum()}/707")

    # S6: GADM geography
    geo = compute_geo_features(f"{raw_dir}/gadm41_IND_shp.zip")
    state_fix_g = {
        "ANDAMAN AND NICOBAR": "ANDAMAN & NICOBAR ISLANDS",
        "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
        "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA & NAGAR HAVELI",
    }
    geo["_sg"] = geo["_key_g"].apply(lambda x: x.split("||")[1])
    geo["_sg"] = geo["_sg"].map(lambda x: state_fix_g.get(x, x))
    geo["_key_final"] = geo["_key_g"].apply(lambda x: x.split("||")[0]) + "||" + geo["_sg"]
    geo_cols = ["_key_final", "geo_area_km2", "geo_perimeter_km", "geo_compactness",
                "geo_centroid_lat", "geo_centroid_lon", "geo_dist_to_capital_km",
                "geo_tax_index", "geo_tax_quintile", "GID_2"]
    master = master.merge(
        geo[geo_cols].groupby("_key_final").first().reset_index(),
        left_on="_key", right_on="_key_final", how="left"
    ).drop(columns=["_key_final"], errors="ignore")
    # Fill geo_area_km2 from census where GADM unmatched
    mask = master["geo_area_km2"].isna() & master["census_area_sqkm"].notna()
    master.loc[mask, "geo_area_km2"] = master.loc[mask, "census_area_sqkm"]
    log.info(f"Geo matched: {master['geo_tax_index'].notna().sum()}/707")

    # Final cleanup
    master.replace("*", np.nan, inplace=True)
    master.drop(columns=["_key"], inplace=True, errors="ignore")
    master = master.drop_duplicates(subset=["District_norm", "State_norm"], keep="first")

    assert len(master) == 707, f"Expected 707 rows, got {len(master)}"
    master.to_csv(out_path, index=False)
    log.info(f"=== SAVED: {out_path} ({master.shape[0]} rows × {master.shape[1]} columns) ===")
    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw", help="Directory with raw source files")
    parser.add_argument("--out", default="data/master_final_v3.csv", help="Output path")
    args = parser.parse_args()
    build_master(args.raw_dir, args.out)

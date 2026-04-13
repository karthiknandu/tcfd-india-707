# Data Sources

All raw data files required by `src/data_pipeline/master_merge.py`.
Download and place in `data/raw/` before running the pipeline.

---

## S1: NFHS-5 District Factsheets (2019–21)

- **Source:** International Institute for Population Sciences (IIPS) / Ministry of Health and Family Welfare, Government of India
- **URL:** https://rchiips.org/nfhs/NFHS-5Reports/
- **Download:** State-wise district factsheets (Excel) → consolidate into `master_final_v2.csv`
- **Full 109-variable dataset:** SSRN working paper dataset (`ssrn_datasheet.xls`)
- **Coverage:** 707 districts, 36 States/UTs
- **Licence:** Government of India Open Data

---

## S2: Mukherji et al. 2024 — UHCd Index

- **Source:** WHO Bulletin supplementary repository (figshare)
- **DOI:** https://doi.org/10.6084/m9.figshare.25982521
- **Download:** figshare.com/articles/dataset/Supplementary_material_District-level_monitoring_of_universal_health_coverage_India_/25982521
- **File in repo:** `data/25982521.zip` (contains PDF Technical Report — UHCd extracted programmatically)
- **Coverage:** 687 of 707 districts (525 matched in this study)
- **Licence:** CC BY IGO 3.0 (World Health Organization)
- **Citation:** Mukherji A et al. Bull World Health Organ 2024;102:630–638B. doi:10.2471/BLT.23.290854

---

## S3: PM-JAY Parliamentary Q&A Data

- **Hospitalizations (2018–24):** Rajya Sabha RS 265 Unstarred Q1723, 6 August 2024
  - URL: https://data.gov.in/resource/stateut-wise-details-total-number-beneficiaries...
- **Funds allocated (2023–25):** Rajya Sabha RS 266 Starred Q88, 3 December 2024
  - URL: https://data.gov.in/resource/stateut-wise-details-funds-allocated...
- **File:** `PMJAY_UHC_India_Data_Validated.xlsx` (in this repo under `data/`)
- **Coverage:** 36 States/UTs (state-level)

---

## S4: IHME Global Burden of Disease 2023 — India Subnational

- **Source:** Institute for Health Metrics and Evaluation (IHME)
- **URL:** https://vizhub.healthdata.org/gbd-results
- **Download steps:**
  1. Go to GBD Results Tool
  2. Select: India → Sub-national | All causes | DALYs | Rate | Both | All ages | 2021
  3. Export CSV → `IHME-GBD_2023_DATA-9b13d1c4.zip`
- **Coverage:** 32 Indian states (not district-level)
- **Licence:** CC BY 4.0

---

## S5: Census 2011 — Primary Census Abstract (A-1 Series)

- **Source:** Office of the Registrar General of India
- **URL:** https://censusindia.gov.in/census.website/data/census-tables
- **Table:** A-1: Number of Villages, Towns, Households, Population and Area
- **File:** `A-1_Census_2011.xlsx` (place in `data/raw/`)
- **Coverage:** 640 of 707 districts (newer districts post-2011 not in Census)

---

## S6: GADM v4.1 — India District Shapefiles

- **Source:** Global Administrative Areas
- **URL:** https://gadm.org/download_country.html → India → Level 2
- **File:** `gadm41_IND_shp.zip` (place in `data/raw/`)
- **Coverage:** 676 polygons (Level 2 = districts)
- **CRS:** WGS84 (EPSG:4326); reprojected to UTM Zone 44N (EPSG:32644) for area/distance computation
- **Licence:** CC BY 4.0

---

## S7: NHA Annual Reports / Joseph et al. 2021

- **NHA source:** https://nha.gov.in — Resources → Annual Reports
- **Hospital empanelment data (peer-reviewed):**
  - Joseph J, Sankar DH, Nambiar D. PLOS ONE 2021;16(5):e0251814
  - doi:10.1371/journal.pone.0251814
- **File:** `NHA_District_DIU_Cards_Hospitals.xlsx` (in this repo under `data/`)

---

## S8: NFHS Policy Tracker — Appendix A (130 Changed Districts)

- **Source:** Appendix A provided by Walsh College supervisor
- **File:** `NFHS_Policy_Tracker_A.xlsx` (place in `data/raw/`)
- **Purpose:** Flag 115 districts whose boundaries changed between NFHS-4 and NFHS-5

---

## S9 / S10: WHO UHC SCI (Transportability)

- **Source:** World Bank Data API
- **URL:** https://data.worldbank.org/indicator/SH.UHC.SRVS.CV.XD
- **Countries:** India (IND), Nigeria (NGA), Bangladesh (BGD), Kenya (KEN), Cambodia (KHM)
- **Also:** WHO Global Health Observatory — https://www.who.int/data/gho

---

*Last updated: April 2026*

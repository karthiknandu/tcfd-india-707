"""
district_normaliser.py
======================
Standardises district/state names for consistent joins across
all 10 data sources. Handles common spelling variants and
Census 2011 vs NFHS-5 boundary differences.
"""

import re
import pandas as pd
from typing import Optional

# Known district name corrections: source_name -> canonical (NFHS-5)
DISTRICT_CORRECTIONS = {
    "NORTH & MIDDLE ANDAMAN": "NORTH AND MIDDLE ANDAMAN",
    "NICOBAR ISLANDS": "NICOBAR",
    "BALESHWAR": "BALASORE",
    "JAJAPUR": "JAJPUR",
    "NABARANGAPUR": "NABARANGPUR",
    "BAUDH": "BOUDH",
    "DEOGARH": "DEBAGARH",
    "SUBARNAPUR": "SONEPUR",
    "GAJAPATI": "GAJAPATI",
    "Y.S.R.": "YSR",
    "SRI POTTI SRI RAMULU NELLORE": "NELLORE",
    "RANGAREDDY": "RANGA REDDY",
    "MAHBUBNAGAR": "MAHABUBNAGAR",
    "ADILABAD": "ADILABAD",
    "LEH(LADAKH)": "LEH",
    "KINNAUR": "KINNAUR",
    "LAHUL & SPITI": "LAHAUL AND SPITI",
    "BUDGAM": "BADGAM",
    "SHOPIAN": "SHUPIYAN",
    "BANDIPORE": "BANDIPORA",
    "BARAMULLA": "BARAMULA",
    "EAST SINGHBHUM": "PURBI SINGHBHUM",
    "WEST SINGHBHUM": "PASHCHIMI SINGHBHUM",
    "PURBA CHAMPARAN": "EAST CHAMPARAN",
    "PASCHIM CHAMPARAN": "WEST CHAMPARAN",
    "MUZAFFARPUR": "MUZAFFARPUR",
    "RAIGARH": "RAIGARH",
    "SURGUJA": "SARGUJA",
    "DHAMTARI": "DHAMTARI",
}

# State name corrections
STATE_CORRECTIONS = {
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN & NICOBAR ISLANDS",
    "ANDAMAN AND NICOBAR": "ANDAMAN & NICOBAR ISLANDS",
    "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
    "JAMMU & KASHMIR": "JAMMU & KASHMIR",
    "DADRA AND NAGAR HAVELI": "DADRA & NAGAR HAVELI",
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA & NAGAR HAVELI",
    "DAMAN AND DIU": "DADRA & NAGAR HAVELI",
    "UTTARANCHAL": "UTTARAKHAND",
    "ORISSA": "ODISHA",
    "NCT OF DELHI": "DELHI",
    "MAHARASTRA": "MAHARASHTRA",
}


def normalise_name(s: str) -> str:
    """Uppercase, strip, collapse spaces."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).upper().strip())


def normalise_district(district: str, state: Optional[str] = None) -> str:
    """Normalise district name with known corrections."""
    n = normalise_name(district)
    return DISTRICT_CORRECTIONS.get(n, n)


def normalise_state(state: str) -> str:
    """Normalise state name to canonical form."""
    n = normalise_name(state)
    return STATE_CORRECTIONS.get(n, n)


def make_join_key(district: str, state: str) -> str:
    """Create composite join key: DISTRICT_NORM||STATE_NORM"""
    return normalise_district(district) + "||" + normalise_state(state)


def normalise_dataframe(df: pd.DataFrame,
                        district_col: str = "District",
                        state_col: str = "State") -> pd.DataFrame:
    """Add _key column to dataframe for consistent joins."""
    df = df.copy()
    df["District_norm"] = df[district_col].apply(normalise_district)
    df["State_norm"] = df[state_col].apply(normalise_state)
    df["_key"] = df.apply(lambda r: make_join_key(r[district_col], r[state_col]), axis=1)
    return df


if __name__ == "__main__":
    # Quick test
    test_cases = [
        ("North & Middle Andaman", "Andaman and Nicobar Islands"),
        ("Leh(Ladakh)", "Jammu & Kashmir"),
        ("Y.S.R.", "Andhra Pradesh"),
    ]
    for d, s in test_cases:
        key = make_join_key(d, s)
        print(f"{d}, {s} -> {key}")

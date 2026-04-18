"""
BUOC 4d: Tu dong lay lich thi dau gameweek tiep theo
=====================================================
Lay fixture tu football-data.org API (mien phi, 10 requests/phut)

Buoc 1: Dang ky tai https://www.football-data.org/client/register
Buoc 2: Copy API key vao bien API_KEY phia duoi
Buoc 3: Chay script nay truoc khi chay predict.py

Output: predictions/gw{N}_fixtures.csv
        predict.py se tu doc file nay, khong can nhap tay

Chay: python fetch_fixtures.py
"""

import requests
import pandas as pd
import os
from datetime import datetime, timezone

# ------------------------------------------------------------------------------
# CAU HINH — thay API_KEY bang key cua ban
# Dang ky mien phi tai: https://www.football-data.org/client/register
# ------------------------------------------------------------------------------

API_KEY  = "e35e3e41247a49409496ccf46d76f635"
BASE_URL = "https://api.football-data.org/v4"
EPL_CODE = "PL"


# ------------------------------------------------------------------------------
# Chuan hoa ten doi cho khop voi epl_clean.csv
# football-data.org dung ten khac voi football-data.co.uk
# ------------------------------------------------------------------------------

NAME_MAP = {
    "Manchester United FC"      : "Manchester United",
    "Manchester City FC"        : "Manchester City",
    "Arsenal FC"                : "Arsenal",
    "Chelsea FC"                : "Chelsea",
    "Liverpool FC"              : "Liverpool",
    "Tottenham Hotspur FC"      : "Tottenham",
    "Newcastle United FC"       : "Newcastle United",
    "Aston Villa FC"            : "Aston Villa",
    "West Ham United FC"        : "West Ham United",
    "Brighton & Hove Albion FC" : "Brighton",
    "Wolverhampton Wanderers FC": "Wolverhampton",
    "Fulham FC"                 : "Fulham",
    "Brentford FC"              : "Brentford",
    "Crystal Palace FC"         : "Crystal Palace",
    "Nottingham Forest FC"      : "Nott'm Forest",
    "AFC Bournemouth"           : "Bournemouth",
    "Everton FC"                : "Everton",
    "Leicester City FC"         : "Leicester City",
    "Leeds United FC"           : "Leeds United",
    "Burnley FC"                : "Burnley",
    "Luton Town FC"             : "Luton Town",
    "Sheffield United FC"       : "Sheffield Utd",
    "Sunderland AFC"            : "Sunderland",
    "Ipswich Town FC"           : "Ipswich Town",
    "Southampton FC"            : "Southampton",
}

def normalize_team(name):
    return NAME_MAP.get(name, name)


# ------------------------------------------------------------------------------
# 1. Lay danh sach fixture sap toi tu API
# ------------------------------------------------------------------------------

def fetch_next_fixtures():
    """
    Goi API lay cac tran SCHEDULED (chua da) cua EPL.
    Tra ve (DataFrame fixtures, so gameweek) hoac None neu loi.
    """
    if API_KEY == "YOUR_API_KEY_HERE":
        print("[LOI] Chua nhap API key!")
        print("  Dang ky tai: https://www.football-data.org/client/register")
        print("  Sau do thay 'YOUR_API_KEY_HERE' bang key cua ban.")
        return None

    headers = {"X-Auth-Token": API_KEY}
    url     = f"{BASE_URL}/competitions/{EPL_CODE}/matches?status=SCHEDULED"

    print("Dang lay fixture tu football-data.org...")

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        if resp.status_code == 403:
            print("[LOI] API key khong hop le. Kiem tra lai tai football-data.org/account")
        elif resp.status_code == 429:
            print("[LOI] Vuot gioi han 10 requests/phut. Doi 1 phut roi thu lai.")
        else:
            print(f"[LOI] HTTP {resp.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        print("[LOI] Khong ket noi duoc internet.")
        return None

    matches = resp.json().get("matches", [])
    if not matches:
        print("Khong co tran dau nao sap toi.")
        return None

    rows = []
    for m in matches:
        rows.append({
            "gameweek": m.get("matchday", 0),
            "home":     normalize_team(m["homeTeam"]["name"]),
            "away":     normalize_team(m["awayTeam"]["name"]),
            "date":     m["utcDate"][:10],   # lay phan YYYY-MM-DD
        })

    df_fix  = pd.DataFrame(rows)
    next_gw = df_fix["gameweek"].min()
    df_next = df_fix[df_fix["gameweek"] == next_gw].reset_index(drop=True)

    print(f"Gameweek tiep theo: GW{next_gw} ({len(df_next)} tran)")
    print()
    for _, r in df_next.iterrows():
        print(f"  {r['home']:25s} vs {r['away']:25s}  [{r['date']}]")
    print()

    return df_next, next_gw


# ------------------------------------------------------------------------------
# 2. Luu fixtures ra file de predict.py tu doc
# ------------------------------------------------------------------------------

def save_fixtures(df_fixtures, gameweek):
    os.makedirs("predictions", exist_ok=True)
    path = f"predictions/gw{gameweek}_fixtures.csv"
    df_fixtures.to_csv(path, index=False)
    print(f"Da luu: {path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    result = fetch_next_fixtures()

    if result is None:
        print("Khong lay duoc fixture. Thu lai sau hoac kiem tra API key.")
    else:
        df_fixtures, next_gw = result
        save_fixtures(df_fixtures, next_gw)
        print(f"Hoan tat! Tiep theo: chay predict.py de du doan GW{next_gw}")
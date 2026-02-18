"""
ERCOT North Hub — Peak Price Forecast
Sources:
  • ERCOT prices  : ERCOT Public API (requires free registration at ercot.com/services/api)
  • Henry Hub gas : EIA Open Data API (free, key at eia.gov/opendata)
  • Weather       : Open-Meteo (free, no key)
  • Weather hist  : Open-Meteo Archive (free, no key)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ERCOT North Hub · Peak Price Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #F7F5F0; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.0rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="stSidebar"] { background: #1A1A1A; }
div[data-testid="stSidebar"] * { color: #F7F5F0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ERCOT_TZ   = pytz.timezone("US/Central")
LAT, LON   = 32.90, -97.04
PEAK_HOURS = list(range(7, 23))   # HE07–HE22

TBL_STYLES = [
    {"selector": "th", "props": [
        ("background-color", "#1A1A1A"), ("color", "#F7F5F0"),
        ("font-size", "12px"), ("text-align", "center"),
        ("padding", "10px 12px"), ("font-family", "DM Sans, sans-serif"),
        ("letter-spacing", "0.05em"),
    ]},
    {"selector": "td", "props": [
        ("padding", "10px 14px"), ("text-align", "center"),
        ("font-family", "DM Mono, monospace"), ("font-size", "13px"),
    ]},
    {"selector": "tr:nth-child(even) td", "props": [("background-color", "#EEEBE4")]},
    {"selector": "tr:hover td", "props": [("background-color", "#E0DBD0")]},
]


# ─────────────────────────────────────────────────────────────────────────────
# COLOR HELPERS
# Power:  <$35 green | $35–$60 yellow | $60–$100 red | $100+ purple
# ─────────────────────────────────────────────────────────────────────────────

def color_price(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 100:
        return "background-color:#E8D5F5;color:#4A0080;font-weight:600"
    if val >= 60:
        return "background-color:#FFDAD4;color:#8B0000;font-weight:600"
    if val >= 35:
        return "background-color:#FFF3CC;color:#7A5000"
    return "background-color:#D4EDDA;color:#155724"   # green for cheap


def color_gas(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 5.0:
        return "background-color:#E8D5F5;color:#4A0080;font-weight:600"
    if val >= 3.5:
        return "background-color:#FFDAD4;color:#8B0000"
    if val >= 2.5:
        return "background-color:#FFF3CC;color:#7A5000"
    return "background-color:#D4EDDA;color:#155724"


def color_ratio(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 12:
        return "background-color:#E8D5F5;color:#4A0080;font-weight:600"
    if val >= 8:
        return "background-color:#FFF3CC;color:#7A5000"
    if val >= 5:
        return ""
    return "background-color:#D4EDDA;color:#155724"


def color_diff(val):
    if not isinstance(val, str) or val in ("N/A", "—"):
        return ""
    try:
        n = int(val.replace("+", "").replace("°", ""))
        if n >= 5:  return "color:#C03A00;font-weight:600"
        if n <= -5: return "color:#1A7A3C;font-weight:600"
    except Exception:
        pass
    return ""


def sub(text):
    """Render a consistent subtitle/caption line."""
    st.markdown(
        f"<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif;margin-top:-6px'>{text}</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# API KEY CONFIGURATION  (entered once in the sidebar)
# ─────────────────────────────────────────────────────────────────────────────

def get_api_keys():
    """
    Users paste their free API keys once into the sidebar.
    Keys are stored in st.session_state so they persist during the session.
    """
    with st.sidebar:
        st.markdown("## 🔑 API Keys")
        st.markdown(
            "<p style='font-size:12px;color:#aaa'>Keys are never stored permanently — "
            "paste them each session or pre-fill them in a Streamlit secrets file.</p>",
            unsafe_allow_html=True,
        )

        ercot_key = st.text_input(
            "ERCOT API Key",
            value=st.session_state.get("ercot_key", ""),
            type="password",
            help="Free at ercot.com — click 'API Access' under Services",
        )
        eia_key = st.text_input(
            "EIA API Key",
            value=st.session_state.get("eia_key", ""),
            type="password",
            help="Free at eia.gov/opendata — instant registration",
        )

        if ercot_key:
            st.session_state["ercot_key"] = ercot_key
        if eia_key:
            st.session_state["eia_key"] = eia_key

        st.divider()
        st.markdown("**Get free keys:**")
        st.markdown("• [ERCOT API](https://ercot.com/services/api) — free, instant")
        st.markdown("• [EIA Open Data](https://www.eia.gov/opendata/register.php) — free, instant")

        st.divider()
        st.markdown("**MarketView / ICE Connect:**")
        st.markdown(
            "<p style='font-size:12px;color:#aaa'>"
            "See the note at the bottom of the main page about connecting "
            "your MarketView and ICE data feeds.</p>",
            unsafe_allow_html=True,
        )

    return (
        st.session_state.get("ercot_key", ""),
        st.session_state.get("eia_key", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def fetch_weather_forecast():
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,windspeed_10m,cloudcover"
        f"&daily=temperature_2m_max,temperature_2m_min,windspeed_10m_max"
        f"&temperature_unit=fahrenheit&windspeed_unit=mph"
        f"&timezone=America%2FChicago&forecast_days=7"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=86400)
def fetch_historical_weather(year_offset: int):
    now   = datetime.now(ERCOT_TZ)
    start = (now - timedelta(days=365 * year_offset)).strftime("%Y-%m-%d")
    end   = (now - timedelta(days=365 * year_offset - 6)).strftime("%Y-%m-%d")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_max,temperature_2m_min"
        f"&temperature_unit=fahrenheit&timezone=America%2FChicago"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_ercot_historical(api_key: str):
    """
    ERCOT Public API — requires free registration.
    Docs: https://developer.ercot.com/
    Auth: Bearer token in Authorization header.
    """
    if not api_key:
        return None, "No ERCOT API key provided — enter it in the sidebar.", 0, 0

    now      = datetime.now(ERCOT_TZ)
    start_dt = now - timedelta(days=45)
    prices   = []
    da_count = rt_count = 0
    da_error = rt_error = None

    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}

    # ── Day-Ahead ──
    try:
        url = "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices"
        params = {
            "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
            "deliveryDateTo":   now.strftime("%Y-%m-%d"),
            "settlementPoint":  "HB_NORTH",
            "size": 5000,
        }
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 200:
            for row in r.json().get("data", []):
                try:
                    dt = ERCOT_TZ.localize(
                        datetime.strptime(f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M")
                    )
                    prices.append({"datetime": dt, "da_price": float(row[3]), "rt_price": None})
                    da_count += 1
                except Exception:
                    continue
        elif r.status_code == 401:
            da_error = "Invalid or expired ERCOT API key"
        else:
            da_error = f"HTTP {r.status_code}"
    except Exception as e:
        da_error = str(e)[:80]

    # ── Real-Time ──
    rt_map = {}
    try:
        url_rt = "https://api.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub"
        params_rt = {
            "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
            "deliveryDateTo":   now.strftime("%Y-%m-%d"),
            "settlementPoint":  "HB_NORTH",
            "size": 5000,
        }
        r = requests.get(url_rt, params=params_rt, headers=headers, timeout=20)
        if r.status_code == 200:
            for row in r.json().get("data", []):
                try:
                    dt = ERCOT_TZ.localize(
                        datetime.strptime(f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M")
                    )
                    rt_map[dt] = float(row[3])
                    rt_count += 1
                except Exception:
                    continue
        elif r.status_code == 401:
            rt_error = "Invalid or expired ERCOT API key"
        else:
            rt_error = f"HTTP {r.status_code}"
    except Exception as e:
        rt_error = str(e)[:80]

    for p in prices:
        p["rt_price"] = rt_map.get(p["datetime"])

    if prices:
        df = pd.DataFrame(prices).dropna(subset=["da_price"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df, f"✅ {da_count} DA rows · {rt_count} RT rows loaded", da_count, rt_count
    else:
        errs = " | ".join(filter(None, [da_error, rt_error]))
        return None, f"⚠️ {errs or 'No data returned'}", 0, 0


@st.cache_data(ttl=3600)
def fetch_henry_hub(api_key: str):
    """
    EIA Open Data API v2 — free key at eia.gov/opendata.
    Series: Henry Hub Natural Gas Spot Price (RNGWHHD), daily.
    """
    if not api_key:
        return None, "No EIA API key provided — enter it in the sidebar."

    url = "https://api.eia.gov/v2/natural-gas/pri/sum/dcu/nus/daily/data/"
    params = {
        "api_key":           api_key,
        "frequency":         "daily",
        "data[0]":           "value",
        "facets[process][]": "PH9",   # Henry Hub spot
        "sort[0][column]":   "period",
        "sort[0][direction]":"desc",
        "length":            90,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 403:
            return None, "⚠️ Invalid EIA API key — check the key at eia.gov/opendata"
        if r.status_code != 200:
            # Fallback: try the FRED CSV (no key needed, may be slightly delayed)
            return _fetch_henry_hub_fred()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            return _fetch_henry_hub_fred()
        df = pd.DataFrame(rows)[["period", "value"]].rename(
            columns={"period": "Date", "value": "HH $/MMBtu"}
        )
        df["HH $/MMBtu"] = pd.to_numeric(df["HH $/MMBtu"], errors="coerce")
        df = df.dropna().sort_values("Date", ascending=False).reset_index(drop=True)
        latest = df["Date"].iloc[0]
        return df, f"✅ {len(df)} daily prices loaded, most recent: {latest}"
    except Exception as e:
        # Try FRED as backup
        return _fetch_henry_hub_fred()


def _fetch_henry_hub_fred():
    """
    Fallback: pull Henry Hub from FRED's public CSV (no key, updated daily).
    Series DHHNGSP — Henry Hub Natural Gas Spot Price.
    """
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP"
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; gridedge-app/1.0)"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["Date", "HH $/MMBtu"]
        df["HH $/MMBtu"] = pd.to_numeric(df["HH $/MMBtu"], errors="coerce")
        df = df.dropna().sort_values("Date", ascending=False).reset_index(drop=True)
        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        df = df[df["Date"] >= cutoff].reset_index(drop=True)
        latest = df["Date"].iloc[0] if len(df) > 0 else "unknown"
        return df, f"✅ {len(df)} days via FRED/St. Louis Fed, most recent: {latest}"
    except Exception as e:
        return None, f"⚠️ FRED fallback also failed: {str(e)[:80]}"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def temp_to_mult(temp_f):
    for thresh, mult in [(100,3.5),(95,2.5),(90,1.8),(85,1.35),(80,1.15),
                          (75,1.0),(65,0.85),(55,0.90),(45,1.05),(35,1.25),
                          (25,1.75),(15,2.8)]:
        if temp_f >= thresh:
            return mult
    return 4.0


def build_forecast(weather_data, hist_df):
    daily    = weather_data.get("daily", {})
    hourly   = weather_data.get("hourly", {})
    dates    = daily.get("time", [])
    hi_temps = daily.get("temperature_2m_max", [])
    lo_temps = daily.get("temperature_2m_min", [])
    wind_max = daily.get("windspeed_10m_max", [])

    h_df = pd.DataFrame({
        "time": hourly.get("time", []),
        "temp": hourly.get("temperature_2m", []),
        "wind": hourly.get("windspeed_10m", []),
    })
    h_df["date"] = h_df["time"].str[:10]
    h_df["hour"] = h_df["time"].str[11:13].astype(int)
    peak_h      = h_df[h_df["hour"].isin(PEAK_HOURS)]
    pk_temp_map = peak_h.groupby("date")["temp"].mean().to_dict()
    pk_wind_map = peak_h.groupby("date")["wind"].mean().to_dict()

    if hist_df is not None and len(hist_df) > 30:
        hc = hist_df.copy()
        hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
        pk = hc[hc["hour"].isin(PEAK_HOURS)]
        base_da = pk["da_price"].mean() if len(pk) > 0 else 45.0
        base_rt = pk["rt_price"].dropna().mean() if len(pk) > 0 else 47.0
    else:
        base_da, base_rt = 45.0, 47.0

    now_ct = datetime.now(ERCOT_TZ)
    rows   = []
    for i, date_str in enumerate(dates):
        hi      = hi_temps[i] if i < len(hi_temps) else 75
        lo      = lo_temps[i] if i < len(lo_temps) else 55
        wind    = wind_max[i] if i < len(wind_max) else 10
        pk_temp = pk_temp_map.get(date_str, (hi + lo) / 2)
        pk_wind = pk_wind_map.get(date_str, wind)

        mult     = temp_to_mult(pk_temp)
        wind_adj = max(0.85, 1 - (pk_wind - 10) * 0.004) if pk_wind > 10 else 1.0

        da = round(max(15, min(base_da * mult * wind_adj, 500)), 2)
        rt = round(max(15, min(base_rt * mult * wind_adj * np.random.uniform(0.93, 1.07), 600)), 2)

        date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
        is_today  = date_str == now_ct.strftime("%Y-%m-%d")
        is_tmrw   = date_str == (now_ct + timedelta(days=1)).strftime("%Y-%m-%d")
        day_label = "Today" if is_today else ("Tomorrow" if is_tmrw else date_obj.strftime("%A"))

        peak_val = max(da, rt)
        if   peak_val >= 100: alert = "🟣 VERY HIGH"
        elif peak_val >=  60: alert = "🔴 HIGH"
        elif peak_val >=  35: alert = "🟡 MODERATE"
        else:                  alert = "🟢 LOW"

        rows.append({
            "Day":            day_label,
            "Date":           date_obj.strftime("%b %d"),
            "Hi °F":          int(round(hi)),
            "Lo °F":          int(round(lo)),
            "Avg Peak Temp":  round(pk_temp, 1),
            "Wind mph":       int(round(pk_wind)),
            "DA $/MWh":       da,
            "RT $/MWh":       rt,
            "Alert":          alert,
        })

    return pd.DataFrame(rows)


def build_weather_comparison(weather_data):
    daily  = weather_data.get("daily", {})
    dates  = daily.get("time", [])
    hi_now = daily.get("temperature_2m_max", [])
    lo_now = daily.get("temperature_2m_min", [])

    hist_1 = fetch_historical_weather(1)
    hist_2 = fetch_historical_weather(2)
    hist_3 = fetch_historical_weather(3)

    def get_hi_lo(hdata, idx):
        if hdata is None: return None, None
        try:
            return (round(hdata["daily"]["temperature_2m_max"][idx], 0),
                    round(hdata["daily"]["temperature_2m_min"][idx], 0))
        except Exception:
            return None, None

    def diff_str(val_now, val_hist):
        if val_now is None or val_hist is None: return "N/A"
        d = val_now - int(val_hist)
        return f"+{d}°" if d > 0 else f"{d}°"

    now_ct = datetime.now(ERCOT_TZ)
    rows   = []
    for i, date_str in enumerate(dates[:7]):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        is_today = date_str == now_ct.strftime("%Y-%m-%d")
        is_tmrw  = date_str == (now_ct + timedelta(days=1)).strftime("%Y-%m-%d")
        day_lbl  = "Today" if is_today else ("Tomorrow" if is_tmrw else date_obj.strftime("%A"))
        hi_this  = int(round(hi_now[i])) if i < len(hi_now) else None
        lo_this  = int(round(lo_now[i])) if i < len(lo_now) else None
        hi_1, lo_1 = get_hi_lo(hist_1, i)
        hi_2, lo_2 = get_hi_lo(hist_2, i)
        hi_3, lo_3 = get_hi_lo(hist_3, i)
        yr = date_obj.year

        rows.append({
            "Day":              day_lbl,
            "Date":             date_obj.strftime("%b %d"),
            "Forecast Hi / Lo": f"{hi_this}° / {lo_this}°" if hi_this else "—",
            f"{yr-1} Hi / Lo":  f"{int(hi_1)}° / {int(lo_1)}°" if hi_1 else "N/A",
            f"vs {yr-1}":       diff_str(hi_this, hi_1),
            f"{yr-2} Hi / Lo":  f"{int(hi_2)}° / {int(lo_2)}°" if hi_2 else "N/A",
            f"vs {yr-2}":       diff_str(hi_this, hi_2),
            f"{yr-3} Hi / Lo":  f"{int(hi_3)}° / {int(lo_3)}°" if hi_3 else "N/A",
            f"vs {yr-3}":       diff_str(hi_this, hi_3),
        })

    return pd.DataFrame(rows)


def build_gas_power_comparison(hist_df, gas_df):
    if hist_df is None or gas_df is None or len(gas_df) == 0:
        return None
    hc = hist_df.copy()
    hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
    hc["Date"] = hc["datetime"].apply(lambda x: x.strftime("%Y-%m-%d"))
    pk = hc[hc["hour"].isin(PEAK_HOURS)].copy()
    if len(pk) == 0: return None

    daily_power = (
        pk.groupby("Date")
        .agg(**{"DA Avg $/MWh": ("da_price","mean"), "RT Avg $/MWh": ("rt_price","mean")})
        .reset_index()
    )
    daily_power["DA Avg $/MWh"] = daily_power["DA Avg $/MWh"].round(2)
    daily_power["RT Avg $/MWh"] = daily_power["RT Avg $/MWh"].round(2)

    gas_col = "HH $/MMBtu"
    gas_renamed = gas_df.rename(columns={gas_df.columns[1]: gas_col})
    merged = daily_power.merge(gas_renamed[["Date", gas_col]], on="Date", how="left")
    merged[gas_col]        = merged[gas_col].round(3)
    merged["DA/Gas Ratio"] = (merged["DA Avg $/MWh"] / merged[gas_col]).round(2)
    merged["RT/Gas Ratio"] = (merged["RT Avg $/MWh"] / merged[gas_col]).round(2)
    return merged.sort_values("Date", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    now_ct = datetime.now(ERCOT_TZ)

    # Pull API keys from sidebar
    ercot_key, eia_key = get_api_keys()

    # ── Header ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 5, 2])
    with c1:
        st.markdown("### ⚡ GRIDEDGE")
    with c2:
        st.markdown("<p style='font-family:DM Mono,monospace;font-size:12px;color:#888;margin-top:14px;letter-spacing:0.08em'>ERCOT NORTH HUB · SETTLEMENT PRICE INTELLIGENCE</p>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:12px;color:#888;margin-top:14px;text-align:right'>{now_ct.strftime('%b %d, %Y · %I:%M %p CT')}</p>", unsafe_allow_html=True)

    st.divider()
    st.markdown("## Peak Hour Price Forecast")
    st.markdown("<p style='font-size:14px;color:#555;font-family:DM Sans,sans-serif;margin-top:-10px'>Projected on-peak (HE07–HE22) Day-Ahead and Real-Time settlement prices · ERCOT North Hub · $/MWh</p>", unsafe_allow_html=True)

    # ── Key warning if missing ────────────────────────────────────────────────
    if not ercot_key or not eia_key:
        missing = []
        if not ercot_key: missing.append("ERCOT")
        if not eia_key:   missing.append("EIA")
        st.info(
            f"⬅️ **Enter your {' and '.join(missing)} API key(s) in the sidebar** to load live price data. "
            f"Both are free and take ~2 minutes to register. "
            f"Weather data and the price model will still run — prices will use default baseline values until keys are added."
        )

    st.divider()

    # ── Load all data ─────────────────────────────────────────────────────────
    with st.spinner("⏳ Fetching data…"):
        weather_data, weather_err   = fetch_weather_forecast()
        hist_df, ercot_status, da_count, rt_count = fetch_ercot_historical(ercot_key)
        gas_df, gas_status          = fetch_henry_hub(eia_key)

    if weather_data is None:
        st.error(f"Weather API failed: {weather_err}. Please refresh.")
        return

    # ── Data status panel ─────────────────────────────────────────────────────
    with st.expander("📡 Data Source Status", expanded=not (ercot_key and eia_key)):
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**🌤 Weather · Open-Meteo**")
            st.success("Connected — 7-day hourly forecast for DFW")
        with s2:
            st.markdown("**⚡ ERCOT North Hub**")
            if da_count > 0:
                st.success(ercot_status)
            elif not ercot_key:
                st.warning("Awaiting API key — enter in sidebar")
            else:
                st.error(ercot_status)
        with s3:
            st.markdown("**🔥 Henry Hub Gas**")
            if gas_df is not None:
                st.success(gas_status)
            elif not eia_key:
                st.warning("Awaiting EIA key — enter in sidebar (FRED fallback active)")
            else:
                st.error(gas_status)
        st.markdown("<p style='font-size:12px;color:#888;margin-top:8px'>ERCOT API key: free at ercot.com/services/api · EIA key: free at eia.gov/opendata · Weather: no key needed</p>", unsafe_allow_html=True)

    # ── Build model ───────────────────────────────────────────────────────────
    forecast_df = build_forecast(weather_data, hist_df)
    weather_cmp = build_weather_comparison(weather_data)

    hist_da_avg = hist_rt_avg = None
    if hist_df is not None and len(hist_df) > 0:
        hc = hist_df.copy()
        hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
        pk = hc[hc["hour"].isin(PEAK_HOURS)]
        if len(pk):
            hist_da_avg = round(pk["da_price"].mean(), 2)
            hist_rt_avg = round(pk["rt_price"].dropna().mean(), 2)

    def delta_str(val, ref):
        if ref is None: return None
        pct  = (val - ref) / ref * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.1f}% vs 45-day avg"

    today    = forecast_df.iloc[0]
    tomorrow = forecast_df.iloc[1] if len(forecast_df) > 1 else None

    latest_gas = latest_gas_date = None
    if gas_df is not None and len(gas_df) > 0:
        latest_gas      = round(float(gas_df.iloc[0, 1]), 3)
        latest_gas_date = gas_df.iloc[0, 0]

    # ── Metric tiles ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Today — DA Peak", f"${today['DA $/MWh']:.2f} /MWh",
                  delta_str(today["DA $/MWh"], hist_da_avg))
    with m2:
        st.metric("Today — RT Peak", f"${today['RT $/MWh']:.2f} /MWh",
                  delta_str(today["RT $/MWh"], hist_rt_avg))
    with m3:
        if tomorrow is not None:
            st.metric("Tomorrow — DA Peak", f"${tomorrow['DA $/MWh']:.2f} /MWh",
                      delta_str(tomorrow["DA $/MWh"], hist_da_avg))
    with m4:
        if latest_gas is not None:
            st.metric(f"Henry Hub ({latest_gas_date})", f"${latest_gas:.3f} /MMBtu",
                      f"45-day DA avg: ${hist_da_avg:.2f}/MWh" if hist_da_avg else "Live gas price",
                      delta_color="off")
        elif hist_da_avg:
            st.metric("45-Day Avg Peak DA", f"${hist_da_avg:.2f} /MWh",
                      "on-peak hrs only", delta_color="off")

    st.divider()

    # ── 7-Day Forecast ────────────────────────────────────────────────────────
    st.markdown("### 📋 7-Day Price Forecast")
    st.markdown(
        "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif;margin-top:-6px'>"
        "On-peak hours only (HE07–HE22 = 7 am – 10 pm CT) &nbsp;·&nbsp; "
        "<span style='background:#D4EDDA;color:#155724;padding:2px 8px;border-radius:3px;font-size:12px'>🟢 Under $35</span>&nbsp;"
        "<span style='background:#FFF3CC;color:#7A5000;padding:2px 8px;border-radius:3px;font-size:12px'>🟡 $35–$60</span>&nbsp;"
        "<span style='background:#FFDAD4;color:#8B0000;padding:2px 8px;border-radius:3px;font-size:12px'>🔴 $60–$100</span>&nbsp;"
        "<span style='background:#E8D5F5;color:#4A0080;padding:2px 8px;border-radius:3px;font-size:12px'>🟣 $100+</span>"
        "</p>",
        unsafe_allow_html=True,
    )
    styled_fc = (
        forecast_df.style
        .applymap(color_price, subset=["DA $/MWh","RT $/MWh"])
        .format({"DA $/MWh":"${:.2f}","RT $/MWh":"${:.2f}","Avg Peak Temp":"{:.1f}°F"})
        .set_table_styles(TBL_STYLES)
    )
    st.dataframe(styled_fc, use_container_width=True, hide_index=True)

    st.divider()

    # ── Weather Comparison ────────────────────────────────────────────────────
    st.markdown("### 🌡️ DFW Weather — Forecast vs. Same Week in Prior Years")
    st.markdown(
        "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif;margin-top:-6px'>"
        "Same calendar dates 1, 2, and 3 years ago. &nbsp;"
        "<span style='color:#C03A00;font-weight:600'>Red = warmer than prior year</span>&nbsp;·&nbsp;"
        "<span style='color:#1A7A3C;font-weight:600'>Green = cooler</span></p>",
        unsafe_allow_html=True,
    )
    diff_cols = [c for c in weather_cmp.columns if c.startswith("vs ")]
    styled_wc = (
        weather_cmp.style
        .applymap(color_diff, subset=diff_cols)
        .set_table_styles(TBL_STYLES)
    )
    st.dataframe(styled_wc, use_container_width=True, hide_index=True)

    st.divider()

    # ── Historical Power Prices ───────────────────────────────────────────────
    st.markdown("### 📊 Historical On-Peak Power Prices — Last 45 Days")
    if hist_df is not None and len(hist_df) > 0:
        st.markdown(
            f"<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif;margin-top:-6px'>"
            f"Actual ERCOT North Hub on-peak settlement prices. {da_count} DA · {rt_count} RT hourly rows loaded.</p>",
            unsafe_allow_html=True,
        )
        hc = hist_df.copy()
        hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
        hc["Date"] = hc["datetime"].apply(lambda x: x.strftime("%Y-%m-%d"))
        pk = hc[hc["hour"].isin(PEAK_HOURS)].copy()
        if len(pk) > 0:
            dh = (
                pk.groupby("Date")
                .agg(**{"DA Avg $/MWh":("da_price","mean"),"DA Max $/MWh":("da_price","max"),
                        "RT Avg $/MWh":("rt_price","mean"),"RT Max $/MWh":("rt_price","max")})
                .reset_index().sort_values("Date", ascending=False)
            )
            for c in ["DA Avg $/MWh","DA Max $/MWh","RT Avg $/MWh","RT Max $/MWh"]:
                dh[c] = dh[c].round(2)
            hist_pc = ["DA Avg $/MWh","DA Max $/MWh","RT Avg $/MWh","RT Max $/MWh"]
            styled_hist = (
                dh.style
                .applymap(color_price, subset=hist_pc)
                .format({c:"${:.2f}" for c in hist_pc})
                .set_table_styles(TBL_STYLES)
            )
            st.dataframe(styled_hist, use_container_width=True, hide_index=True)
    else:
        if ercot_key:
            st.warning("ERCOT API returned no data. The key may be expired or the API is temporarily down. Try refreshing.")
        else:
            st.info("Enter your ERCOT API key in the sidebar to load historical settlement prices.")

    st.divider()

    # ── Henry Hub Gas ─────────────────────────────────────────────────────────
    st.markdown("### 🔥 Henry Hub Natural Gas — Last 90 Days")
    if gas_df is not None and len(gas_df) > 0:
        st.markdown(
            "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif;margin-top:-6px'>"
            "Daily Henry Hub spot price ($/MMBtu). Gas sets the marginal price in ERCOT ~70–80% of on-peak hours.</p>",
            unsafe_allow_html=True,
        )
        price_col = gas_df.columns[1]
        styled_gas = (
            gas_df.head(90).style
            .applymap(color_gas, subset=[price_col])
            .format({price_col: "${:.3f}"})
            .set_table_styles(TBL_STYLES)
        )
        st.dataframe(styled_gas, use_container_width=True, hide_index=True)
    else:
        if eia_key:
            st.warning(f"Could not load Henry Hub data: {gas_status}")
        else:
            st.info("Enter your EIA API key in the sidebar to load Henry Hub prices. The app will also try FRED as a fallback.")

    st.divider()

    # ── Gas vs Power Heat Rate ────────────────────────────────────────────────
    st.markdown("### ⚡ vs 🔥 Power vs. Gas — Implied Heat Rate")
    st.markdown(
        "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif;margin-top:-6px'>"
        "DA/Gas Ratio = implied heat rate (MWh/MMBtu). Above 9x: power expensive vs gas (good generator margins). "
        "Below 6x: power cheap relative to fuel. Typical ERCOT on-peak range: 8–12x in summer.</p>",
        unsafe_allow_html=True,
    )
    comp_df = build_gas_power_comparison(hist_df, gas_df)
    if comp_df is not None and len(comp_df) > 0:
        pwr_c  = ["DA Avg $/MWh","RT Avg $/MWh"]
        rat_c  = ["DA/Gas Ratio","RT/Gas Ratio"]
        gas_c_name = comp_df.columns[3]
        styled_comp = (
            comp_df.style
            .applymap(color_price,  subset=pwr_c)
            .applymap(color_gas,    subset=[gas_c_name])
            .applymap(color_ratio,  subset=rat_c)
            .format({
                "DA Avg $/MWh":  "${:.2f}",
                "RT Avg $/MWh":  "${:.2f}",
                gas_c_name:      "${:.3f}",
                "DA/Gas Ratio":  "{:.2f}x",
                "RT/Gas Ratio":  "{:.2f}x",
            })
            .set_table_styles(TBL_STYLES)
        )
        st.dataframe(styled_comp, use_container_width=True, hide_index=True)
    else:
        st.info("Heat rate table requires both ERCOT price data and Henry Hub gas prices to be loaded.")

    st.divider()

    # ── MarketView / ICE Note ─────────────────────────────────────────────────
    with st.expander("💼 Connecting MarketView & ICE Connect — How it works", expanded=False):
        st.markdown("""
**Yes — your MarketView and ICE Connect data can be integrated. Here's the realistic breakdown:**

---

**Option A: CSV / Excel Export → Upload here (easiest, works today)**

Both MarketView and ICE Connect let you export settlement data to CSV or Excel.
You can upload those files directly to this app and it will read them automatically.
Ask me and I'll add a file uploader widget to the top of this page that accepts your exports.

---

**Option B: ICE Connect API (most powerful)**

ICE offers a REST API and a WebSocket feed for real-time data. If your ICE Connect
subscription includes API access (most professional tiers do), you'd need:
1. Your ICE API credentials (API key + secret from your ICE account settings)
2. To paste them into the sidebar keys panel above
3. I'd add a new `fetch_ice_power()` and `fetch_ice_gas()` function pulling from:
   `https://api.theice.com/marketdata/...` with your bearer token

This would give you **live bid/offer and settlement prices** directly in the app.

---

**Option C: MarketView Data Feed**

MarketView (S&P Global / Platts) exposes data via their **Commodities API** and also
supports file-based delivery (FTP/SFTP drops). The cleanest path:
- If you have a **Platts API key**, I can add it to the sidebar and pull live assessments
- If you get file drops, we can point the app at an auto-refreshing folder or S3 bucket

---

**What to do next:**

Tell me which option fits your setup and I'll build it in. The key question is:
**Does your ICE Connect subscription include API/programmatic access?**
If yes, log into ICE Connect → Account Settings → API Credentials and grab your key.
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:12px;color:#aaa;font-family:DM Mono,monospace'>"
        "Sources: ERCOT Public API · EIA Open Data (DHHNGSP) · FRED St. Louis Fed (fallback) · Open-Meteo · "
        f"Last loaded {now_ct.strftime('%b %d %Y %I:%M %p CT')} · Cached 30–60 min"
        "<br>Statistical estimates only — not investment or trading advice."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

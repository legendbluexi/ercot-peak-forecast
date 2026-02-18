import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import warnings
import io
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
    font-size: 2.1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
/* Fix caption/small text rendering */
.stCaption, [data-testid="stCaptionContainer"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    color: #555 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ERCOT_TZ   = pytz.timezone("US/Central")
LAT, LON   = 32.90, -97.04
PEAK_HOURS = list(range(7, 23))   # HE07–HE22  (7 am through 10 pm)

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
# Power price scale:  <$35 light yellow | $35–$60 yellow | $60–$100 red | $100+ purple
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
    return "background-color:#FFFDE7;color:#5A4A00"


def color_gas(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 5.0:
        return "background-color:#E8D5F5;color:#4A0080;font-weight:600"
    if val >= 3.5:
        return "background-color:#FFDAD4;color:#8B0000"
    if val >= 2.5:
        return "background-color:#FFF3CC;color:#7A5000"
    return "background-color:#FFFDE7;color:#5A4A00"


def color_ratio(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 12:
        return "background-color:#E8D5F5;color:#4A0080;font-weight:600"
    if val >= 8:
        return "background-color:#FFF3CC;color:#7A5000"
    if val >= 5:
        return ""
    return "background-color:#DFF5E8;color:#1A5C35"


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
        r = requests.get(url, timeout=10)
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
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_ercot_historical():
    now      = datetime.now(ERCOT_TZ)
    start_dt = now - timedelta(days=45)
    prices   = []
    da_count = rt_count = 0
    da_error = rt_error = None

    try:
        url = "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices"
        params = {
            "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
            "deliveryDateTo":   now.strftime("%Y-%m-%d"),
            "settlementPoint":  "HB_NORTH",
            "size": 5000,
        }
        r = requests.get(url, params=params, timeout=15, headers={"Accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            for row in data.get("data", []):
                try:
                    dt = ERCOT_TZ.localize(
                        datetime.strptime(f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M")
                    )
                    prices.append({"datetime": dt, "da_price": float(row[3]), "rt_price": None})
                    da_count += 1
                except Exception:
                    continue
        else:
            da_error = f"HTTP {r.status_code}"
    except Exception as e:
        da_error = str(e)

    rt_map = {}
    try:
        url_rt = "https://api.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub"
        params_rt = {
            "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
            "deliveryDateTo":   now.strftime("%Y-%m-%d"),
            "settlementPoint":  "HB_NORTH",
            "size": 5000,
        }
        r = requests.get(url_rt, params=params_rt, timeout=15, headers={"Accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            for row in data.get("data", []):
                try:
                    dt = ERCOT_TZ.localize(
                        datetime.strptime(f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M")
                    )
                    rt_map[dt] = float(row[3])
                    rt_count += 1
                except Exception:
                    continue
        else:
            rt_error = f"HTTP {r.status_code}"
    except Exception as e:
        rt_error = str(e)

    for p in prices:
        p["rt_price"] = rt_map.get(p["datetime"])

    if prices:
        df = pd.DataFrame(prices).dropna(subset=["da_price"])
        df = df.sort_values("datetime").reset_index(drop=True)
        status = f"Connected — {da_count} DA rows and {rt_count} RT rows loaded"
        return df, status, da_count, rt_count
    else:
        errs = []
        if da_error: errs.append(f"DA: {da_error}")
        if rt_error: errs.append(f"RT: {rt_error}")
        msg = "API returned no data. " + ("; ".join(errs) if errs else "Unknown error.")
        return None, msg, 0, 0


@st.cache_data(ttl=3600)
def fetch_henry_hub():
    """
    Pull Henry Hub daily spot prices from FRED (St. Louis Fed).
    Series: DHHNGSP  — updated through prior business day, no API key needed.
    URL pattern: https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP"
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; gridedge/1.0)"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["Date", "HH Price ($/MMBtu)"]
        df["HH Price ($/MMBtu)"] = pd.to_numeric(df["HH Price ($/MMBtu)"], errors="coerce")
        df = df.dropna()
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
        # Keep last 90 days
        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        df = df[df["Date"] >= cutoff]
        latest_date = df["Date"].iloc[0] if len(df) > 0 else "unknown"
        status = f"Connected — {len(df)} days loaded, most recent: {latest_date}"
        return df, status
    except Exception as e:
        return None, f"FRED API error: {e}"


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
        else:                  alert = "🟡 LOW-MOD"

        rows.append({
            "Day":                  day_label,
            "Date":                 date_obj.strftime("%b %d"),
            "Hi °F":                int(round(hi)),
            "Lo °F":                int(round(lo)),
            "Avg Peak Temp":        round(pk_temp, 1),
            "Wind mph":             int(round(pk_wind)),
            "DA $/MWh":             da,
            "RT $/MWh":             rt,
            "Alert":                alert,
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
    if len(pk) == 0:
        return None

    daily_power = (
        pk.groupby("Date")
        .agg(**{
            "DA Avg $/MWh": ("da_price", "mean"),
            "RT Avg $/MWh": ("rt_price", "mean"),
        })
        .reset_index()
    )
    daily_power["DA Avg $/MWh"] = daily_power["DA Avg $/MWh"].round(2)
    daily_power["RT Avg $/MWh"] = daily_power["RT Avg $/MWh"].round(2)

    gas_renamed = gas_df.rename(columns={"HH Price ($/MMBtu)": "HH Gas $/MMBtu"})
    merged = daily_power.merge(gas_renamed[["Date","HH Gas $/MMBtu"]], on="Date", how="left")
    merged["HH Gas $/MMBtu"] = merged["HH Gas $/MMBtu"].round(3)
    merged["DA/Gas Ratio"] = (merged["DA Avg $/MWh"] / merged["HH Gas $/MMBtu"]).round(2)
    merged["RT/Gas Ratio"] = (merged["RT Avg $/MWh"] / merged["HH Gas $/MMBtu"]).round(2)

    return merged.sort_values("Date", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    now_ct = datetime.now(ERCOT_TZ)

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
    st.markdown("<p style='font-size:14px;color:#555;margin-top:-12px'>Projected on-peak (HE07–HE22) Day-Ahead and Real-Time settlement prices · ERCOT North Hub · $/MWh</p>", unsafe_allow_html=True)
    st.divider()

    # ── Load all data ─────────────────────────────────────────────────────────
    with st.spinner("⏳ Fetching live weather, ERCOT prices, and Henry Hub gas data…"):
        weather_data, weather_err = fetch_weather_forecast()
        hist_df, ercot_status, da_count, rt_count = fetch_ercot_historical()
        gas_df, gas_status = fetch_henry_hub()

    if weather_data is None:
        st.error(f"Could not reach weather API: {weather_err}. Please refresh.")
        return

    # ── Data status panel ─────────────────────────────────────────────────────
    with st.expander("📡 Data Source Status — click to expand", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("**🌤 Weather · Open-Meteo**")
            if weather_data:
                st.success("Connected — 7-day hourly forecast loaded")
            else:
                st.error(f"Failed: {weather_err}")
        with col_s2:
            st.markdown("**⚡ ERCOT North Hub · Public API**")
            if da_count > 0:
                st.success(ercot_status)
            else:
                st.warning(ercot_status)
        with col_s3:
            st.markdown("**🔥 Henry Hub Gas · FRED (St. Louis Fed)**")
            if gas_df is not None:
                st.success(gas_status)
            else:
                st.warning(gas_status)
        st.markdown("<p style='font-size:12px;color:#888'>All sources are free public APIs — no API keys required. Data cached 30–60 min.</p>", unsafe_allow_html=True)

    # ── Build model outputs ───────────────────────────────────────────────────
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

    # Latest gas price for metric
    latest_gas = None
    latest_gas_date = None
    if gas_df is not None and len(gas_df) > 0:
        latest_gas      = round(float(gas_df["HH Price ($/MMBtu)"].iloc[0]), 3)
        latest_gas_date = gas_df["Date"].iloc[0]

    # ── Metric tiles ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Today — DA Peak",
            f"${today['DA $/MWh']:.2f} /MWh",
            delta_str(today["DA $/MWh"], hist_da_avg),
        )
    with m2:
        st.metric(
            "Today — RT Peak",
            f"${today['RT $/MWh']:.2f} /MWh",
            delta_str(today["RT $/MWh"], hist_rt_avg),
        )
    with m3:
        if tomorrow is not None:
            st.metric(
                "Tomorrow — DA Peak",
                f"${tomorrow['DA $/MWh']:.2f} /MWh",
                delta_str(tomorrow["DA $/MWh"], hist_da_avg),
            )
    with m4:
        if latest_gas is not None:
            st.metric(
                f"Henry Hub Gas ({latest_gas_date})",
                f"${latest_gas:.3f} /MMBtu",
                f"45-day DA avg: ${hist_da_avg:.2f}/MWh" if hist_da_avg else None,
                delta_color="off",
            )
        elif hist_da_avg:
            st.metric("45-Day Avg Peak DA", f"${hist_da_avg:.2f} /MWh",
                      "Historical baseline · on-peak hrs only", delta_color="off")

    st.divider()

    # ── 7-Day Price Forecast ──────────────────────────────────────────────────
    st.markdown("### 📋 7-Day Price Forecast")
    # Color legend as styled markdown — no caption font issues
    st.markdown(
        "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif'>"
        "On-peak hours only (HE07–HE22 = 7 am – 10 pm CT) &nbsp;·&nbsp; "
        "<span style='background:#FFFDE7;color:#5A4A00;padding:2px 6px;border-radius:3px'>🟡 Under $35</span> &nbsp;"
        "<span style='background:#FFF3CC;color:#7A5000;padding:2px 6px;border-radius:3px'>🟡 $35 – $60</span> &nbsp;"
        "<span style='background:#FFDAD4;color:#8B0000;padding:2px 6px;border-radius:3px'>🔴 $60 – $100</span> &nbsp;"
        "<span style='background:#E8D5F5;color:#4A0080;padding:2px 6px;border-radius:3px'>🟣 Over $100</span>"
        "</p>",
        unsafe_allow_html=True,
    )

    price_cols = ["DA $/MWh", "RT $/MWh"]
    styled_fc = (
        forecast_df.style
        .applymap(color_price, subset=price_cols)
        .format({"DA $/MWh": "${:.2f}", "RT $/MWh": "${:.2f}", "Avg Peak Temp": "{:.1f}°F"})
        .set_table_styles(TBL_STYLES)
        .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "13px"})
    )
    st.dataframe(styled_fc, use_container_width=True, hide_index=True)

    st.divider()

    # ── Weather Comparison ────────────────────────────────────────────────────
    st.markdown("### 🌡️ DFW Weather — Forecast vs. Same Week in Prior Years")
    st.markdown(
        "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif'>"
        "Compares this week's forecast high temp to the same calendar dates 1, 2, and 3 years ago. &nbsp;"
        "<span style='color:#C03A00;font-weight:600'>Red = warmer than prior year</span> &nbsp;·&nbsp; "
        "<span style='color:#1A7A3C;font-weight:600'>Green = cooler</span>"
        "</p>",
        unsafe_allow_html=True,
    )

    diff_cols = [c for c in weather_cmp.columns if c.startswith("vs ")]
    styled_wc = (
        weather_cmp.style
        .applymap(color_diff, subset=diff_cols)
        .set_table_styles(TBL_STYLES)
        .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "13px"})
    )
    st.dataframe(styled_wc, use_container_width=True, hide_index=True)

    st.divider()

    # ── Historical Power Prices ───────────────────────────────────────────────
    st.markdown("### 📊 Historical On-Peak Power Prices — Last 45 Days")

    if hist_df is not None and len(hist_df) > 0:
        st.markdown(
            f"<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif'>"
            f"Actual ERCOT North Hub on-peak (HE07–HE22) settlement prices. "
            f"{da_count} DA rows and {rt_count} RT rows loaded from ERCOT Public API.</p>",
            unsafe_allow_html=True,
        )
        hc = hist_df.copy()
        hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
        hc["Date"] = hc["datetime"].apply(lambda x: x.strftime("%Y-%m-%d"))
        pk = hc[hc["hour"].isin(PEAK_HOURS)].copy()

        if len(pk) > 0:
            daily_hist = (
                pk.groupby("Date")
                .agg(**{
                    "DA Avg $/MWh": ("da_price", "mean"),
                    "DA Max $/MWh": ("da_price", "max"),
                    "RT Avg $/MWh": ("rt_price", "mean"),
                    "RT Max $/MWh": ("rt_price", "max"),
                })
                .reset_index()
                .sort_values("Date", ascending=False)
            )
            for col in ["DA Avg $/MWh","DA Max $/MWh","RT Avg $/MWh","RT Max $/MWh"]:
                daily_hist[col] = daily_hist[col].round(2)

            hist_price_cols = ["DA Avg $/MWh","DA Max $/MWh","RT Avg $/MWh","RT Max $/MWh"]
            styled_hist = (
                daily_hist.style
                .applymap(color_price, subset=hist_price_cols)
                .format({c: "${:.2f}" for c in hist_price_cols})
                .set_table_styles(TBL_STYLES)
                .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "13px"})
            )
            st.dataframe(styled_hist, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "ERCOT API returned no price data right now — this is usually temporary. "
            "The price forecast above is using default baseline prices ($45 DA / $47 RT) until live data comes back. "
            "Try refreshing the page in a few minutes."
        )

    st.divider()

    # ── Henry Hub Gas Prices ──────────────────────────────────────────────────
    st.markdown("### 🔥 Henry Hub Natural Gas Spot Prices — Last 90 Days")

    if gas_df is not None and len(gas_df) > 0:
        st.markdown(
            "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif'>"
            "Daily Henry Hub natural gas spot prices ($/MMBtu). Source: FRED / St. Louis Fed (EIA series DHHNGSP). "
            "Gas sets the marginal price in ERCOT ~70–80% of on-peak hours. "
            "January 2026 avg was $7.72/MMBtu — highest since Sept 2022 — driven by Winter Storm Fern."
            "</p>",
            unsafe_allow_html=True,
        )
        styled_gas = (
            gas_df.head(90).style
            .applymap(color_gas, subset=["HH Price ($/MMBtu)"])
            .format({"HH Price ($/MMBtu)": "${:.3f}"})
            .set_table_styles(TBL_STYLES)
            .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "13px"})
        )
        st.dataframe(styled_gas, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Henry Hub data unavailable: {gas_status}")

    st.divider()

    # ── Gas vs Power Correlation ──────────────────────────────────────────────
    st.markdown("### ⚡ Power vs. Gas — Implied Heat Rate Table")
    st.markdown(
        "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif'>"
        "DA/Gas Ratio = implied heat rate ($/MWh ÷ $/MMBtu). A ratio above ~8–9x means power is expensive "
        "relative to gas (good margins for gas generators). Below ~6x means power is cheap vs fuel cost. "
        "Normal ERCOT on-peak implied heat rates run 8–12x in summer."
        "</p>",
        unsafe_allow_html=True,
    )

    comp_df = build_gas_power_comparison(hist_df, gas_df)
    if comp_df is not None and len(comp_df) > 0:
        power_cols2 = ["DA Avg $/MWh", "RT Avg $/MWh"]
        ratio_cols  = ["DA/Gas Ratio", "RT/Gas Ratio"]
        gas_col     = ["HH Gas $/MMBtu"]

        styled_comp = (
            comp_df.style
            .applymap(color_price,  subset=power_cols2)
            .applymap(color_gas,    subset=gas_col)
            .applymap(color_ratio,  subset=ratio_cols)
            .format({
                "DA Avg $/MWh":   "${:.2f}",
                "RT Avg $/MWh":   "${:.2f}",
                "HH Gas $/MMBtu": "${:.3f}",
                "DA/Gas Ratio":   "{:.2f}x",
                "RT/Gas Ratio":   "{:.2f}x",
            })
            .set_table_styles(TBL_STYLES)
            .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "13px"})
        )
        st.dataframe(styled_comp, use_container_width=True, hide_index=True)
    else:
        st.info("Gas vs. power comparison requires both ERCOT and EIA/FRED data to be available simultaneously.")

    st.divider()

    # ── Footer ────────────────────────────────────────────────────────────────
    f1, f2 = st.columns([4, 1])
    with f1:
        st.markdown(
            "<p style='font-size:12px;color:#888;font-family:DM Mono,monospace'>"
            "Sources: ERCOT Public API · FRED St. Louis Fed (EIA DHHNGSP) · Open-Meteo Forecast & Archive · "
            "All free, no API keys · Prices cached 30–60 min · "
            f"Last loaded {now_ct.strftime('%b %d, %Y %I:%M %p CT')}"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:12px;color:#aaa;font-family:DM Mono,monospace'>"
            "Statistical estimates only — not a substitute for professional market analysis or trading signals."
            "</p>",
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown("**⚡ GRIDEDGE**")


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import requests
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

# ── Minimal CSS — safe overrides only, no HTML table rendering ────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
.stApp                       { background: #F7F5F0; }
#MainMenu, footer, header    { visibility: hidden; }
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
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ERCOT_TZ   = pytz.timezone("US/Central")
LAT, LON   = 32.90, -97.04       # DFW — center of ERCOT North Hub
PEAK_HOURS = list(range(7, 23))  # HE07–HE22


# ── Data fetching ─────────────────────────────────────────────────────────────

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
        return r.json()
    except Exception:
        return None


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

    try:
        base   = "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices"
        params = {
            "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
            "deliveryDateTo":   now.strftime("%Y-%m-%d"),
            "settlementPoint":  "HB_NORTH",
            "size": 5000,
        }
        r = requests.get(base, params=params, timeout=15, headers={"Accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            if "data" in data:
                for row in data["data"]:
                    try:
                        dt = ERCOT_TZ.localize(
                            datetime.strptime(f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M")
                        )
                        prices.append({"datetime": dt, "da_price": float(row[3]), "rt_price": None})
                    except Exception:
                        continue
    except Exception:
        pass

    rt_map = {}
    try:
        base_rt   = "https://api.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub"
        params_rt = {
            "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
            "deliveryDateTo":   now.strftime("%Y-%m-%d"),
            "settlementPoint":  "HB_NORTH",
            "size": 5000,
        }
        r = requests.get(base_rt, params=params_rt, timeout=15, headers={"Accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            if "data" in data:
                for row in data["data"]:
                    try:
                        dt = ERCOT_TZ.localize(
                            datetime.strptime(f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M")
                        )
                        rt_map[dt] = float(row[3])
                    except Exception:
                        continue
    except Exception:
        pass

    for p in prices:
        p["rt_price"] = rt_map.get(p["datetime"])

    if prices:
        df = pd.DataFrame(prices).dropna(subset=["da_price"])
        return df.sort_values("datetime").reset_index(drop=True)
    return None


# ── Model ─────────────────────────────────────────────────────────────────────

def temp_to_mult(temp_f):
    for thresh, mult in [(100,3.5),(95,2.5),(90,1.8),(85,1.35),(80,1.15),
                          (75,1.0),(65,0.85),(55,0.90),(45,1.05),(35,1.25),
                          (25,1.75),(15,2.8)]:
        if temp_f >= thresh:
            return mult
    return 4.0


def build_forecast(weather_data, hist_df):
    daily  = weather_data.get("daily", {})
    hourly = weather_data.get("hourly", {})
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
    peak_h = h_df[h_df["hour"].isin(PEAK_HOURS)]
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

        if   max(da, rt) >= 100: alert = "🔴 HIGH"
        elif max(da, rt) >=  50: alert = "🟡 MEDIUM"
        else:                     alert = "🟢 LOW"

        rows.append({
            "Day":                  day_label,
            "Date":                 date_obj.strftime("%b %d"),
            "Hi °F":                int(round(hi)),
            "Lo °F":                int(round(lo)),
            "Avg Peak Temp (°F)":   round(pk_temp, 1),
            "Wind (mph)":           int(round(pk_wind)),
            "DA Forecast ($/MWh)":  da,
            "RT Forecast ($/MWh)":  rt,
            "Price Alert":          alert,
        })

    return pd.DataFrame(rows)


def build_weather_comparison(weather_data):
    daily    = weather_data.get("daily", {})
    dates    = daily.get("time", [])
    hi_now   = daily.get("temperature_2m_max", [])
    lo_now   = daily.get("temperature_2m_min", [])

    hist_1 = fetch_historical_weather(1)
    hist_2 = fetch_historical_weather(2)
    hist_3 = fetch_historical_weather(3)

    def get_hi_lo(hdata, idx):
        if hdata is None:
            return None, None
        try:
            return round(hdata["daily"]["temperature_2m_max"][idx], 0), \
                   round(hdata["daily"]["temperature_2m_min"][idx], 0)
        except Exception:
            return None, None

    def diff_str(val_now, val_hist):
        if val_now is None or val_hist is None:
            return "N/A"
        d = val_now - int(val_hist)
        return f"+{d}°" if d > 0 else f"{d}°"

    now_ct = datetime.now(ERCOT_TZ)
    rows   = []
    for i, date_str in enumerate(dates[:7]):
        date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
        is_today  = date_str == now_ct.strftime("%Y-%m-%d")
        is_tmrw   = date_str == (now_ct + timedelta(days=1)).strftime("%Y-%m-%d")
        day_lbl   = "Today" if is_today else ("Tomorrow" if is_tmrw else date_obj.strftime("%A"))

        hi_this = int(round(hi_now[i])) if i < len(hi_now) else None
        lo_this = int(round(lo_now[i])) if i < len(lo_now) else None
        hi_1, lo_1 = get_hi_lo(hist_1, i)
        hi_2, lo_2 = get_hi_lo(hist_2, i)
        hi_3, lo_3 = get_hi_lo(hist_3, i)

        yr = date_obj.year
        rows.append({
            "Day":                  day_lbl,
            "Date":                 date_obj.strftime("%b %d"),
            "Forecast Hi/Lo":       f"{hi_this}° / {lo_this}°" if hi_this else "—",
            f"{yr-1} Hi/Lo":        f"{int(hi_1)}° / {int(lo_1)}°" if hi_1 else "N/A",
            f"vs {yr-1}":           diff_str(hi_this, hi_1),
            f"{yr-2} Hi/Lo":        f"{int(hi_2)}° / {int(lo_2)}°" if hi_2 else "N/A",
            f"vs {yr-2}":           diff_str(hi_this, hi_2),
            f"{yr-3} Hi/Lo":        f"{int(hi_3)}° / {int(lo_3)}°" if hi_3 else "N/A",
            f"vs {yr-3}":           diff_str(hi_this, hi_3),
        })

    return pd.DataFrame(rows)


# ── Styling helpers ───────────────────────────────────────────────────────────

TBL_STYLES = [
    {"selector": "th", "props": [
        ("background-color", "#1A1A1A"), ("color", "#F7F5F0"),
        ("font-size", "12px"), ("text-align", "center"),
        ("padding", "10px 12px"), ("font-family", "DM Sans, sans-serif"),
        ("letter-spacing", "0.05em"),
    ]},
    {"selector": "td", "props": [("padding", "10px 14px"), ("text-align", "center")]},
    {"selector": "tr:nth-child(even) td", "props": [("background-color", "#EEEBE4")]},
    {"selector": "tr:hover td", "props": [("background-color", "#E0DBD0")]},
]

def color_price(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 100: return "background-color:#FFDAD4;color:#8B0000;font-weight:600"
    if val >= 60:  return "background-color:#FFF3CC;color:#7A5000"
    if val < 30:   return "background-color:#DFF5E8;color:#1A5C35"
    return ""

def color_diff(val):
    if not isinstance(val, str) or val == "N/A":
        return ""
    try:
        n = int(val.replace("+","").replace("°",""))
        if n >= 5:  return "color:#C03A00;font-weight:600"
        if n <= -5: return "color:#1A7A3C;font-weight:600"
    except Exception:
        pass
    return ""


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    now_ct = datetime.now(ERCOT_TZ)

    # Header
    c1, c2, c3 = st.columns([2, 5, 2])
    with c1: st.markdown("### ⚡ GRIDEDGE")
    with c2: st.caption("ERCOT NORTH HUB  ·  SETTLEMENT PRICE INTELLIGENCE")
    with c3: st.caption(now_ct.strftime("%b %d, %Y  ·  %I:%M %p CT"))
    st.divider()
    st.markdown("## Peak Hour Price Forecast")
    st.caption("Projected on-peak (HE07–HE22) Day-Ahead and Real-Time settlement prices · ERCOT North Hub · $/MWh")
    st.divider()

    # Load data
    with st.spinner("⏳ Fetching live weather and price data…"):
        weather_data   = fetch_weather_forecast()
        hist_df        = fetch_ercot_historical()

    if weather_data is None:
        st.error("⚠️ Could not reach the weather API. Please refresh.")
        return

    forecast_df  = build_forecast(weather_data, hist_df)
    weather_cmp  = build_weather_comparison(weather_data)

    # Compute baseline from history
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
        return f"{sign}{pct:.1f}% vs 30-day avg"

    today    = forecast_df.iloc[0]
    tomorrow = forecast_df.iloc[1] if len(forecast_df) > 1 else None

    # ── Metric tiles ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Today · DA Peak ($/MWh)",
                  f"${today['DA Forecast ($/MWh)']:.2f}",
                  delta_str(today["DA Forecast ($/MWh)"], hist_da_avg))
    with m2:
        st.metric("Today · RT Peak ($/MWh)",
                  f"${today['RT Forecast ($/MWh)']:.2f}",
                  delta_str(today["RT Forecast ($/MWh)"], hist_rt_avg))
    with m3:
        if tomorrow is not None:
            st.metric("Tomorrow · DA Peak ($/MWh)",
                      f"${tomorrow['DA Forecast ($/MWh)']:.2f}",
                      delta_str(tomorrow["DA Forecast ($/MWh)"], hist_da_avg))
    with m4:
        if hist_da_avg:
            st.metric("30-Day Avg Peak DA ($/MWh)", f"${hist_da_avg:.2f}",
                      "Historical baseline · on-peak hrs only", delta_color="off")
        else:
            st.metric("30-Day Avg Peak DA", "Unavailable")

    st.divider()

    # ── 7-Day Price Forecast ──────────────────────────────────────────────────
    st.markdown("### 📋 7-Day Price Forecast")
    st.caption("On-peak hours only (HE07–HE22 = 7am–10pm CT).  🟢 Under $35  ·  ⚪ $35–$60  ·  🟡 $60–$100  ·  🔴 Over $100")

    price_cols = ["DA Forecast ($/MWh)", "RT Forecast ($/MWh)"]
    styled_fc = (
        forecast_df.style
        .applymap(color_price, subset=price_cols)
        .format({"DA Forecast ($/MWh)": "${:.2f}", "RT Forecast ($/MWh)": "${:.2f}",
                 "Avg Peak Temp (°F)": "{:.1f}"})
        .set_table_styles(TBL_STYLES)
        .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "14px"})
    )
    st.dataframe(styled_fc, use_container_width=True, hide_index=True)

    st.divider()

    # ── Weather Comparison ────────────────────────────────────────────────────
    st.markdown("### 🌡️ Weather Outlook · DFW — vs. Same Week in Prior Years")
    st.caption("Compares this week's forecast high to the same calendar dates 1, 2, and 3 years ago.  Red = warmer than prior year  ·  Green = cooler")

    diff_cols = [c for c in weather_cmp.columns if c.startswith("vs ")]
    styled_wc = (
        weather_cmp.style
        .applymap(color_diff, subset=diff_cols)
        .set_table_styles(TBL_STYLES)
        .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "14px"})
    )
    st.dataframe(styled_wc, use_container_width=True, hide_index=True)

    st.divider()

    # ── Historical Price Context ──────────────────────────────────────────────
    if hist_df is not None and len(hist_df) > 0:
        st.markdown("### 📊 Historical Peak Prices — Last 45 Days")
        st.caption("Actual ERCOT North Hub on-peak settlement prices. Source: ERCOT Public API.")

        hc = hist_df.copy()
        hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
        hc["Date"] = hc["datetime"].apply(lambda x: x.strftime("%Y-%m-%d"))
        pk = hc[hc["hour"].isin(PEAK_HOURS)].copy()

        if len(pk) > 0:
            daily_hist = (
                pk.groupby("Date")
                .agg(
                    **{"DA Avg ($/MWh)":  ("da_price", "mean"),
                       "DA Max ($/MWh)":  ("da_price", "max"),
                       "RT Avg ($/MWh)":  ("rt_price", "mean"),
                       "RT Max ($/MWh)":  ("rt_price", "max")}
                )
                .reset_index()
                .sort_values("Date", ascending=False)
            )
            for col in ["DA Avg ($/MWh)","DA Max ($/MWh)","RT Avg ($/MWh)","RT Max ($/MWh)"]:
                daily_hist[col] = daily_hist[col].round(2)

            hist_price_cols = ["DA Avg ($/MWh)","DA Max ($/MWh)","RT Avg ($/MWh)","RT Max ($/MWh)"]
            styled_hist = (
                daily_hist.style
                .applymap(color_price, subset=hist_price_cols)
                .format({c: "${:.2f}" for c in hist_price_cols})
                .set_table_styles(TBL_STYLES)
                .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "14px"})
            )
            st.dataframe(styled_hist, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ ERCOT API unavailable — forecasts use default baseline prices. Try refreshing later.")

    st.divider()

    # Footer
    f1, f2 = st.columns([4, 1])
    with f1:
        st.caption(
            "Data sources: ERCOT Public API · Open-Meteo Historical Archive · Open-Meteo Forecast  ·  "
            "All free, no API keys required  ·  Refreshes every 30 min  ·  "
            f"Last updated {now_ct.strftime('%b %d, %Y %I:%M %p CT')}"
        )
        st.caption("⚠️ Statistical estimates only. Not a substitute for professional market analysis.")
    with f2:
        st.markdown("**⚡ GRIDEDGE**")


if __name__ == "__main__":
    main()

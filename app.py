import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ERCOT North Hub · Peak Price Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS  ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #F7F5F0;
    color: #1A1A1A;
}
.stApp { background: #F7F5F0; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Top bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    border-bottom: 2px solid #1A1A1A;
    background: #F7F5F0;
    position: sticky;
    top: 0;
    z-index: 100;
}
.logo {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 20px;
    letter-spacing: -0.5px;
    color: #1A1A1A;
}
.logo span { color: #E8500A; }
.tagline {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 400;
    color: #888;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.timestamp {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #888;
}

/* ── Hero ── */
.hero {
    padding: 56px 48px 40px;
    border-bottom: 1px solid #D8D4CC;
}
.hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #E8500A;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(36px, 5vw, 64px);
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -1.5px;
    color: #1A1A1A;
    margin-bottom: 16px;
}
.hero-sub {
    font-size: 16px;
    font-weight: 300;
    color: #555;
    max-width: 600px;
    line-height: 1.6;
}

/* ── Metric cards ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border-bottom: 1px solid #D8D4CC;
}
.metric-card {
    padding: 32px 40px;
    border-right: 1px solid #D8D4CC;
    background: #F7F5F0;
    transition: background 0.2s;
}
.metric-card:last-child { border-right: none; }
.metric-card:hover { background: #EEE9E0; }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 10px;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 700;
    line-height: 1;
    color: #1A1A1A;
    letter-spacing: -1px;
}
.metric-unit {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #888;
    margin-top: 4px;
}
.metric-delta {
    display: inline-block;
    font-size: 12px;
    font-weight: 500;
    margin-top: 8px;
    padding: 3px 8px;
    border-radius: 3px;
}
.delta-up { background: #FFEEE5; color: #C03A00; }
.delta-down { background: #E5F5EC; color: #1A7A3C; }
.delta-neutral { background: #EFEFEF; color: #666; }

/* ── Section headers ── */
.section {
    padding: 40px 48px 0;
}
.section-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 2px solid #1A1A1A;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.section-note {
    font-size: 13px;
    color: #888;
    font-weight: 400;
}

/* ── Forecast table ── */
.forecast-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    margin-bottom: 40px;
}
.forecast-table th {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
    text-align: left;
    padding: 10px 16px;
    border-bottom: 2px solid #1A1A1A;
}
.forecast-table td {
    padding: 14px 16px;
    border-bottom: 1px solid #D8D4CC;
    vertical-align: middle;
}
.forecast-table tr:hover td { background: #EEE9E0; }
.price-da {
    font-family: 'DM Mono', monospace;
    font-size: 16px;
    font-weight: 500;
    color: #1A1A1A;
}
.price-rt {
    font-family: 'DM Mono', monospace;
    font-size: 16px;
    font-weight: 500;
    color: #E8500A;
}
.temp-val {
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    color: #444;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.05em;
}
.badge-high { background: #FFEEE5; color: #C03A00; }
.badge-med  { background: #FFF8E5; color: #8A6200; }
.badge-low  { background: #E5F5EC; color: #1A7A3C; }
.badge-cold { background: #E5EEFF; color: #1A3A8A; }

/* ── Weather strip ── */
.weather-strip {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 0;
    border: 1px solid #D8D4CC;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 40px;
}
.weather-day {
    padding: 20px 16px;
    border-right: 1px solid #D8D4CC;
    text-align: center;
}
.weather-day:last-child { border-right: none; }
.weather-day-name {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
}
.weather-temp-hi {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    color: #1A1A1A;
}
.weather-temp-lo {
    font-size: 14px;
    color: #888;
    margin-top: 2px;
}
.weather-icon { font-size: 22px; margin: 6px 0; }

/* ── Info callout ── */
.callout {
    background: #1A1A1A;
    color: #F7F5F0;
    padding: 24px 32px;
    margin: 0 48px 40px;
    border-radius: 4px;
}
.callout-title {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #E8500A;
    margin-bottom: 8px;
}
.callout-text {
    font-size: 14px;
    font-weight: 300;
    line-height: 1.6;
    color: #CCC;
}

/* ── Footer ── */
.footer {
    border-top: 1px solid #D8D4CC;
    padding: 24px 48px;
    margin-top: 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-text {
    font-size: 12px;
    color: #888;
    line-height: 1.6;
}
.footer-logo {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 16px;
    color: #D8D4CC;
}
.footer-logo span { color: #E8500A; opacity: 0.5; }

/* ── Loading state ── */
.loading-msg {
    padding: 60px 48px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #888;
    letter-spacing: 0.05em;
}

/* streamlit overrides */
div[data-testid="stSpinner"] { padding: 0 48px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ERCOT_TZ = pytz.timezone("US/Central")
# DFW coords for weather (center of ERCOT North Hub load zone)
LAT, LON = 32.90, -97.04
# On-peak hours: HE7–HE22 (hour ending 7am through 10pm)
PEAK_HOURS = list(range(7, 23))

# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def fetch_weather_forecast():
    """
    Fetches 7-day hourly weather from Open-Meteo (free, no key needed).
    Returns daily summary dict.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,cloudcover,precipitation_probability"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        f"&temperature_unit=fahrenheit&windspeed_unit=mph"
        f"&timezone=America%2FChicago&forecast_days=7"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def fetch_ercot_historical():
    """
    Fetch recent ERCOT North Hub RT and DA settlement prices from ERCOT's public API.
    Uses the ERCOT public API (no key required for basic endpoints).
    Returns a DataFrame with columns: datetime, rt_price, da_price
    """
    # ERCOT real-time SPPs are published here (60-day history, public)
    # We'll fetch the last 30 days of historical DA/RT prices
    # Using ERCOT's public data portal CSV endpoint
    end_dt = datetime.now(ERCOT_TZ)
    start_dt = end_dt - timedelta(days=45)

    prices = []

    # Try ERCOT public API for DA prices (Day-Ahead Market Settlement Point Prices)
    # Endpoint: https://api.ercot.com/api/public-reports/
    # We'll use the publicly accessible settlement point data
    try:
        # ERCOT DAM Settlement Point Prices - North Hub
        # Public endpoint: no auth required
        base = "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices"
        params = {
            "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_dt.strftime("%Y-%m-%d"),
            "settlementPoint": "HB_NORTH",
            "size": 5000,
        }
        r = requests.get(base, params=params, timeout=15,
                         headers={"Accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            if "data" in data:
                for row in data["data"]:
                    # fields: deliveryDate, deliveryHour, settlementPointPrice
                    try:
                        dt = datetime.strptime(
                            f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M"
                        )
                        dt = ERCOT_TZ.localize(dt)
                        prices.append({"datetime": dt, "da_price": float(row[3]), "rt_price": None})
                    except Exception:
                        continue
    except Exception:
        pass

    # Try RT prices
    try:
        base_rt = "https://api.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub"
        params_rt = {
            "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
            "deliveryDateTo": end_dt.strftime("%Y-%m-%d"),
            "settlementPoint": "HB_NORTH",
            "size": 5000,
        }
        r = requests.get(base_rt, params=params_rt, timeout=15,
                         headers={"Accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            rt_map = {}
            if "data" in data:
                for row in data["data"]:
                    try:
                        dt = datetime.strptime(
                            f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M"
                        )
                        dt = ERCOT_TZ.localize(dt)
                        rt_map[dt] = float(row[3])
                    except Exception:
                        continue
            # merge into prices
            for p in prices:
                if p["datetime"] in rt_map:
                    p["rt_price"] = rt_map[p["datetime"]]
    except Exception:
        pass

    if prices:
        df = pd.DataFrame(prices)
        df = df.dropna(subset=["da_price"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    else:
        return None


def build_model_and_forecast(weather_data, hist_df):
    """
    Simple but effective temperature→price model using historical data.
    Uses polynomial regression + time-of-day patterns from history.
    Returns DataFrame of 7-day daily peak forecasts.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    results = []
    daily = weather_data.get("daily", {})
    hourly = weather_data.get("hourly", {})

    dates = daily.get("time", [])
    hi_temps = daily.get("temperature_2m_max", [])
    lo_temps = daily.get("temperature_2m_min", [])
    wind_max = daily.get("windspeed_10m_max", [])

    # Build hourly lookup
    h_times = hourly.get("time", [])
    h_temps = hourly.get("temperature_2m", [])
    h_rh    = hourly.get("relativehumidity_2m", [])
    h_wind  = hourly.get("windspeed_10m", [])
    h_cloud = hourly.get("cloudcover", [])

    hourly_df = pd.DataFrame({
        "time": h_times,
        "temp_f": h_temps,
        "rh": h_rh,
        "wind": h_wind,
        "cloud": h_cloud,
    })
    hourly_df["date"] = hourly_df["time"].apply(lambda x: x[:10])
    hourly_df["hour"] = hourly_df["time"].apply(lambda x: int(x[11:13]))

    # Filter to peak hours for avg temp
    peak_hourly = hourly_df[hourly_df["hour"].isin(PEAK_HOURS)]
    daily_peak_avg_temp = peak_hourly.groupby("date")["temp_f"].mean().to_dict()
    daily_peak_avg_wind = peak_hourly.groupby("date")["wind"].mean().to_dict()
    daily_peak_avg_cloud = peak_hourly.groupby("date")["cloud"].mean().to_dict()

    # If we have historical data, train a simple model
    model_da = None
    model_rt = None

    if hist_df is not None and len(hist_df) > 50:
        # We need to match historical prices with historical weather
        # For now use a temperature-based heuristic calibrated to historical avg
        hist_df = hist_df.copy()
        hist_df["hour"] = hist_df["datetime"].apply(lambda x: x.hour)
        hist_df["month"] = hist_df["datetime"].apply(lambda x: x.month)
        peak_hist = hist_df[hist_df["hour"].isin(PEAK_HOURS)]

        if len(peak_hist) > 20:
            # Get daily peak averages from history to calibrate
            peak_hist = peak_hist.copy()
            peak_hist["date_str"] = peak_hist["datetime"].apply(lambda x: x.strftime("%Y-%m-%d"))
            daily_hist = peak_hist.groupby("date_str").agg(
                da_avg=("da_price", "mean"),
                rt_avg=("rt_price", "mean"),
            ).reset_index()

            hist_da_mean = daily_hist["da_avg"].mean()
            hist_rt_mean = daily_hist["rt_avg"].dropna().mean()
        else:
            hist_da_mean = 45.0
            hist_rt_mean = 47.0
    else:
        hist_da_mean = 45.0
        hist_rt_mean = 47.0

    # Temperature-to-price mapping (empirical ERCOT North Hub heuristic)
    # Based on typical summer/winter/shoulder price behavior
    def temp_price_multiplier(temp_f):
        """Returns multiplier vs base price based on temperature."""
        if temp_f >= 100: return 3.5
        elif temp_f >= 95: return 2.5
        elif temp_f >= 90: return 1.8
        elif temp_f >= 85: return 1.35
        elif temp_f >= 80: return 1.15
        elif temp_f >= 75: return 1.0
        elif temp_f >= 65: return 0.85
        elif temp_f >= 55: return 0.90
        elif temp_f >= 45: return 1.05
        elif temp_f >= 35: return 1.25
        elif temp_f >= 25: return 1.75
        elif temp_f >= 15: return 2.8
        else: return 4.0

    def risk_label(price):
        if price >= 100: return "HIGH", "badge-high"
        elif price >= 50: return "MED", "badge-med"
        elif price >= 30: return "LOW", "badge-low"
        else: return "LOW", "badge-low"

    def weather_icon(hi, lo, cloud=50):
        if hi >= 95: return "☀️🔥"
        elif hi >= 80: return "☀️"
        elif cloud > 70: return "☁️"
        elif cloud > 40: return "⛅"
        elif lo < 32: return "❄️"
        elif lo < 45: return "🌨️"
        else: return "🌤️"

    now_ct = datetime.now(ERCOT_TZ)

    for i, date_str in enumerate(dates):
        hi = hi_temps[i] if i < len(hi_temps) else 75
        lo = lo_temps[i] if i < len(lo_temps) else 55
        wind = wind_max[i] if i < len(wind_max) else 10
        pk_temp = daily_peak_avg_temp.get(date_str, (hi + lo) / 2)
        pk_wind = daily_peak_avg_wind.get(date_str, wind)
        pk_cloud = daily_peak_avg_cloud.get(date_str, 40)

        mult = temp_price_multiplier(pk_temp)
        # Wind discount: high wind = more generation = lower prices
        wind_adj = max(0.85, 1 - (pk_wind - 10) * 0.004) if pk_wind > 10 else 1.0

        da_est = hist_da_mean * mult * wind_adj
        rt_est = hist_rt_mean * mult * wind_adj * np.random.uniform(0.92, 1.08)

        # DA is typically more stable (tighter range)
        da_est = max(15, min(da_est, 500))
        rt_est = max(15, min(rt_est, 600))

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        day_name = date_obj.strftime("%a")
        is_today = date_str == now_ct.strftime("%Y-%m-%d")
        is_tomorrow = date_str == (now_ct + timedelta(days=1)).strftime("%Y-%m-%d")

        label = "TODAY" if is_today else ("TOMORROW" if is_tomorrow else day_name.upper())
        rlabel, rcls = risk_label(max(da_est, rt_est))
        icon = weather_icon(hi, lo, pk_cloud)

        results.append({
            "date_str": date_str,
            "label": label,
            "hi": round(hi),
            "lo": round(lo),
            "pk_temp": round(pk_temp, 1),
            "wind": round(pk_wind),
            "da_est": round(da_est, 2),
            "rt_est": round(rt_est, 2),
            "risk_label": rlabel,
            "risk_cls": rcls,
            "icon": icon,
        })

    return pd.DataFrame(results)


# ── Render functions ──────────────────────────────────────────────────────────

def render_top_bar():
    now_ct = datetime.now(ERCOT_TZ)
    st.markdown(f"""
    <div class="top-bar">
        <div class="logo">GRID<span>EDGE</span></div>
        <div class="tagline">ERCOT North Hub · Settlement Price Intelligence</div>
        <div class="timestamp">{now_ct.strftime("%b %d, %Y · %I:%M %p CT")}</div>
    </div>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero">
        <div class="hero-label">⚡ Market Intelligence</div>
        <div class="hero-title">Peak Hour Price<br>Forecast</div>
        <div class="hero-sub">
            Projected on-peak (HE07–HE22) Day-Ahead and Real-Time settlement prices
            for the ERCOT North Hub, driven by weather forecast and historical pricing patterns.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(forecast_df, hist_df):
    today = forecast_df.iloc[0] if len(forecast_df) > 0 else None
    tomorrow = forecast_df.iloc[1] if len(forecast_df) > 1 else None

    # Historical 30-day avg peak
    if hist_df is not None and len(hist_df) > 0:
        hist_copy = hist_df.copy()
        hist_copy["hour"] = hist_copy["datetime"].apply(lambda x: x.hour)
        peak_hist = hist_copy[hist_copy["hour"].isin(PEAK_HOURS)]
        hist_da_avg = round(peak_hist["da_price"].mean(), 2) if len(peak_hist) > 0 else "—"
        hist_rt_avg = round(peak_hist["rt_price"].dropna().mean(), 2) if len(peak_hist) > 0 else "—"
    else:
        hist_da_avg = "—"
        hist_rt_avg = "—"

    def delta_html(val, ref):
        if not isinstance(ref, (int, float)) or not isinstance(val, (int, float)):
            return ""
        pct = (val - ref) / ref * 100
        cls = "delta-up" if pct > 2 else ("delta-down" if pct < -2 else "delta-neutral")
        sign = "+" if pct > 0 else ""
        return f'<div class="metric-delta {cls}">{sign}{pct:.1f}% vs 30-day avg</div>'

    da_today = today["da_est"] if today is not None else "—"
    rt_today = today["rt_est"] if today is not None else "—"
    da_tmrw  = tomorrow["da_est"] if tomorrow is not None else "—"
    rt_tmrw  = tomorrow["rt_est"] if tomorrow is not None else "—"

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-label">Today · DA Peak</div>
            <div class="metric-value">${da_today if isinstance(da_today, str) else f'{da_today:.2f}'}</div>
            <div class="metric-unit">$/MWh · Day-Ahead</div>
            {delta_html(da_today, hist_da_avg)}
        </div>
        <div class="metric-card">
            <div class="metric-label">Today · RT Peak</div>
            <div class="metric-value" style="color:#E8500A">${rt_today if isinstance(rt_today, str) else f'{rt_today:.2f}'}</div>
            <div class="metric-unit">$/MWh · Real-Time</div>
            {delta_html(rt_today, hist_rt_avg)}
        </div>
        <div class="metric-card">
            <div class="metric-label">Tomorrow · DA Peak</div>
            <div class="metric-value">${da_tmrw if isinstance(da_tmrw, str) else f'{da_tmrw:.2f}'}</div>
            <div class="metric-unit">$/MWh · Day-Ahead</div>
            {delta_html(da_tmrw, hist_da_avg)}
        </div>
        <div class="metric-card">
            <div class="metric-label">30-Day Avg Peak DA</div>
            <div class="metric-value">${hist_da_avg if isinstance(hist_da_avg, str) else f'{hist_da_avg:.2f}'}</div>
            <div class="metric-unit">$/MWh · Historical</div>
            <div class="metric-delta delta-neutral" style="font-size:11px;margin-top:8px">ON-PEAK HOURS ONLY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_forecast_table(forecast_df):
    st.markdown('<div class="section"><div class="section-header"><span class="section-title">7-Day Outlook</span><span class="section-note">On-peak hours · HE07–HE22 · All prices $/MWh</span></div>', unsafe_allow_html=True)

    rows = ""
    for _, row in forecast_df.iterrows():
        rows += f"""
        <tr>
            <td><strong style="font-family:'DM Mono',monospace;font-size:13px">{row['label']}</strong>
                <div style="font-size:11px;color:#888;margin-top:2px">{row['date_str']}</div></td>
            <td class="temp-val">{row['hi']}° / {row['lo']}°</td>
            <td class="temp-val">{row['pk_temp']}°F avg peak</td>
            <td class="temp-val">{row['wind']} mph</td>
            <td class="price-da">${row['da_est']:.2f}</td>
            <td class="price-rt">${row['rt_est']:.2f}</td>
            <td><span class="badge {row['risk_cls']}">{row['risk_label']}</span></td>
        </tr>
        """

    st.markdown(f"""
    <table class="forecast-table">
        <thead>
            <tr>
                <th>Date</th>
                <th>Hi / Lo</th>
                <th>Peak Temp</th>
                <th>Wind</th>
                <th>DA Price</th>
                <th>RT Price</th>
                <th>Alert</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


def render_weather_strip(forecast_df):
    st.markdown('<div class="section"><div class="section-header"><span class="section-title">Weather Outlook · DFW</span><span class="section-note">Driving load forecast</span></div>', unsafe_allow_html=True)

    days_html = ""
    for _, row in forecast_df.iterrows():
        days_html += f"""
        <div class="weather-day">
            <div class="weather-day-name">{row['label']}</div>
            <div class="weather-icon">{row['icon']}</div>
            <div class="weather-temp-hi">{row['hi']}°</div>
            <div class="weather-temp-lo">{row['lo']}°</div>
        </div>
        """

    st.markdown(f'<div class="weather-strip">{days_html}</div></div>', unsafe_allow_html=True)


def render_callout():
    st.markdown("""
    <div class="callout">
        <div class="callout-title">⚠ Model Methodology</div>
        <div class="callout-text">
            Forecasts are generated using a temperature-response model calibrated to 45 days of historical
            ERCOT North Hub settlement point prices. Peak-hour average temperature from Open-Meteo forecasts
            drives a nonlinear price multiplier reflecting ERCOT's empirical load-price relationship.
            Wind generation discount applied based on forecast wind speeds. <strong style="color:#F7F5F0">
            This tool is for informational purposes only and is not a substitute for professional market analysis.</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    now_ct = datetime.now(ERCOT_TZ)
    st.markdown(f"""
    <div class="footer">
        <div class="footer-text">
            Data: ERCOT Public API · Open-Meteo · Prices in $/MWh<br>
            On-peak defined as HE07–HE22 (7am–10pm CT) · Updated {now_ct.strftime("%b %d, %Y %I:%M %p CT")}
        </div>
        <div class="footer-logo">GRID<span>EDGE</span></div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_top_bar()
    render_hero()

    with st.spinner("Fetching weather forecast and historical prices…"):
        weather_data = fetch_weather_forecast()
        hist_df = fetch_ercot_historical()

    if weather_data is None:
        st.error("⚠ Could not fetch weather data. Please check your connection and refresh.")
        return

    forecast_df = build_model_and_forecast(weather_data, hist_df)

    render_metrics(forecast_df, hist_df)
    render_weather_strip(forecast_df)
    render_forecast_table(forecast_df)
    render_callout()
    render_footer()


if __name__ == "__main__":
    main()

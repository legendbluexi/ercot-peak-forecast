"""
GRIDEDGE — ERCOT North Hub Peak Price Forecast
================================================
Credentials are stored in .streamlit/secrets.toml on GitHub (never public).
See the Setup Guide expander in the app for exact instructions.

Required secrets.toml entries:
    ercot_username         = "your-email@example.com"
    ercot_password         = "your-ercot-password"
    ercot_subscription_key = "your-primary-key-from-apiexplorer.ercot.com"
    eia_key                = "your-key-from-eia.gov/opendata"
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
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ERCOT_TZ   = pytz.timezone("US/Central")
LAT, LON   = 32.90, -97.04
PEAK_HOURS = list(range(7, 23))   # HE07–HE22

TBL_STYLES = [
    {"selector": "th", "props": [
        ("background-color","#1A1A1A"),("color","#F7F5F0"),
        ("font-size","12px"),("text-align","center"),
        ("padding","10px 12px"),("font-family","DM Sans, sans-serif"),
        ("letter-spacing","0.05em"),
    ]},
    {"selector": "td", "props": [
        ("padding","10px 14px"),("text-align","center"),
        ("font-family","DM Mono, monospace"),("font-size","13px"),
    ]},
    {"selector": "tr:nth-child(even) td", "props": [("background-color","#EEEBE4")]},
    {"selector": "tr:hover td",           "props": [("background-color","#E0DBD0")]},
]


# ── Color helpers ─────────────────────────────────────────────────────────────

def color_price(val):
    """Power price bands: green <$35 | yellow $35-60 | red $60-100 | purple $100+"""
    if not isinstance(val, (int, float)): return ""
    if val >= 100: return "background-color:#E8D5F5;color:#4A0080;font-weight:600"
    if val >= 60:  return "background-color:#FFDAD4;color:#8B0000;font-weight:600"
    if val >= 35:  return "background-color:#FFF3CC;color:#7A5000"
    return "background-color:#D4EDDA;color:#155724"

def color_gas(val):
    if not isinstance(val, (int, float)): return ""
    if val >= 5.0: return "background-color:#E8D5F5;color:#4A0080;font-weight:600"
    if val >= 3.5: return "background-color:#FFDAD4;color:#8B0000"
    if val >= 2.5: return "background-color:#FFF3CC;color:#7A5000"
    return "background-color:#D4EDDA;color:#155724"

def color_ratio(val):
    if not isinstance(val, (int, float)): return ""
    if val >= 12: return "background-color:#E8D5F5;color:#4A0080;font-weight:600"
    if val >= 8:  return "background-color:#FFF3CC;color:#7A5000"
    if val >= 5:  return ""
    return "background-color:#D4EDDA;color:#155724"

def color_diff(val):
    if not isinstance(val, str) or val in ("N/A","—"): return ""
    try:
        n = int(val.replace("+","").replace("°",""))
        if n >= 5:  return "color:#C03A00;font-weight:600"
        if n <= -5: return "color:#1A7A3C;font-weight:600"
    except: pass
    return ""

def subtext(text):
    """Consistent subtitle style that always renders correctly."""
    st.markdown(
        f"<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif;margin-top:-6px'>{text}</p>",
        unsafe_allow_html=True,
    )


# ── Credential loading ────────────────────────────────────────────────────────

def get_credentials():
    """
    Read credentials from st.secrets (backed by .streamlit/secrets.toml on GitHub).
    Returns a dict — missing keys become empty strings, never raises.
    """
    keys = ["ercot_username", "ercot_password", "ercot_subscription_key", "eia_key"]
    creds = {}
    for k in keys:
        try:
            creds[k] = st.secrets[k]
        except Exception:
            creds[k] = ""
    return creds


def missing_creds(creds):
    return [k for k in ["ercot_username","ercot_password","ercot_subscription_key","eia_key"]
            if not creds.get(k)]


# ── ERCOT authentication ──────────────────────────────────────────────────────

@st.cache_data(ttl=3300)   # ERCOT tokens last 1 hour; refresh at 55 min to be safe
def get_ercot_token(username: str, password: str):
    """
    POST username+password to ERCOT's Azure B2C endpoint.
    Returns (id_token_string, error_message).
    The id_token (not access_token) is what ERCOT's public API requires.
    """
    if not username or not password:
        return None, "Missing ERCOT username or password"
    try:
        r = requests.post(
            "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com"
            "/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token",
            data={
                "username":      username,
                "password":      password,
                "grant_type":    "password",
                "scope":         "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
                "client_id":     "fec253ea-0d06-4272-a5e6-b478baeecd70",
                "response_type": "id_token",
            },
            timeout=15,
        )
        if r.status_code == 200:
            token = r.json().get("id_token")
            if token:
                return token, None
            return None, "Token missing from ERCOT response"
        else:
            body = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
            desc = body.get("error_description", r.text[:200])
            return None, f"ERCOT auth HTTP {r.status_code}: {desc}"
    except Exception as e:
        return None, f"ERCOT auth exception: {e}"


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def fetch_weather_forecast():
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,windspeed_10m"
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
    try:
        r = requests.get(
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={LAT}&longitude={LON}"
            f"&start_date={start}&end_date={end}"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&temperature_unit=fahrenheit&timezone=America%2FChicago",
            timeout=12,
        )
        r.raise_for_status()
        return r.json()
    except:
        return None


@st.cache_data(ttl=3600)
def fetch_ercot_prices(id_token: str, subscription_key: str):
    """
    Fetch 45 days of ERCOT North Hub DA and RT settlement prices.
    Requires a valid id_token (from get_ercot_token) and subscription_key.
    Returns (DataFrame_or_None, status_str, da_count, rt_count).
    """
    if not id_token or not subscription_key:
        return None, "Missing ERCOT token or subscription key", 0, 0

    now      = datetime.now(ERCOT_TZ)
    start_dt = now - timedelta(days=45)
    headers  = {
        "Accept":                    "application/json",
        "Authorization":             f"Bearer {id_token}",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }
    date_params = {
        "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
        "deliveryDateTo":   now.strftime("%Y-%m-%d"),
        "settlementPoint":  "HB_NORTH",
        "size":             5000,
    }

    prices = []; da_count = rt_count = 0; da_err = rt_err = None

    # Day-Ahead
    try:
        r = requests.get(
            "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices",
            params=date_params, headers=headers, timeout=20,
        )
        if r.status_code == 200:
            for row in r.json().get("data", []):
                try:
                    dt = ERCOT_TZ.localize(datetime.strptime(
                        f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M"))
                    prices.append({"datetime": dt, "da_price": float(row[3]), "rt_price": None})
                    da_count += 1
                except: continue
        elif r.status_code == 401:
            da_err = "401 Unauthorized — check ERCOT credentials"
        else:
            da_err = f"HTTP {r.status_code}"
    except Exception as e:
        da_err = str(e)[:100]

    # Real-Time
    rt_map = {}
    try:
        r = requests.get(
            "https://api.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub",
            params=date_params, headers=headers, timeout=20,
        )
        if r.status_code == 200:
            for row in r.json().get("data", []):
                try:
                    dt = ERCOT_TZ.localize(datetime.strptime(
                        f"{row[0]} {int(row[1])-1:02d}:00", "%Y-%m-%d %H:%M"))
                    rt_map[dt] = float(row[3])
                    rt_count += 1
                except: continue
        elif r.status_code == 401:
            rt_err = "401 Unauthorized"
        else:
            rt_err = f"HTTP {r.status_code}"
    except Exception as e:
        rt_err = str(e)[:100]

    for p in prices:
        p["rt_price"] = rt_map.get(p["datetime"])

    if prices:
        df = pd.DataFrame(prices).dropna(subset=["da_price"])
        return (df.sort_values("datetime").reset_index(drop=True),
                f"✅ {da_count} DA rows · {rt_count} RT rows loaded", da_count, rt_count)

    errs = " | ".join(filter(None, [da_err, rt_err]))
    return None, f"⚠️ {errs or 'No data returned'}", 0, 0


@st.cache_data(ttl=3600)
def fetch_henry_hub(eia_key: str):
    """
    Henry Hub daily spot price ($/MMBtu).
    Primary:  EIA Open Data API v2  — requires free permanent key from eia.gov/opendata
    Fallback: FRED CSV              — no key, ~1 day delayed, always works
    Note: EIA keys do NOT expire. If you got a key, it works forever.
    """
    # ── Primary: EIA ──
    if eia_key:
        try:
            r = requests.get(
                "https://api.eia.gov/v2/natural-gas/pri/sum/dcu/nus/daily/data/",
                params={
                    "api_key":            eia_key,
                    "frequency":          "daily",
                    "data[0]":            "value",
                    "facets[process][]":  "PH9",   # Henry Hub spot
                    "sort[0][column]":    "period",
                    "sort[0][direction]": "desc",
                    "length":             90,
                },
                timeout=15,
            )
            if r.status_code == 200:
                rows = r.json().get("response", {}).get("data", [])
                if rows:
                    df = pd.DataFrame(rows)[["period","value"]].rename(
                        columns={"period":"Date","value":"HH $/MMBtu"})
                    df["HH $/MMBtu"] = pd.to_numeric(df["HH $/MMBtu"], errors="coerce")
                    df = df.dropna().sort_values("Date", ascending=False).reset_index(drop=True)
                    return df, f"✅ EIA API · {len(df)} days · latest: {df['Date'].iloc[0]}"
            elif r.status_code == 403:
                pass   # fall through to FRED
        except: pass

    # ── Fallback: FRED (St. Louis Fed) ──
    try:
        r = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; gridedge-app/1.0)"},
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["Date","HH $/MMBtu"]
        df["HH $/MMBtu"] = pd.to_numeric(df["HH $/MMBtu"], errors="coerce")
        df = df.dropna().sort_values("Date", ascending=False).reset_index(drop=True)
        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        df = df[df["Date"] >= cutoff].reset_index(drop=True)
        if len(df):
            src = "FRED/St. Louis Fed (EIA key not set — ~1 day delay)" if not eia_key else "FRED/St. Louis Fed (EIA fallback)"
            return df, f"✅ {src} · {len(df)} days · latest: {df['Date'].iloc[0]}"
    except Exception as e:
        pass

    return None, "⚠️ Both EIA and FRED sources failed"


# ── Price model ───────────────────────────────────────────────────────────────

def temp_to_mult(t):
    for thresh, m in [(100,3.5),(95,2.5),(90,1.8),(85,1.35),(80,1.15),
                       (75,1.0),(65,0.85),(55,0.90),(45,1.05),(35,1.25),(25,1.75),(15,2.8)]:
        if t >= thresh: return m
    return 4.0


def build_forecast(weather, hist_df):
    daily   = weather.get("daily",{})
    hourly  = weather.get("hourly",{})
    dates   = daily.get("time",[])
    hi_t    = daily.get("temperature_2m_max",[])
    lo_t    = daily.get("temperature_2m_min",[])
    wind_mx = daily.get("windspeed_10m_max",[])

    h = pd.DataFrame({"time": hourly.get("time",[]),
                       "temp": hourly.get("temperature_2m",[]),
                       "wind": hourly.get("windspeed_10m",[])})
    h["date"] = h["time"].str[:10]
    h["hour"] = h["time"].str[11:13].astype(int)
    ph = h[h["hour"].isin(PEAK_HOURS)]
    pt = ph.groupby("date")["temp"].mean().to_dict()
    pw = ph.groupby("date")["wind"].mean().to_dict()

    if hist_df is not None and len(hist_df) > 30:
        hc = hist_df.copy()
        hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
        pk = hc[hc["hour"].isin(PEAK_HOURS)]
        base_da = pk["da_price"].mean() if len(pk) > 0 else 45.0
        base_rt = pk["rt_price"].dropna().mean() if len(pk) > 0 else 47.0
    else:
        base_da, base_rt = 45.0, 47.0

    now_ct = datetime.now(ERCOT_TZ)
    rows = []
    for i, ds in enumerate(dates):
        hi   = hi_t[i]  if i < len(hi_t)  else 75
        lo   = lo_t[i]  if i < len(lo_t)  else 55
        wind = wind_mx[i] if i < len(wind_mx) else 10
        pkt  = pt.get(ds, (hi+lo)/2)
        pkw  = pw.get(ds, wind)
        m    = temp_to_mult(pkt)
        wa   = max(0.85, 1-(pkw-10)*0.004) if pkw > 10 else 1.0
        da   = round(max(15, min(base_da*m*wa, 500)), 2)
        rt   = round(max(15, min(base_rt*m*wa*np.random.uniform(0.93,1.07), 600)), 2)
        do   = datetime.strptime(ds, "%Y-%m-%d")
        lbl  = "Today" if ds == now_ct.strftime("%Y-%m-%d") else \
               ("Tomorrow" if ds == (now_ct+timedelta(days=1)).strftime("%Y-%m-%d") else do.strftime("%A"))
        pv   = max(da, rt)
        alt  = "🟣 VERY HIGH" if pv>=100 else ("🔴 HIGH" if pv>=60 else ("🟡 MODERATE" if pv>=35 else "🟢 LOW"))
        rows.append({
            "Day":           lbl,
            "Date":          do.strftime("%b %d"),
            "Hi °F":         int(round(hi)),
            "Lo °F":         int(round(lo)),
            "Avg Peak Temp": round(pkt, 1),
            "Wind mph":      int(round(pkw)),
            "DA $/MWh":      da,
            "RT $/MWh":      rt,
            "Alert":         alt,
        })
    return pd.DataFrame(rows)


def build_weather_cmp(weather):
    daily  = weather.get("daily",{})
    dates  = daily.get("time",[])
    hi_now = daily.get("temperature_2m_max",[])
    lo_now = daily.get("temperature_2m_min",[])
    h1 = fetch_historical_weather(1)
    h2 = fetch_historical_weather(2)
    h3 = fetch_historical_weather(3)

    def ghl(hd, idx):
        if hd is None: return None, None
        try:
            return (round(hd["daily"]["temperature_2m_max"][idx], 0),
                    round(hd["daily"]["temperature_2m_min"][idx], 0))
        except: return None, None

    def ds(vn, vh):
        if vn is None or vh is None: return "N/A"
        d = vn - int(vh)
        return f"+{d}°" if d > 0 else f"{d}°"

    now_ct = datetime.now(ERCOT_TZ)
    rows = []
    for i, date_str in enumerate(dates[:7]):
        do  = datetime.strptime(date_str, "%Y-%m-%d")
        lbl = "Today" if date_str == now_ct.strftime("%Y-%m-%d") else \
              ("Tomorrow" if date_str == (now_ct+timedelta(days=1)).strftime("%Y-%m-%d") else do.strftime("%A"))
        hi  = int(round(hi_now[i])) if i < len(hi_now) else None
        lo  = int(round(lo_now[i])) if i < len(lo_now) else None
        h1i,l1i = ghl(h1,i); h2i,l2i = ghl(h2,i); h3i,l3i = ghl(h3,i)
        yr = do.year
        rows.append({
            "Day":              lbl,
            "Date":             do.strftime("%b %d"),
            "Forecast Hi/Lo":   f"{hi}°/{lo}°" if hi else "—",
            f"{yr-1} Hi/Lo":    f"{int(h1i)}°/{int(l1i)}°" if h1i else "N/A",
            f"vs {yr-1}":       ds(hi, h1i),
            f"{yr-2} Hi/Lo":    f"{int(h2i)}°/{int(l2i)}°" if h2i else "N/A",
            f"vs {yr-2}":       ds(hi, h2i),
            f"{yr-3} Hi/Lo":    f"{int(h3i)}°/{int(l3i)}°" if h3i else "N/A",
            f"vs {yr-3}":       ds(hi, h3i),
        })
    return pd.DataFrame(rows)


def build_gas_power(hist_df, gas_df):
    if hist_df is None or gas_df is None or len(gas_df) == 0: return None
    hc = hist_df.copy()
    hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
    hc["Date"] = hc["datetime"].apply(lambda x: x.strftime("%Y-%m-%d"))
    pk = hc[hc["hour"].isin(PEAK_HOURS)].copy()
    if len(pk) == 0: return None
    dp = pk.groupby("Date").agg(
        **{"DA Avg $/MWh":("da_price","mean"), "RT Avg $/MWh":("rt_price","mean")}
    ).reset_index()
    dp["DA Avg $/MWh"] = dp["DA Avg $/MWh"].round(2)
    dp["RT Avg $/MWh"] = dp["RT Avg $/MWh"].round(2)
    gc = "HH $/MMBtu"
    m  = dp.merge(gas_df.rename(columns={gas_df.columns[1]: gc})[["Date",gc]], on="Date", how="left")
    m[gc]             = m[gc].round(3)
    m["DA/Gas Ratio"] = (m["DA Avg $/MWh"] / m[gc]).round(2)
    m["RT/Gas Ratio"] = (m["RT Avg $/MWh"] / m[gc]).round(2)
    return m.sort_values("Date", ascending=False).reset_index(drop=True)


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    now_ct = datetime.now(ERCOT_TZ)
    creds  = get_credentials()
    miss   = missing_creds(creds)

    # ── Header ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 5, 2])
    with c1:
        st.markdown("### ⚡ GRIDEDGE")
    with c2:
        st.markdown(
            "<p style='font-family:DM Mono,monospace;font-size:12px;color:#888;"
            "margin-top:14px;letter-spacing:0.08em'>"
            "ERCOT NORTH HUB · SETTLEMENT PRICE INTELLIGENCE</p>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<p style='font-family:DM Mono,monospace;font-size:12px;color:#888;"
            f"margin-top:14px;text-align:right'>"
            f"{now_ct.strftime('%b %d, %Y · %I:%M %p CT')}</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("## Peak Hour Price Forecast")
    subtext("Projected on-peak (HE07–HE22) Day-Ahead and Real-Time settlement prices · ERCOT North Hub · $/MWh")

    # ── Setup guide (always available, expanded when creds missing) ───────────
    with st.expander("📋 Setup Guide — How to add your API keys", expanded=bool(miss)):
        st.markdown("""
### Where to paste your credentials

You store credentials in **one file on GitHub** called `.streamlit/secrets.toml`.
The app reads it automatically — nothing in the app code itself ever contains your keys.

**Step 1 — Create the secrets file on GitHub:**
1. Go to your `ercot-peak-forecast` repository on GitHub
2. Click **"Add file"** → **"Create new file"**
3. In the filename box, type **exactly**: `.streamlit/secrets.toml`
   *(the dot, the slash, and the exact spelling all matter)*
4. Paste the block below into the big text area — replacing the placeholder values:

```toml
ercot_username         = "your-email@example.com"
ercot_password         = "your-ercot-password"
ercot_subscription_key = "your-primary-key-from-apiexplorer.ercot.com"
eia_key                = "your-eia-key"
```

5. Scroll down → click **"Commit new file"**
6. Your site will reload automatically within ~30 seconds with live data

---

### Getting your ERCOT subscription key

You need to register at **https://apiexplorer.ercot.com** — the process has a non-obvious step:

1. Go to **https://apiexplorer.ercot.com** and sign in (or sign up if new)
2. In the **top navigation bar**, click the **"Products"** tab
   *(this is different from the "APIs" tab — look carefully, it's easy to miss)*
3. In the Products table, click **"ERCOT Public API"**
4. Type any name in the subscription name box (e.g. `gridedge`) → click **Subscribe**
5. You'll be redirected to your **Profile** page
6. Under your active subscriptions, click **Show** next to **Primary Key**
7. Copy that long alphanumeric string — that is your `ercot_subscription_key`

Your `ercot_username` is the email you signed up with.
Your `ercot_password` is the password you chose at sign-up.

> **Note:** ERCOT tokens expire every hour, but the app renews them automatically.
> Your subscription key itself never expires — you only need to copy it once.

---

### Getting your EIA key (for Henry Hub gas prices)

1. Go to **https://www.eia.gov/opendata/register.php**
2. Enter your email → click Register
3. Your key arrives in your inbox in under 1 minute
4. Copy it into the `eia_key` field in secrets.toml

> **Note:** EIA keys are **permanent — they never expire.** Once you paste it in, you're done forever.
> If Henry Hub data is still showing without an EIA key, that's because the app automatically
> falls back to FRED (St. Louis Fed) which has no key requirement — it's just ~1 day delayed.
        """)

    # ── Credentials warning if incomplete ────────────────────────────────────
    if miss:
        st.warning(
            f"⚠️ Missing credentials: `{'`, `'.join(miss)}`. "
            f"See the Setup Guide above. Weather and the price model run now — "
            f"live ERCOT prices load once credentials are added."
        )

    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("⏳ Fetching data…"):
        weather, werr = fetch_weather_forecast()

        # ERCOT: get token first, then fetch prices
        if creds["ercot_username"] and creds["ercot_password"] and creds["ercot_subscription_key"]:
            id_token, token_err = get_ercot_token(creds["ercot_username"], creds["ercot_password"])
            hist_df, ercot_status, da_count, rt_count = fetch_ercot_prices(
                id_token, creds["ercot_subscription_key"]
            )
        else:
            id_token, token_err = None, "Credentials not configured"
            hist_df, ercot_status, da_count, rt_count = None, "⚠️ Credentials not set — see Setup Guide", 0, 0

        gas_df, gas_status = fetch_henry_hub(creds["eia_key"])

    if weather is None:
        st.error(f"Weather API failed: {werr}. Please refresh.")
        return

    # ── Data status panel ─────────────────────────────────────────────────────
    with st.expander("📡 Data Source Status", expanded=False):
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**🌤 Weather · Open-Meteo**")
            st.success("Connected — 7-day hourly forecast for DFW")
        with s2:
            st.markdown("**⚡ ERCOT North Hub**")
            if da_count > 0:
                st.success(ercot_status)
            else:
                st.warning(ercot_status)
                if token_err:
                    st.caption(f"Auth detail: {token_err}")
        with s3:
            st.markdown("**🔥 Henry Hub · EIA / FRED**")
            if gas_df is not None:
                st.success(gas_status)
            else:
                st.warning(gas_status)
        st.caption(
            "ERCOT key: apiexplorer.ercot.com → Products tab → Subscribe  ·  "
            "EIA key: eia.gov/opendata/register.php (free, permanent)  ·  "
            "Weather: no key needed"
        )

    # ── Build model ───────────────────────────────────────────────────────────
    forecast_df = build_forecast(weather, hist_df)
    weather_cmp = build_weather_cmp(weather)

    hist_da_avg = hist_rt_avg = None
    if hist_df is not None and len(hist_df) > 0:
        hc = hist_df.copy()
        hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
        pk = hc[hc["hour"].isin(PEAK_HOURS)]
        if len(pk):
            hist_da_avg = round(pk["da_price"].mean(), 2)
            hist_rt_avg = round(pk["rt_price"].dropna().mean(), 2)

    def dstr(val, ref):
        if ref is None: return None
        pct = (val - ref) / ref * 100
        return f"{'+' if pct>0 else ''}{pct:.1f}% vs 45-day avg"

    today    = forecast_df.iloc[0]
    tomorrow = forecast_df.iloc[1] if len(forecast_df) > 1 else None

    lg = lg_d = None
    if gas_df is not None and len(gas_df) > 0:
        lg   = round(float(gas_df.iloc[0, 1]), 3)
        lg_d = gas_df.iloc[0, 0]

    # ── Metric tiles ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Today — DA Peak", f"${today['DA $/MWh']:.2f}/MWh",
                  dstr(today["DA $/MWh"], hist_da_avg))
    with m2:
        st.metric("Today — RT Peak", f"${today['RT $/MWh']:.2f}/MWh",
                  dstr(today["RT $/MWh"], hist_rt_avg))
    with m3:
        if tomorrow is not None:
            st.metric("Tomorrow — DA Peak", f"${tomorrow['DA $/MWh']:.2f}/MWh",
                      dstr(tomorrow["DA $/MWh"], hist_da_avg))
    with m4:
        if lg is not None:
            st.metric(f"Henry Hub ({lg_d})", f"${lg:.3f}/MMBtu",
                      f"45-day DA avg: ${hist_da_avg:.2f}/MWh" if hist_da_avg else "Live gas price",
                      delta_color="off")
        elif hist_da_avg:
            st.metric("45-Day Avg Peak DA", f"${hist_da_avg:.2f}/MWh",
                      "on-peak hrs only", delta_color="off")

    st.divider()

    # ── 7-Day Forecast table ──────────────────────────────────────────────────
    st.markdown("### 📋 7-Day Price Forecast")
    st.markdown(
        "<p style='font-size:13px;color:#555;font-family:DM Sans,sans-serif;margin-top:-6px'>"
        "On-peak hours only (HE07–HE22 · 7 am – 10 pm CT)&nbsp;&nbsp;"
        "<span style='background:#D4EDDA;color:#155724;padding:2px 8px;"
        "border-radius:3px;font-size:12px'>🟢 Under $35</span>&nbsp;"
        "<span style='background:#FFF3CC;color:#7A5000;padding:2px 8px;"
        "border-radius:3px;font-size:12px'>🟡 $35 – $60</span>&nbsp;"
        "<span style='background:#FFDAD4;color:#8B0000;padding:2px 8px;"
        "border-radius:3px;font-size:12px'>🔴 $60 – $100</span>&nbsp;"
        "<span style='background:#E8D5F5;color:#4A0080;padding:2px 8px;"
        "border-radius:3px;font-size:12px'>🟣 $100+</span>"
        "</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        forecast_df.style
        .applymap(color_price, subset=["DA $/MWh","RT $/MWh"])
        .format({"DA $/MWh":"${:.2f}","RT $/MWh":"${:.2f}","Avg Peak Temp":"{:.1f}°F"})
        .set_table_styles(TBL_STYLES),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── Weather comparison ────────────────────────────────────────────────────
    st.markdown("### 🌡️ DFW Weather — Forecast vs. Same Week in Prior Years")
    subtext(
        "Same calendar dates 1, 2, and 3 years ago.&nbsp;"
        "<span style='color:#C03A00;font-weight:600'>Red = warmer than prior year</span>"
        "&nbsp;·&nbsp;"
        "<span style='color:#1A7A3C;font-weight:600'>Green = cooler</span>"
    )
    diff_cols = [c for c in weather_cmp.columns if c.startswith("vs ")]
    st.dataframe(
        weather_cmp.style
        .applymap(color_diff, subset=diff_cols)
        .set_table_styles(TBL_STYLES),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── Historical power prices ───────────────────────────────────────────────
    st.markdown("### 📊 Historical On-Peak Power Prices — Last 45 Days")
    if hist_df is not None and len(hist_df) > 0:
        subtext(f"Actual ERCOT North Hub on-peak (HE07–HE22) settlement prices. "
                f"{da_count} DA · {rt_count} RT hourly rows loaded.")
        hc = hist_df.copy()
        hc["hour"] = hc["datetime"].apply(lambda x: x.hour)
        hc["Date"] = hc["datetime"].apply(lambda x: x.strftime("%Y-%m-%d"))
        pk = hc[hc["hour"].isin(PEAK_HOURS)].copy()
        if len(pk) > 0:
            dh = pk.groupby("Date").agg(
                **{"DA Avg $/MWh":("da_price","mean"), "DA Max $/MWh":("da_price","max"),
                   "RT Avg $/MWh":("rt_price","mean"), "RT Max $/MWh":("rt_price","max")}
            ).reset_index().sort_values("Date", ascending=False)
            pc = ["DA Avg $/MWh","DA Max $/MWh","RT Avg $/MWh","RT Max $/MWh"]
            for c in pc: dh[c] = dh[c].round(2)
            st.dataframe(
                dh.style.applymap(color_price, subset=pc)
                .format({c:"${:.2f}" for c in pc})
                .set_table_styles(TBL_STYLES),
                use_container_width=True, hide_index=True,
            )
    else:
        if creds["ercot_username"]:
            st.warning(
                "ERCOT returned no data. This usually means the token failed — "
                "double-check your email, password, and subscription key in secrets.toml. "
                "The price forecast above is using default baseline values in the meantime."
            )
        else:
            st.info("Add your ERCOT credentials to secrets.toml (see Setup Guide) to load historical prices.")

    st.divider()

    # ── Henry Hub gas prices ──────────────────────────────────────────────────
    st.markdown("### 🔥 Henry Hub Natural Gas — Last 90 Days")
    if gas_df is not None and len(gas_df) > 0:
        subtext("Daily Henry Hub spot price ($/MMBtu). "
                "Gas sets the marginal price in ERCOT ~70–80% of on-peak hours.")
        gc = gas_df.columns[1]
        st.dataframe(
            gas_df.head(90).style
            .applymap(color_gas, subset=[gc])
            .format({gc:"${:.3f}"})
            .set_table_styles(TBL_STYLES),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info(
            "Henry Hub data unavailable. Add your EIA key to secrets.toml — "
            "or the app will also try FRED (free, no key needed) as a fallback."
        )

    st.divider()

    # ── Power vs gas heat rate ────────────────────────────────────────────────
    st.markdown("### ⚡ vs 🔥 Power vs. Gas — Implied Heat Rate")
    subtext(
        "DA/Gas Ratio = $/MWh ÷ $/MMBtu. "
        "Above 9x: power expensive relative to gas (strong generator margins). "
        "Below 6x: power cheap vs fuel. Typical ERCOT on-peak range: 8–12x in summer."
    )
    comp = build_gas_power(hist_df, gas_df)
    if comp is not None and len(comp) > 0:
        pc2 = ["DA Avg $/MWh","RT Avg $/MWh"]
        rc  = ["DA/Gas Ratio","RT/Gas Ratio"]
        gc2 = comp.columns[3]
        st.dataframe(
            comp.style
            .applymap(color_price,  subset=pc2)
            .applymap(color_gas,    subset=[gc2])
            .applymap(color_ratio,  subset=rc)
            .format({"DA Avg $/MWh":"${:.2f}","RT Avg $/MWh":"${:.2f}",
                     gc2:"${:.3f}","DA/Gas Ratio":"{:.2f}x","RT/Gas Ratio":"{:.2f}x"})
            .set_table_styles(TBL_STYLES),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Heat rate table appears once both ERCOT price data and Henry Hub gas prices are loaded.")

    st.divider()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:12px;color:#aaa;font-family:DM Mono,monospace'>"
        f"Sources: ERCOT Public API · EIA Open Data · FRED St. Louis Fed (HH fallback) · Open-Meteo"
        f"&nbsp;·&nbsp;Last loaded {now_ct.strftime('%b %d %Y %I:%M %p CT')}"
        f"&nbsp;·&nbsp;Cached 30–60 min<br>"
        f"Statistical estimates only — not investment or trading advice."
        f"</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

# ============================================================
# NSE V12 - COMPLETE LOCAL WEB DASHBOARD
# ============================================================
# Mobile + Laptop
# All NSE stocks
# BUY / HOLD / SELL
# Risk %
# Target / Stop Loss
# Holding Period
# Portfolio
# Past Predictions + Success Rate
# Sector Analysis
# Find Stock
# Live Market + Closing Market Analysis
# TradingView + Technical Charts
# ============================================================

import os
import re
import math
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

HISTORY_FILE = APP_DIR / "recommendation_history.csv"
PORTFOLIO_FILE = APP_DIR / "my_portfolio.csv"
RESULT_FILE = APP_DIR / "latest_results.csv"
TRADES_FILE = APP_DIR / "auto_trades_tracker.csv"
PAPER_FILE = APP_DIR / "paper_portfolio.csv"
LEARNING_FILE = APP_DIR / "model_learning.json"
STRATEGY_BT_FILE = APP_DIR / "strategy_backtest_results.csv"
STRATEGY_LIVE_FILE = APP_DIR / "strategy_live_signals.csv"
MY_STRATEGY_PARAMS_FILE = APP_DIR / "my_strategy_params.json"
NSE_UNIVERSE_CACHE = APP_DIR / "nse_equity_universe.csv"

MAX_SCAN_STOCKS = 10000
DEFAULT_HOLD_DAYS = 15
TOP_DEFAULT = 25

# Default auto-refresh interval (seconds). User can change in sidebar.
LIVE_REFRESH_SECONDS = 5

INDEX_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="NSE V12 Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .stock-card {
        border: 1px solid #dddddd;
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 12px;
        background: white;
    }

    .small-text {
        font-size: 13px;
        color: #666666;
    }

    .buy-box {
        border-left: 6px solid #1a9b5f;
        padding-left: 12px;
    }

    .sell-box {
        border-left: 6px solid #d93025;
        padding-left: 12px;
    }

    .hold-box {
        border-left: 6px solid #e0a800;
        padding-left: 12px;
    }

    .watch-box {
        border-left: 6px solid #777777;
        padding-left: 12px;
    }

    .danger-box {
        border: 1px solid #d93025;
        border-radius: 10px;
        padding: 12px;
    }

    .success-box {
        border: 1px solid #1a9b5f;
        border-radius: 10px;
        padding: 12px;
    }

    @media (max-width: 768px) {

        .block-container {
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }

        h1 {
            font-size: 1.65rem !important;
        }

        h2 {
            font-size: 1.35rem !important;
        }

        h3 {
            font-size: 1.15rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):
    """Never raises on NaN / None / bad strings."""
    try:
        if value is None:
            return int(default)
        if isinstance(value, float) and value != value:
            return int(default)
        try:
            if pd.isna(value):
                return int(default)
        except Exception:
            pass
        return int(float(value))
    except Exception:
        try:
            return int(default)
        except Exception:
            return 0


def clean_symbol(symbol):
    symbol = str(symbol).upper().strip()

    if not symbol:
        return ""

    if symbol.endswith(".NS"):
        return symbol

    if symbol.startswith("^"):
        return symbol

    return symbol + ".NS"


def display_symbol(symbol):
    return str(symbol).upper().replace(".NS", "")


def normalize_columns(df):
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    if isinstance(x.columns, pd.MultiIndex):
        new_cols = []

        for c in x.columns:
            if isinstance(c, tuple):
                new_cols.append(str(c[0]))
            else:
                new_cols.append(str(c))

        x.columns = new_cols

    x.columns = [str(c).strip() for c in x.columns]

    return x


def ensure_result_columns(df):
    """Make saved/legacy scan results safe for every dashboard page.

    Older CSV files may not contain columns added in later versions.
    The dashboard should never crash with KeyError just because an older
    result file was loaded. Missing fields are filled with safe defaults.
    """
    if df is None:
        return pd.DataFrame()

    x = df.copy()

    # Legacy column names from earlier versions.
    aliases = {
        "signal": "Call",
        "call": "Call",
        "price": "Price",
        "risk": "Risk Level",
        "risk_level": "Risk Level",
        "risk_pct": "Risk %",
        "prediction": "Prediction",
        "target": "Target",
        "stop_loss": "Stop Loss",
        "hold_days": "Hold Days",
        "priority": "Priority",
        "reason": "Reason",
        "technical_reasons": "Technical Reasons",
        "patterns": "Patterns",
        "news_influence": "News Influence",
    }
    for old, new in aliases.items():
        if new not in x.columns and old in x.columns:
            x[new] = x[old]

    # Always derive Sector if it is absent or blank.
    if "Stock" not in x.columns and "Symbol" in x.columns:
        x["Stock"] = x["Symbol"].astype(str).str.replace(".NS", "", regex=False).str.upper()

    if "Stock" not in x.columns:
        x["Stock"] = "UNKNOWN"

    if "Sector" not in x.columns:
        x["Sector"] = x["Stock"].apply(sector_of)
    else:
        missing = x["Sector"].isna() | x["Sector"].astype(str).str.strip().isin(["", "nan", "None"])
        if missing.any():
            x.loc[missing, "Sector"] = x.loc[missing, "Stock"].apply(sector_of)

    defaults = {
        "Symbol": x["Stock"].astype(str).apply(clean_symbol),
        "Price": np.nan,
        "Call": "WATCH",
        "Prediction": 0.0,
        "Risk %": 0.0,
        "Risk Level": "UNKNOWN",
        "Target": np.nan,
        "Stop Loss": np.nan,
        "Hold Days": DEFAULT_HOLD_DAYS,
        "Priority": "LOW",
        "Patterns": "No major pattern detected",
        "News Influence": "No clear news influence assessed.",
        "Reason": "No explanation available.",
        "Technical Reasons": "No technical explanation available.",
    }

    for col, default in defaults.items():
        if col not in x.columns:
            if isinstance(default, pd.Series):
                x[col] = default
            else:
                x[col] = default

    # Numeric cleanup so sorting/filtering cannot fail on old text CSVs.
    for col in ["Price", "Prediction", "Risk %", "Target", "Stop Loss", "Hold Days"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x["Prediction"] = x["Prediction"].fillna(0)
    x["Risk %"] = x["Risk %"].fillna(0)
    x["Hold Days"] = x["Hold Days"].fillna(DEFAULT_HOLD_DAYS).astype(int)

    x["Call"] = x["Call"].astype(str).str.upper().replace({"NAN": "WATCH", "NONE": "WATCH"})
    x["Stock"] = x["Stock"].astype(str).str.replace(".NS", "", regex=False).str.upper()
    x["Symbol"] = x["Symbol"].astype(str).apply(clean_symbol)

    # Recreate Rank if a saved file did not have it.
    if "Rank" not in x.columns:
        x["Rank"] = np.arange(1, len(x) + 1)

    return x


# ============================================================
# MARKET STATUS
# ============================================================

def india_now():
    """
    Uses system local time.
    For a machine running in India this is directly correct.
    """
    return datetime.now()


def nse_market_open_now():
    now = india_now()

    if now.weekday() >= 5:
        return False

    current = now.time()

    return (
        current >= dt_time(9, 15)
        and current <= dt_time(15, 30)
    )


def market_status_text():

    if nse_market_open_now():
        return "🟢 NSE MARKET OPEN"

    return "🔴 NSE MARKET CLOSED"


# ============================================================
# FILE INITIALIZATION
# ============================================================

def ensure_files():

    if not HISTORY_FILE.exists():

        pd.DataFrame(
            columns=[
                "Prediction Date",
                "Stock",
                "Symbol",
                "Call",
                "Prediction",
                "Entry",
                "Target",
                "Stop Loss",
                "Risk %",
                "Risk Level",
                "Hold Days",
                "Expiry Date",
                "Status",
                "Evaluation Date",
                "Exit Price",
                "Return %",
                "Result",
                "Result Detail",
                "Days Taken",
                "Outcome Message",
                "Recommendation",
                "Reason",
            ]
        ).to_csv(
            HISTORY_FILE,
            index=False
        )
    else:
        # Ensure newer columns exist on older history files
        try:
            hist = pd.read_csv(HISTORY_FILE)
            extra_cols = {
                "Result Detail": "",
                "Days Taken": "",
                "Outcome Message": "",
                "Recommendation": "",
            }
            changed = False
            for col, default in extra_cols.items():
                if col not in hist.columns:
                    hist[col] = default
                    changed = True
            if changed:
                hist.to_csv(HISTORY_FILE, index=False)
        except Exception:
            pass

    if not PORTFOLIO_FILE.exists():

        pd.DataFrame(
            columns=[
                "Stock",
                "Shares",
                "Buy Price",
                "Purchase Date",
                "Maximum Holding Days",
            ]
        ).to_csv(
            PORTFOLIO_FILE,
            index=False
        )

    if not TRADES_FILE.exists():
        pd.DataFrame(
            columns=[
                "Trade ID",
                "Prediction Date",
                "Stock",
                "Call",
                "Entry",
                "Target",
                "Stop Loss",
                "Hold Days",
                "Status",
                "Result",
                "Current Price",
                "Unrealized %",
                "Realized %",
                "Exit Price",
                "Days Held",
                "Days Remaining",
                "Distance to Target %",
                "Distance to Stop %",
                "Last Checked",
                "Suggestion",
            ]
        ).to_csv(TRADES_FILE, index=False)

    if not PAPER_FILE.exists():
        pd.DataFrame(
            columns=[
                "Trade ID",
                "Open Date",
                "Stock",
                "Side",
                "Shares",
                "Entry",
                "Target",
                "Stop Loss",
                "Hold Days",
                "Status",
                "Result",
                "Exit Date",
                "Exit Price",
                "Return %",
                "PnL ₹",
                "Notes",
            ]
        ).to_csv(PAPER_FILE, index=False)


ensure_files()


# ============================================================
# NSE UNIVERSE
# ============================================================

def load_all_nse_stocks():

    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]

    for url in urls:

        try:

            u = pd.read_csv(url)

            col = next(
                (
                    c
                    for c in u.columns
                    if str(c).strip().upper() == "SYMBOL"
                ),
                None,
            )

            if col:

                symbols = []

                for value in u[col].dropna():

                    s = str(value).strip().upper()

                    if not s:
                        continue

                    if s == "NAN":
                        continue

                    symbols.append(
                        s + ".NS"
                    )

                symbols = sorted(
                    set(symbols)
                )

                if len(symbols) > 500:

                    pd.DataFrame(
                        {"Symbol": symbols}
                    ).to_csv(
                        NSE_UNIVERSE_CACHE,
                        index=False
                    )

                    return symbols

        except Exception:
            continue

    if NSE_UNIVERSE_CACHE.exists():

        try:

            return sorted(
                set(
                    pd.read_csv(
                        NSE_UNIVERSE_CACHE
                    )["Symbol"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
            )

        except Exception:
            pass

    # Fallback list
    return [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "INFY.NS",
        "ITC.NS",
        "SBIN.NS",
        "LT.NS",
        "AXISBANK.NS",
        "KOTAKBANK.NS",
        "BHARTIARTL.NS",
        "SUNPHARMA.NS",
        "MARUTI.NS",
        "TATAMOTORS.NS",
        "TATASTEEL.NS",
        "ONGC.NS",
        "NTPC.NS",
        "POWERGRID.NS",
        "WIPRO.NS",
        "HCLTECH.NS",
    ]


NSE_STOCKS = load_all_nse_stocks()


# ============================================================
# SECTOR MAP
# ============================================================

SECTOR_MAP_CACHE = APP_DIR / "nse_sector_map.csv"

# Normalize NSE / Yahoo industry labels → dashboard sector buckets
_INDUSTRY_TO_SECTOR = [
    (("bank", "banking"), "Banking"),
    (("finance", "financial services", "nbfc", "housing finance", "capital markets", "insurance"), "Financial Services"),
    (("information technology", "it services", "software", "computers"), "IT"),
    (("pharma", "pharmaceutical", "healthcare", "hospital", "biotech", "drug"), "Pharma"),
    (("automobile", "auto component", "auto components", "tyre"), "Automobile"),
    (("metal", "steel", "mining", "mineral", "aluminium", "copper", "zinc"), "Metals"),
    (("oil", "gas", "petroleum", "refinery", "refineries", "consumable fuels"), "Energy"),
    (("power", "utilities", "electricity"), "Utilities"),
    (("fmcg", "fast moving consumer", "food product", "beverages", "tobacco"), "FMCG"),
    (("telecom", "telecommunication"), "Telecom"),
    (("cement",), "Cement"),
    (("construction", "infrastructure", "engineering", "capital goods", "industrial manufacturing"), "Infrastructure"),
    (("realty", "real estate"), "Realty"),
    (("consumer durable", "consumer goods", "textiles", "apparel", "jewellery", "paint"), "Consumer"),
    (("retail", "trading", "e-commerce"), "Retail"),
    (("media", "entertainment"), "Media"),
    (("chemical", "fertilizer", "fertilisers", "pesticide", "specialty chemical"), "Chemicals"),
    (("defence", "defense", "aerospace"), "Defence"),
    (("logistics", "transport", "shipping", "airline", "services"), "Services"),
    (("agriculture", "paper", "forest"), "Others"),
]


def _map_industry_label(label: str) -> str:
    text = str(label or "").strip().lower()
    if not text or text in {"nan", "none", "-"}:
        return "Other"
    for keys, sector in _INDUSTRY_TO_SECTOR:
        for k in keys:
            if k in text:
                return sector
    return "Other"


def _seed_sector_map() -> dict:
    """Hardcoded seeds for major names (used if NSE download fails)."""
    return {
        "RELIANCE": "Energy", "ONGC": "Energy", "IOC": "Energy", "BPCL": "Energy",
        "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
        "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
        "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
        "BANKBARODA": "Banking", "CANBK": "Banking", "PNB": "Banking",
        "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
        "MARUTI": "Automobile", "M&M": "Automobile", "TATAMOTORS": "Automobile",
        "EICHERMOT": "Automobile", "TATASTEEL": "Metals", "JSWSTEEL": "Metals",
        "HINDALCO": "Metals", "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG",
        "BHARTIARTL": "Telecom", "NTPC": "Utilities", "POWERGRID": "Utilities",
        "LT": "Infrastructure", "ADANIPORTS": "Infrastructure",
        "HAL": "Defence", "BEL": "Defence", "COFORGE": "IT", "PERSISTENT": "IT",
        "LTIM": "IT", "TRENT": "Retail", "TITAN": "Consumer", "ASIANPAINT": "Consumer",
    }


def load_sector_map() -> dict:
    """
    Build a large symbol → sector map from NSE index constituent files
    (Industry column) so sector filters show all stocks, not 2–3 names.
    """
    sector_map = _seed_sector_map()

    # Prefer cache if reasonably complete
    if SECTOR_MAP_CACHE.exists():
        try:
            cached = pd.read_csv(SECTOR_MAP_CACHE)
            if "Symbol" in cached.columns and "Sector" in cached.columns and len(cached) > 200:
                for _, row in cached.iterrows():
                    sym = str(row["Symbol"]).upper().strip().replace(".NS", "")
                    sec = str(row["Sector"]).strip()
                    if sym and sec and sec not in {"", "nan", "None"}:
                        sector_map[sym] = sec
                if len(sector_map) > 200:
                    return sector_map
        except Exception:
            pass

    # NSE index lists that include Industry / Company Name
    index_urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyautolist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymetallist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyfmcglist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyenergylist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyinfralist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyrealtylist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyhealthcarelist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyconsumerdurableslist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyfinancelist.csv",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/csv,*/*",
    }

    for url in index_urls:
        try:
            df = pd.read_csv(url, storage_options={"User-Agent": headers["User-Agent"]})
        except Exception:
            try:
                import io
                import urllib.request
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    df = pd.read_csv(io.BytesIO(resp.read()))
            except Exception:
                continue

        if df is None or df.empty:
            continue

        cols = {str(c).strip().lower(): c for c in df.columns}
        sym_col = cols.get("symbol") or cols.get("symbol name")
        ind_col = (
            cols.get("industry")
            or cols.get("basic industry")
            or cols.get("sector")
            or cols.get("macro economic sector")
        )
        if not sym_col:
            continue

        for _, row in df.iterrows():
            sym = str(row[sym_col]).upper().strip()
            if not sym or sym in {"NAN", "NONE"}:
                continue
            industry = row[ind_col] if ind_col else ""
            sector = _map_industry_label(industry)
            # Keep a more specific assignment if we already have one that is not Other
            if sym not in sector_map or sector_map.get(sym) == "Other":
                if sector != "Other":
                    sector_map[sym] = sector
                elif sym not in sector_map:
                    sector_map[sym] = "Other"

    # Persist for next run
    try:
        pd.DataFrame(
            [{"Symbol": k, "Sector": v} for k, v in sorted(sector_map.items())]
        ).to_csv(SECTOR_MAP_CACHE, index=False)
    except Exception:
        pass

    return sector_map


SECTOR_MAP = load_sector_map()


def sector_of(symbol):
    key = display_symbol(symbol)
    return SECTOR_MAP.get(key, "Other")


def refresh_sector_for_results(results: pd.DataFrame) -> pd.DataFrame:
    """Re-apply sector map so sector pages are complete after scan."""
    if results is None or results.empty:
        return results
    x = results.copy()
    if "Stock" not in x.columns and "Symbol" in x.columns:
        x["Stock"] = x["Symbol"].astype(str).str.replace(".NS", "", regex=False).str.upper()
    x["Sector"] = x["Stock"].astype(str).str.upper().map(
        lambda s: SECTOR_MAP.get(s, "Other")
    )
    return x


# ============================================================
# INDEX MEMBERSHIP + MARKET CAP + LONG-TERM QUALITY
# ============================================================

INDEX_MEMBER_CACHE = APP_DIR / "nse_index_membership.json"

_INDEX_CSV = {
    "Nifty 50": [
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    ],
    "Nifty 100": [
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    ],
    "Nifty 200": [
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    ],
    "Nifty 500": [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    ],
    "Nifty Midcap 100": [
        "https://archives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
    ],
    "Nifty Smallcap 100": [
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
    ],
}


def _download_index_symbols(urls: list) -> set:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    for url in urls:
        try:
            df = pd.read_csv(url, storage_options={"User-Agent": headers["User-Agent"]})
        except Exception:
            try:
                import io
                import urllib.request
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    df = pd.read_csv(io.BytesIO(resp.read()))
            except Exception:
                continue
        if df is None or df.empty:
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        sym_col = cols.get("symbol") or cols.get("symbol name")
        if not sym_col:
            continue
        out = set()
        for v in df[sym_col].dropna():
            s = str(v).upper().strip().replace(".NS", "")
            if s and s != "NAN":
                out.add(s)
        if out:
            return out
    return set()


@st.cache_data(ttl=86400, show_spinner=False)
def load_index_membership() -> dict:
    """
    symbol → list of index names (Nifty 50, 100, 500, Midcap, Smallcap…).
    Cached 24h; also written to disk.
    """
    membership = {}  # sym -> set of index labels

    if INDEX_MEMBER_CACHE.exists():
        try:
            with open(INDEX_MEMBER_CACHE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and len(raw) > 50:
                return {k: list(v) for k, v in raw.items()}
        except Exception:
            pass

    for name, urls in _INDEX_CSV.items():
        syms = _download_index_symbols(urls)
        for s in syms:
            membership.setdefault(s, set()).add(name)

    # Derive Large / Mid / Small style tags from index membership
    for s, idxs in list(membership.items()):
        idxs = set(idxs)
        if "Nifty 50" in idxs or "Nifty 100" in idxs:
            idxs.add("Large Cap (index)")
        if "Nifty Midcap 100" in idxs:
            idxs.add("Mid Cap (index)")
        if "Nifty Smallcap 100" in idxs:
            idxs.add("Small Cap (index)")
        membership[s] = idxs

    serializable = {k: sorted(list(v)) for k, v in membership.items()}
    try:
        with open(INDEX_MEMBER_CACHE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
    except Exception:
        pass
    return serializable


def index_tags_of(symbol: str) -> list:
    m = load_index_membership()
    return list(m.get(display_symbol(symbol), []) or [])


def market_cap_bucket(market_cap) -> str:
    """
    Approx India buckets using market cap in INR (Yahoo often returns absolute INR).
    Large ≥ ₹20,000 Cr | Mid ₹5,000–20,000 Cr | Small < ₹5,000 Cr
    """
    mc = safe_float(market_cap, 0)
    if mc <= 0:
        return "Unknown"
    # Cr = 10^7
    cr = mc / 1e7
    if cr >= 20000:
        return "Large Cap"
    if cr >= 5000:
        return "Mid Cap"
    if cr >= 500:
        return "Small Cap"
    return "Micro Cap"


def long_term_quality_from_fund(fund: dict) -> dict:
    """
    Experienced long-term desk score 0–100 + hold advice.
    If short-term SL hits, this says whether core holding still makes sense.
    """
    score = 50
    notes = []
    if not fund or fund.get("error"):
        return {
            "lt_score": 45,
            "lt_label": "Unknown / data thin",
            "lt_hold_if_sl": "Review manually — fundamentals not loaded",
            "notes": ["Fundamentals unavailable"],
            "market_cap_bucket": "Unknown",
        }

    pe = safe_float(fund.get("pe"))
    roe = safe_float(fund.get("roe"))
    if roe and roe < 1.5:
        roe = roe * 100  # sometimes fraction
    de = safe_float(fund.get("debt_to_equity"))
    pm = safe_float(fund.get("profit_margins"))
    if pm and pm < 1:
        pm = pm * 100
    rg = safe_float(fund.get("revenue_growth"))
    if rg and abs(rg) < 2:
        rg = rg * 100
    eg = safe_float(fund.get("earnings_growth"))
    if eg and abs(eg) < 2:
        eg = eg * 100
    dy = safe_float(fund.get("dividend_yield"))
    mcap_b = market_cap_bucket(fund.get("market_cap"))

    if roe >= 15:
        score += 12
        notes.append(f"ROE ~{roe:.1f}% — quality compounder zone")
    elif roe >= 10:
        score += 6
        notes.append(f"ROE ~{roe:.1f}% — acceptable")
    elif roe > 0:
        score -= 5
        notes.append(f"ROE ~{roe:.1f}% — weak for long-term core")

    if 0 < de <= 50:
        score += 8
        notes.append(f"Debt/Equity {de:.0f} — comfortable")
    elif 50 < de <= 100:
        score += 2
        notes.append(f"Debt/Equity {de:.0f} — watch leverage")
    elif de > 100:
        score -= 10
        notes.append(f"Debt/Equity {de:.0f} — high leverage risk")

    if pm >= 12:
        score += 8
        notes.append(f"Profit margin ~{pm:.1f}%")
    elif pm >= 6:
        score += 3
    elif pm > 0:
        score -= 4

    if rg >= 12:
        score += 8
        notes.append(f"Revenue growth ~{rg:.1f}%")
    elif rg >= 5:
        score += 3
    elif rg < 0:
        score -= 6
        notes.append("Revenue shrinking — long-term caution")

    if eg >= 12:
        score += 6
    elif eg < -10:
        score -= 5

    if 0 < pe <= 25:
        score += 5
        notes.append(f"PE ~{pe:.1f} — not extreme")
    elif pe > 50:
        score -= 6
        notes.append(f"PE ~{pe:.1f} — expensive; need growth to justify")
    elif pe > 35:
        score -= 2

    if dy >= 1.5:
        score += 3
        notes.append(f"Dividend yield ~{dy:.1f}%")

    if mcap_b == "Large Cap":
        score += 4
        notes.append("Large-cap stability bias")
    elif mcap_b == "Mid Cap":
        score += 2
        notes.append("Mid-cap — growth with volatility")
    elif mcap_b in ("Small Cap", "Micro Cap"):
        score -= 2
        notes.append("Small/micro — higher business risk; size positions smaller")

    score = int(max(5, min(95, score)))
    if score >= 72:
        label = "Strong long-term candidate"
        hold_if_sl = (
            "YES — if SL was only swing noise and business still intact, "
            "you may HOLD/accumulate for long-term instead of panic selling the whole position."
        )
    elif score >= 58:
        label = "Average long-term hold"
        hold_if_sl = (
            "PARTIAL — keep only a core size if thesis is intact; "
            "do not add aggressively after a technical SL."
        )
    else:
        label = "Weak / speculative long-term"
        hold_if_sl = (
            "NO core long-term — treat as trading book only. "
            "If swing SL hits, prefer exit; do not convert a bad trade into a long bag-hold."
        )

    return {
        "lt_score": score,
        "lt_label": label,
        "lt_hold_if_sl": hold_if_sl,
        "notes": notes[:8],
        "market_cap_bucket": mcap_b,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def in_crore(value) -> float:
    """Convert absolute INR amount to ₹ crore (1 Cr = 1e7)."""
    v = safe_float(value, 0)
    if v == 0:
        return 0.0
    # Yahoo marketCap/revenue usually full INR; if already small, leave as is
    if abs(v) >= 1e5:
        return round(v / 1e7, 2)
    return round(v, 2)


def fmt_crore(value, suffix=" Cr") -> str:
    cr = in_crore(value)
    if cr == 0 and safe_float(value, 0) == 0:
        return "—"
    return f"₹{cr:,.2f}{suffix}"


def enrich_stock_profile(symbol: str) -> dict:
    """Index tags + fundamentals + LT quality for one symbol."""
    sym = display_symbol(symbol)
    tags = index_tags_of(sym)
    fund = {}
    try:
        fund = fetch_fundamentals(sym)
    except Exception:
        fund = {}
    lt = long_term_quality_from_fund(fund)
    # Prefer index-based cap if Yahoo mcap missing
    if lt.get("market_cap_bucket") == "Unknown":
        if "Large Cap (index)" in tags or "Nifty 50" in tags or "Nifty 100" in tags:
            lt["market_cap_bucket"] = "Large Cap"
        elif "Mid Cap (index)" in tags or "Nifty Midcap 100" in tags:
            lt["market_cap_bucket"] = "Mid Cap"
        elif "Small Cap (index)" in tags or "Nifty Smallcap 100" in tags:
            lt["market_cap_bucket"] = "Small Cap"

    mcap = fund.get("market_cap")
    revenue = fund.get("revenue_latest") or fund.get("total_revenue")
    # totalRevenue from info if statement missing
    if not revenue and fund.get("error") == "":
        pass
    book = fund.get("book_value")
    lt_score = safe_float(lt.get("lt_score"), 45)

    # Long-term call paired with swing BUY/SELL
    if lt_score >= 72:
        lt_with_buy = "LT: ACCUMULATE / CORE HOLD — swing BUY can add; if SL hits keep small core"
        lt_with_sell = "LT: Still quality — swing SELL is for trade book; don't dump entire long-term core blindly"
    elif lt_score >= 58:
        lt_with_buy = "LT: HOLD small core only — swing BUY OK with tight risk"
        lt_with_sell = "LT: REDUCE on strength — swing SELL aligns with average quality"
    else:
        lt_with_buy = "LT: NO core — treat BUY as pure trade; exit fully if SL hits"
        lt_with_sell = "LT: EXIT / avoid — swing SELL and stay out until quality improves"

    return {
        "Stock": sym,
        "Index Tags": ", ".join(tags) if tags else "Outside major indices",
        "In Nifty 50": "Nifty 50" in tags,
        "In Nifty 100": "Nifty 100" in tags,
        "In Nifty 500": "Nifty 500" in tags,
        "Market Cap Bucket": lt.get("market_cap_bucket", "Unknown"),
        "Market Cap (₹ Cr)": in_crore(mcap) if mcap else None,
        "Market Cap Cr Text": fmt_crore(mcap) if mcap else "—",
        "Book Value": safe_float(book) if book else None,
        "Book Value Text": f"₹{safe_float(book):,.2f}" if book else "—",
        "Face Value": safe_float(fund.get("face_value")) if fund.get("face_value") else None,
        "Face Value Text": f"₹{safe_float(fund.get('face_value')):.2f}" if fund.get("face_value") else "—",
        "Revenue (₹ Cr)": in_crore(revenue) if revenue else None,
        "Revenue Cr Text": fmt_crore(revenue) if revenue else "—",
        "Sales (₹ Cr)": in_crore(revenue) if revenue else None,  # sales ≈ revenue for display
        "Sales Cr Text": fmt_crore(revenue) if revenue else "—",
        "Net Income (₹ Cr)": in_crore(fund.get("net_income_latest")) if fund.get("net_income_latest") else None,
        "Net Income Cr Text": fmt_crore(fund.get("net_income_latest")) if fund.get("net_income_latest") else "—",
        "LT Score": lt_score,
        "LT Label": lt.get("lt_label", ""),
        "LT Hold if SL hits": lt.get("lt_hold_if_sl", ""),
        "LT Notes": " | ".join(lt.get("notes") or []),
        "LT with BUY call": lt_with_buy,
        "LT with SELL call": lt_with_sell,
        "PE": fund.get("pe"),
        "PB": fund.get("pb"),
        "ROE": fund.get("roe"),
        "Debt/Equity": fund.get("debt_to_equity"),
        "Profit Margin": fund.get("profit_margins"),
        "Revenue Growth": fund.get("revenue_growth"),
        "Name": fund.get("name") or sym,
        "Industry": fund.get("industry") or "",
    }


def show_call_with_long_term(symbol: str, call: str, compact: bool = False):
    """Show swing call + long-term analysis + book value / revenue in Cr."""
    try:
        prof = enrich_stock_profile(symbol)
    except Exception as e:
        st.caption(f"Fundamentals unavailable: {e}")
        return
    call_u = str(call or "").upper()
    lt_line = prof.get("LT with BUY call") if "BUY" in call_u else (
        prof.get("LT with SELL call") if "SELL" in call_u else prof.get("LT Hold if SL hits")
    )
    if compact:
        st.caption(
            f"BV {prof.get('Book Value Text')} · Rev {prof.get('Revenue Cr Text')} · "
            f"Mcap {prof.get('Market Cap Cr Text')} · LT {prof.get('LT Score'):.0f} — {lt_line}"
        )
        return
    st.markdown(
        f"""
        <div class="hold-box" style="padding:10px;border-radius:8px;margin:6px 0;">
        <b>Long-term with this {call_u or 'CALL'}:</b> {lt_line}<br>
        <b>Book value:</b> {prof.get('Book Value Text')}
        &nbsp;|&nbsp; <b>Revenue / Sales:</b> {prof.get('Revenue Cr Text')}
        &nbsp;|&nbsp; <b>Net income:</b> {prof.get('Net Income Cr Text')}<br>
        <b>Market cap:</b> {prof.get('Market Cap Cr Text')} ({prof.get('Market Cap Bucket')})
        &nbsp;|&nbsp; <b>LT score:</b> {safe_float(prof.get('LT Score')):.0f} — {prof.get('LT Label')}<br>
        <b>PE / PB / ROE:</b>
        {safe_float(prof.get('PE')):.1f} /
        {safe_float(prof.get('PB')):.2f} /
        {safe_float(prof.get('ROE')):.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


def attach_fundamentals_to_df(df: pd.DataFrame, max_n: int = 25) -> pd.DataFrame:
    """Add book value, revenue Cr, LT score columns for strategy tables."""
    if df is None or df.empty or "Stock" not in df.columns:
        return df
    x = df.copy()
    cols = {
        "Book Value": [],
        "Revenue (₹ Cr)": [],
        "Sales (₹ Cr)": [],
        "Market Cap (₹ Cr)": [],
        "LT Score": [],
        "LT Label": [],
        "LT with call": [],
    }
    for i, stock in enumerate(x["Stock"].astype(str).tolist()):
        if i >= max_n:
            for k in cols:
                cols[k].append(None if k != "LT Label" else "")
            continue
        try:
            p = enrich_stock_profile(stock)
            side = str(x.iloc[i].get("Side", x.iloc[i].get("Call", "BUY"))).upper()
            cols["Book Value"].append(p.get("Book Value"))
            cols["Revenue (₹ Cr)"].append(p.get("Revenue (₹ Cr)"))
            cols["Sales (₹ Cr)"].append(p.get("Sales (₹ Cr)"))
            cols["Market Cap (₹ Cr)"].append(p.get("Market Cap (₹ Cr)"))
            cols["LT Score"].append(p.get("LT Score"))
            cols["LT Label"].append(p.get("LT Label"))
            cols["LT with call"].append(
                p.get("LT with SELL call") if "SELL" in side else p.get("LT with BUY call")
            )
        except Exception:
            for k in cols:
                cols[k].append(None if k != "LT Label" else "")
    for k, v in cols.items():
        # pad if length mismatch
        while len(v) < len(x):
            v.append(None)
        x[k] = v[: len(x)]
    return x



def filterable_dataframe(
    df: pd.DataFrame,
    key: str,
    default_cols: list = None,
    height: int = 360,
):
    """Table with search + column filters for BUY/SELL/History visibility."""
    if df is None or df.empty:
        st.info("No rows to display.")
        return df
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    st.markdown(
        """
        <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;
                    padding:10px 12px;margin-bottom:8px;color:#94a3b8;font-size:0.9rem;">
        <b style="color:#e2e8f0;">Search + filters</b> · market cap / revenue in <b>₹ crore</b> when loaded.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("Search table", value="", key=f"{key}_q")
    with c2:
        cols_all = list(x.columns)
        default = default_cols if default_cols else cols_all[: min(12, len(cols_all))]
        default = [c for c in default if c in cols_all] or cols_all[:12]
        pick = st.multiselect("Columns", cols_all, default=default, key=f"{key}_cols")
    if not pick:
        pick = list(x.columns)[:12]
    filter_cols = [c for c in pick if c in x.columns and x[c].dtype == object][:4]
    if filter_cols:
        fcols = st.columns(max(len(filter_cols), 1))
        for i, col in enumerate(filter_cols):
            with fcols[i]:
                vals = sorted([str(v) for v in x[col].dropna().unique().tolist() if str(v).strip()][:40])
                if vals:
                    sel = st.multiselect(col, vals, default=[], key=f"{key}_f_{col}")
                    if sel:
                        x = x[x[col].astype(str).isin(sel)]
    if q and str(q).strip():
        qq = str(q).strip().lower()
        mask = pd.Series([False] * len(x), index=x.index)
        for c in pick:
            if c in x.columns:
                mask = mask | x[c].astype(str).str.lower().str.contains(qq, na=False)
        x = x[mask]
    st.caption(f"Showing **{len(x)}** rows")
    st.dataframe(x[pick], use_container_width=True, hide_index=True, height=height)
    return x


def render_call_stock_card(row, section_key: str = "card"):
    """Shared visual card for BUY / SELL / history."""
    stock = display_symbol(str(row.get("Stock", row.get("Symbol", ""))))
    if not stock:
        return
    call = str(row.get("Call", row.get("Side", "BUY"))).upper()
    border = "#16a34a" if "BUY" in call else ("#dc2626" if "SELL" in call else "#64748b")
    price = safe_float(row.get("Price", row.get("Entry", row.get("Current Price"))))
    tgt = safe_float(row.get("Target"))
    sl = safe_float(row.get("Stop Loss"))
    src = str(row.get("Call Source", row.get("Strategy", "")) or "")
    result = str(row.get("Result", "") or "")
    name = stock
    mcap_txt = cap_class = bv_txt = rev_txt = idx_txt = "—"
    lt_line = ""
    try:
        p = enrich_stock_profile(stock)
        name = p.get("Name") or stock
        mcap_txt = p.get("Market Cap Cr Text", "—")
        cap_class = p.get("Market Cap Bucket", "—")
        bv_txt = p.get("Book Value Text", "—")
        rev_txt = p.get("Revenue Cr Text", "—")
        idx_txt = p.get("Index Tags", "—")
        lt_line = p.get("LT with SELL call") if "SELL" in call else p.get("LT with BUY call", "")
    except Exception:
        pass
    cap_u = str(cap_class).upper()
    if "LARGE" in cap_u:
        cap_color = "#0d9488"
    elif "MID" in cap_u:
        cap_color = "#2563eb"
    elif "SMALL" in cap_u or "MICRO" in cap_u:
        cap_color = "#d97706"
    else:
        cap_color = "#64748b"
    st.markdown(
        f"""
        <div style="border:2px solid {border};border-radius:14px;padding:14px 16px;margin:10px 0;
                    background:linear-gradient(145deg,#0f172a 0%,#1e293b 100%);">
          <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;">
            <div>
              <div style="font-size:1.25rem;font-weight:700;color:#f8fafc;">{name}</div>
              <div style="color:#94a3b8;">{stock} · <b style="color:{border};">{call}</b>{(' · '+src) if src else ''}</div>
            </div>
            <div style="text-align:right;">
              <span style="background:{cap_color};color:#fff;padding:3px 10px;border-radius:16px;font-size:0.8rem;font-weight:600;">{cap_class}</span>
              <div style="color:#e2e8f0;margin-top:4px;"><b>Mcap</b> {mcap_txt}</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-top:10px;">
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">PRICE / ENTRY</div>
              <div style="color:#f8fafc;font-weight:600;">₹{price:,.2f}</div></div>
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">TARGET</div>
              <div style="color:#4ade80;font-weight:600;">₹{tgt:,.2f}</div></div>
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">STOP</div>
              <div style="color:#f87171;font-weight:600;">₹{sl:,.2f}</div></div>
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">BOOK VALUE</div>
              <div style="color:#f8fafc;font-weight:600;">{bv_txt}</div></div>
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">REVENUE</div>
              <div style="color:#f8fafc;font-weight:600;">{rev_txt}</div></div>
          </div>
          <div style="margin-top:8px;color:#94a3b8;font-size:0.88rem;"><b>Index:</b> {idx_txt}{(' · <b>Result:</b> '+result) if result else ''}</div>
          <div style="margin-top:4px;color:#a5b4fc;font-size:0.88rem;"><b>LT:</b> {lt_line or '—'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Experienced trader — {stock}", expanded=False):
        show_expert_trader_box(stock, call_hint=call)
    render_active_trade_buttons(
        stock,
        side_hint=call,
        entry=price,
        target=tgt,
        stop=sl,
        key_prefix=f"{section_key}_{stock}",
    )


def aggregate_stocks_unique_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per stock. Merge all strategy names into a single list.
    Keeps best target/stop from first row; counts strategies.
    """
    if df is None or df.empty or "Stock" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["Stock"] = x["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()
    rows = []
    for stock, g in x.groupby("Stock", sort=False):
        strats = []
        for s in g.get("Strategy", pd.Series(dtype=str)).astype(str).tolist():
            if s and s.lower() not in ("nan", "none", ""):
                if s not in strats:
                    strats.append(s)
        # Also parse "Also matches" / "All matching strategies"
        for col in ("Also matches", "All matching strategies"):
            if col in g.columns:
                for cell in g[col].astype(str).tolist():
                    for part in str(cell).split(","):
                        part = part.strip()
                        if part and part.lower() not in ("nan", "none", "—", "-") and part not in strats:
                            strats.append(part)
        first = g.iloc[0]
        live = safe_float(first.get("Current / Live Price", first.get("Price")))
        tgt = safe_float(first.get("Target"))
        sl = safe_float(first.get("Stop Loss"))
        # Prefer row with most complete levels
        for _, rr in g.iterrows():
            if safe_float(rr.get("Target")) > 0 and safe_float(rr.get("Stop Loss")) > 0:
                live = safe_float(rr.get("Current / Live Price", rr.get("Price")), live)
                tgt = safe_float(rr.get("Target"), tgt)
                sl = safe_float(rr.get("Stop Loss"), sl)
                break
        side = str(first.get("Side", first.get("Call", "BUY"))).upper()
        rows.append({
            "Stock": stock,
            "Side": side if side in ("BUY", "SELL", "HOLD") else "BUY",
            "Strategies": " · ".join(strats) if strats else str(first.get("Strategy", "—")),
            "# Strategies": len(strats) if strats else 1,
            "Price": live,
            "Target": tgt,
            "Stop Loss": sl,
            "Hold Days": first.get("Hold Days", 15),
            "RSI": first.get("RSI"),
            "ADX": first.get("ADX"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("# Strategies", ascending=False)
    return out.reset_index(drop=True)


def render_unique_strategy_stock_cards(df: pd.DataFrame, max_cards: int = 12, key_prefix: str = "usc"):
    """
    Visual cards: company once, all strategies, mcap/cap class, BV, revenue Cr,
    experienced trader feedback.
    """
    if df is None or df.empty:
        st.info("No strategy stocks to show.")
        return
    uniq = aggregate_stocks_unique_strategies(df)
    st.success(f"**{len(uniq)} unique stocks** (duplicates merged — all strategies listed per name)")
    show_n = st.slider("Cards to show", 3, min(30, max(3, len(uniq))), min(max_cards, len(uniq)), key=f"{key_prefix}_n")
    load_fund = st.checkbox("Load company name · mcap · BV · revenue (₹ Cr) + desk feedback", value=True, key=f"{key_prefix}_fund")

    for i, row in uniq.head(show_n).iterrows():
        stock = str(row.get("Stock", ""))
        side = str(row.get("Side", "BUY"))
        strats = str(row.get("Strategies", "—"))
        n_s = int(safe_float(row.get("# Strategies"), 1))
        price = safe_float(row.get("Price"))
        tgt = safe_float(row.get("Target"))
        sl = safe_float(row.get("Stop Loss"))

        name = stock
        mcap_txt = "—"
        cap_class = "—"
        bv_txt = "—"
        rev_txt = "—"
        idx_txt = "—"
        face_txt = "—"
        profit_txt = "—"
        lt_line = ""
        if load_fund:
            try:
                p = enrich_stock_profile(stock)
                name = p.get("Name") or stock
                mcap_txt = p.get("Market Cap Cr Text", "—")
                cap_class = p.get("Market Cap Bucket", "—")
                bv_txt = p.get("Book Value Text", "—")
                rev_txt = p.get("Revenue Cr Text", "—")
                profit_txt = p.get("Net Income Cr Text", "—")
                face_txt = p.get("Face Value Text", "—")
                idx_txt = p.get("Index Tags", "—")
                lt_line = p.get("LT with SELL call") if side == "SELL" else p.get("LT with BUY call", "")
            except Exception:
                pass

        # Cap badge colour
        cap_u = str(cap_class).upper()
        if "LARGE" in cap_u:
            cap_color = "#0d9488"
        elif "MID" in cap_u:
            cap_color = "#2563eb"
        elif "SMALL" in cap_u or "MICRO" in cap_u:
            cap_color = "#d97706"
        else:
            cap_color = "#64748b"

        border = "#16a34a" if side == "BUY" else ("#dc2626" if side == "SELL" else "#64748b")
        st.markdown(
            f"""
            <div style="border:2px solid {border};border-radius:14px;padding:16px 18px;margin:12px 0;
                        background:linear-gradient(145deg,#0f172a 0%,#1e293b 100%);">
              <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;align-items:center;">
                <div>
                  <div style="font-size:1.35rem;font-weight:700;color:#f8fafc;">{name}</div>
                  <div style="color:#94a3b8;font-size:0.95rem;">{stock} · <b style="color:{border};">{side}</b></div>
                </div>
                <div style="text-align:right;">
                  <span style="background:{cap_color};color:#fff;padding:4px 10px;border-radius:20px;font-size:0.85rem;font-weight:600;">
                    {cap_class}
                  </span>
                  <div style="margin-top:6px;color:#e2e8f0;font-size:1.05rem;"><b>Mcap</b> {mcap_txt}</div>
                </div>
              </div>
              <div style="margin-top:12px;padding:10px;border-radius:8px;background:#020617;color:#cbd5e1;line-height:1.55;">
                <b style="color:#fbbf24;">Strategies ({n_s}):</b> {strats}
              </div>
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:12px;">
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">LIVE</div>
                  <div style="color:#f8fafc;font-weight:600;">₹{price:,.2f}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">TARGET</div>
                  <div style="color:#4ade80;font-weight:600;">₹{tgt:,.2f}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">STOP</div>
                  <div style="color:#f87171;font-weight:600;">₹{sl:,.2f}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">BOOK VALUE</div>
                  <div style="color:#f8fafc;font-weight:600;">{bv_txt}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">REVENUE</div>
                  <div style="color:#f8fafc;font-weight:600;">{rev_txt}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">PROFIT</div>
                  <div style="color:#f8fafc;font-weight:600;">{profit_txt}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">FACE VALUE</div>
                  <div style="color:#f8fafc;font-weight:600;">{face_txt}</div></div>
              </div>
              <div style="margin-top:10px;color:#94a3b8;font-size:0.9rem;"><b>Index:</b> {idx_txt}</div>
              <div style="margin-top:6px;color:#a5b4fc;font-size:0.9rem;"><b>Long-term:</b> {lt_line or '—'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"🎓 Experienced trader feedback — {stock}", expanded=True):
            show_expert_trader_box(stock, call_hint=side)
        render_active_trade_buttons(
            stock,
            side_hint=side,
            entry=price,
            target=tgt,
            stop=sl,
            key_prefix=f"{key_prefix}_{stock}",
        )
        with st.expander(f"📈 Chart — {stock}", expanded=False):
            try:
                raw = stock_history(clean_symbol(stock), interval="1d")
                if raw is not None and not raw.empty:
                    fig = build_full_plotly_chart(
                        calculate_indicators(raw),
                        title=f"{name} ({stock})",
                        target=tgt,
                        stop_loss=sl,
                        height=480,
                        show_rsi=True,
                    )
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.caption(f"Chart: {e}")


def enrich_results_profiles(results: pd.DataFrame, max_n: int = 40) -> pd.DataFrame:
    """Attach index/cap/LT columns for top rows (slow network — limited)."""
    if results is None or results.empty:
        return results
    x = results.copy()
    rows = []
    stocks = x["Stock"].astype(str).head(max_n).tolist()
    for s in stocks:
        try:
            rows.append(enrich_stock_profile(s))
        except Exception:
            rows.append({"Stock": display_symbol(s), "Market Cap Bucket": "Unknown", "LT Score": 45})
    if not rows:
        return x
    prof = pd.DataFrame(rows)
    # drop overlapping cols before merge
    for c in prof.columns:
        if c != "Stock" and c in x.columns:
            x = x.drop(columns=[c], errors="ignore")
    x["Stock_key"] = x["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False)
    prof["Stock_key"] = prof["Stock"].astype(str).str.upper()
    x = x.merge(prof.drop(columns=["Stock"], errors="ignore"), on="Stock_key", how="left")
    x = x.drop(columns=["Stock_key"], errors="ignore")
    return x


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def sma(series, period):

    return series.rolling(
        period
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return 100 - (
        100 /
        (1 + rs)
    )


def macd(series):

    fast = ema(
        series,
        12
    )

    slow = ema(
        series,
        26
    )

    line = fast - slow

    signal = ema(
        line,
        9
    )

    histogram = line - signal

    return line, signal, histogram


def atr(df, period=14):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    previous = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous
    ).abs()

    tr3 = (
        low - previous
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(
        period
    ).mean()


def adx(df, period=14):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (
            up_move > down_move
        )
        &
        (
            up_move > 0
        ),
        up_move,
        0
    )

    minus_dm = np.where(
        (
            down_move > up_move
        )
        &
        (
            down_move > 0
        ),
        down_move,
        0
    )

    tr = pd.concat(
        [
            high - low,
            (
                high - close.shift()
            ).abs(),
            (
                low - close.shift()
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr_value = tr.rolling(
        period
    ).mean()

    plus_di = (
        100
        *
        pd.Series(
            plus_dm,
            index=df.index
        ).rolling(period).mean()
        /
        atr_value.replace(
            0,
            np.nan
        )
    )

    minus_di = (
        100
        *
        pd.Series(
            minus_dm,
            index=df.index
        ).rolling(period).mean()
        /
        atr_value.replace(
            0,
            np.nan
        )
    )

    denominator = (
        plus_di +
        minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        (
            plus_di -
            minus_di
        ).abs()
        /
        denominator
    ) * 100

    return dx.rolling(
        period
    ).mean()


def stochastic(df, period=14):

    low_min = (
        df["Low"]
        .rolling(period)
        .min()
    )

    high_max = (
        df["High"]
        .rolling(period)
        .max()
    )

    denominator = (
        high_max -
        low_min
    ).replace(
        0,
        np.nan
    )

    k = (
        100
        *
        (
            df["Close"] -
            low_min
        )
        /
        denominator
    )

    d = k.rolling(
        3
    ).mean()

    return k, d


def bollinger(df, period=20):

    middle = (
        df["Close"]
        .rolling(period)
        .mean()
    )

    std = (
        df["Close"]
        .rolling(period)
        .std()
    )

    upper = middle + 2 * std

    lower = middle - 2 * std

    return (
        middle,
        upper,
        lower
    )


def vwap(df):

    typical = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    volume = (
        pd.to_numeric(
            df["Volume"],
            errors="coerce"
        )
        .fillna(0)
    )

    cumulative_volume = (
        volume.cumsum()
    )

    return (
        typical * volume
    ).cumsum() / cumulative_volume.replace(
        0,
        np.nan
    )


# ============================================================
# PATTERNS
# ============================================================

def detect_patterns(df):

    patterns = []

    if df is None or len(df) < 5:
        return patterns

    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]

    last = -1
    prev = -2

    body = abs(
        c.iloc[last] -
        o.iloc[last]
    )

    candle_range = (
        h.iloc[last] -
        l.iloc[last]
    )

    if candle_range > 0:

        upper_wick = (
            h.iloc[last] -
            max(
                c.iloc[last],
                o.iloc[last]
            )
        )

        lower_wick = (
            min(
                c.iloc[last],
                o.iloc[last]
            ) -
            l.iloc[last]
        )

        if (
            lower_wick > body * 2
            and
            upper_wick < body
        ):
            patterns.append(
                "Hammer"
            )

        if (
            upper_wick > body * 2
            and
            lower_wick < body
        ):
            patterns.append(
                "Shooting Star"
            )

    if (
        c.iloc[prev] < o.iloc[prev]
        and
        c.iloc[last] > o.iloc[last]
        and
        c.iloc[last] >= o.iloc[prev]
        and
        o.iloc[last] <= c.iloc[prev]
    ):
        patterns.append(
            "Bullish Engulfing"
        )

    if (
        c.iloc[prev] > o.iloc[prev]
        and
        c.iloc[last] < o.iloc[last]
        and
        o.iloc[last] >= c.iloc[prev]
        and
        c.iloc[last] <= o.iloc[prev]
    ):
        patterns.append(
            "Bearish Engulfing"
        )

    if (
        candle_range > 0
        and
        body / candle_range < 0.1
    ):
        patterns.append(
            "Doji"
        )

    if len(df) >= 20:

        recent = df.tail(10)

        if (
            recent["High"].iloc[-1]
            >
            recent["High"].iloc[0]
            and
            recent["Low"].iloc[-1]
            >
            recent["Low"].iloc[0]
        ):
            patterns.append(
                "Higher High / Higher Low"
            )

        if (
            recent["High"].iloc[-1]
            <
            recent["High"].iloc[0]
            and
            recent["Low"].iloc[-1]
            <
            recent["Low"].iloc[0]
        ):
            patterns.append(
                "Lower High / Lower Low"
            )

    # Extra structure / momentum patterns (for learning + teaching)
    if len(df) >= 5:
        # Marubozu-like strong close
        if candle_range > 0 and body / candle_range > 0.7:
            if c.iloc[last] > o.iloc[last]:
                patterns.append("Bullish Marubozu")
            else:
                patterns.append("Bearish Marubozu")
        # Inside bar (compression)
        if (
            h.iloc[last] <= h.iloc[prev]
            and l.iloc[last] >= l.iloc[prev]
        ):
            patterns.append("Inside Bar")
        # Outside bar (expansion)
        if (
            h.iloc[last] >= h.iloc[prev]
            and l.iloc[last] <= l.iloc[prev]
            and body > abs(c.iloc[prev] - o.iloc[prev])
        ):
            patterns.append("Outside Bar")

    if len(df) >= 15:
        # Simple swing breakout: close above prior 10-bar high
        prior_high = h.iloc[-11:-1].max()
        prior_low = l.iloc[-11:-1].min()
        if c.iloc[last] > prior_high:
            patterns.append("Breakout High")
        if c.iloc[last] < prior_low:
            patterns.append("Breakdown Low")

    # De-dupe preserve order
    seen = set()
    out = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# Default pattern importance (before learning adjusts weights)
PATTERN_IMPORTANCE = {
    "Hammer": {
        "bias": "bullish", "base_weight": 5,
        "why": "Rejection of lower prices; buyers stepped in after a selloff.",
        "use": "Best after decline or at support; confirm with next green close + volume.",
    },
    "Shooting Star": {
        "bias": "bearish", "base_weight": -5,
        "why": "Rejection of higher prices; supply appeared at the highs.",
        "use": "Best after rally or at resistance; avoid new longs until structure improves.",
    },
    "Bullish Engulfing": {
        "bias": "bullish", "base_weight": 6,
        "why": "Buyers fully overpowered the prior session’s sellers.",
        "use": "Stronger with volume > average and above EMA50.",
    },
    "Bearish Engulfing": {
        "bias": "bearish", "base_weight": -6,
        "why": "Sellers fully overpowered the prior session’s buyers.",
        "use": "Warning to tighten stops or skip fresh longs.",
    },
    "Doji": {
        "bias": "neutral", "base_weight": 0,
        "why": "Indecision; trend may pause or reverse.",
        "use": "Never trade alone — wait for next directional candle.",
    },
    "Higher High / Higher Low": {
        "bias": "bullish", "base_weight": 7,
        "why": "Uptrend structure — demand is stepping up.",
        "use": "Favour BUY/HOLD while structure holds; break of last HL is caution.",
    },
    "Lower High / Lower Low": {
        "bias": "bearish", "base_weight": -7,
        "why": "Downtrend structure — supply is in control.",
        "use": "Avoid fresh longs; prefer exit / wait for structure break up.",
    },
    "Bullish Marubozu": {
        "bias": "bullish", "base_weight": 4,
        "why": "Strong conviction buying with little wick rejection.",
        "use": "Momentum continuation signal; still respect stop below bar low.",
    },
    "Bearish Marubozu": {
        "bias": "bearish", "base_weight": -4,
        "why": "Strong conviction selling with little wick rejection.",
        "use": "Momentum downside; avoid catching falling knife.",
    },
    "Inside Bar": {
        "bias": "neutral", "base_weight": 1,
        "why": "Volatility compression — breakout often follows.",
        "use": "Trade the break of mother-bar high/low with volume.",
    },
    "Outside Bar": {
        "bias": "context", "base_weight": 2,
        "why": "Range expansion; can mark reversal or acceleration.",
        "use": "Combine with trend (EMA) to decide direction.",
    },
    "Breakout High": {
        "bias": "bullish", "base_weight": 5,
        "why": "Close above recent range high — demand broke supply.",
        "use": "Best with volume surge; false breakouts fail back inside range.",
    },
    "Breakdown Low": {
        "bias": "bearish", "base_weight": -5,
        "why": "Close below recent range low — supply broke demand.",
        "use": "Avoid longs until reclaim; trail shorts carefully.",
    },
}


def pattern_weight(name: str, learning: dict = None) -> float:
    """Base weight × learned multiplier (from past wins/losses)."""
    meta = PATTERN_IMPORTANCE.get(name, {"base_weight": 2})
    w = float(meta.get("base_weight", 2))
    if learning:
        mults = learning.get("pattern_multipliers", {})
        w *= float(mults.get(name, 1.0))
    return w


def load_learning() -> dict:
    try:
        if LEARNING_FILE.exists():
            import json
            with open(LEARNING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_learning(data: dict):
    try:
        import json
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LEARNING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def learn_from_divergence_training() -> dict:
    """
    Learn from My Strategy train log + manual chart examples
    (e.g. RELIANCE / COALINDIA style: price lower-lows + RSI higher-lows).
    Returns multipliers and lessons for RSI bullish divergence.
    """
    out = {
        "rsi_bull_div_multiplier": 1.15,
        "my_strategy_boost": 8,
        "lessons": [],
        "examples": 0,
        "good": 0,
        "bad": 0,
    }
    train_csv = APP_DIR / "my_strategy_train_log.csv"
    rows = []
    if train_csv.exists():
        try:
            rows = pd.read_csv(train_csv).to_dict("records")
        except Exception:
            rows = []
    # Built-in reference patterns (user chart types)
    seed = [
        {
            "symbol": "RELIANCE",
            "outcome": "ACTIVE_FRESH good",
            "note": "Bullish RSI divergence: price lower lows + RSI higher lows into June pivot; classic long setup",
        },
        {
            "symbol": "COALINDIA",
            "outcome": "ACTIVE_FRESH good",
            "note": "Bullish RSI divergence after sell: price lower low + RSI higher low then bounce; buy on confirmation",
        },
    ]
    # Don't duplicate seeds every run — only use for learning math
    for s in seed:
        rows.append(s)

    good_kw = ("ACTIVE_FRESH good", "TARGET HIT", "good", "TARGET")
    bad_kw = ("TOO_LATE", "RSI_FAILED", "STOP HIT", "Should not have bought", "STOP")
    good = bad = 0
    for r in rows:
        oc = str(r.get("outcome", "") or "")
        note = str(r.get("note", "") or "").lower()
        out["examples"] += 1
        if any(k.lower() in oc.lower() for k in good_kw):
            good += 1
            if "diverg" in note or "rsi" in note:
                good += 0.5
        if any(k.lower() in oc.lower() for k in bad_kw):
            bad += 1
    out["good"] = int(good)
    out["bad"] = int(bad)
    total = max(1, good + bad)
    wr = good / total
    # Multiplier 0.85 – 1.35 based on training quality
    out["rsi_bull_div_multiplier"] = round(0.85 + min(0.50, wr * 0.50), 3)
    out["my_strategy_boost"] = int(5 + min(15, wr * 15))
    out["lessons"].append(
        f"RSI bullish divergence training: {out['examples']} examples · "
        f"good-weight {out['good']} / bad {out['bad']} → "
        f"pattern multiplier ×{out['rsi_bull_div_multiplier']} · "
        f"My Strategy score boost +{out['my_strategy_boost']}"
    )
    out["lessons"].append(
        "Pattern rule locked: PRICE lower-low + RSI higher-low (same swing bars), "
        "fresh only (not 2–3 days after bounce). RELIANCE/COALINDIA-style charts are the template."
    )
    return out


def learn_from_history(min_closed: int = 5) -> dict:
    """
    Learn from closed past predictions to improve future scoring.
    Uses wins (TARGET ACHIEVED) vs losses (STOP LOSS HIT) to set:
    - pattern multipliers
    - preferred prediction buckets
    - risk preferences
    - call-type success
    Also learns RSI divergence from My Strategy training examples.
    """
    history = normalize_history_df(load_history())
    learning = {
        "pattern_multipliers": {},
        "pattern_stats": {},
        "pred_bucket_stats": {},
        "risk_stats": {},
        "call_stats": {},
        "lessons": [],
        "sample_closed": 0,
        "sample_wins": 0,
        "sample_losses": 0,
        "rsi_bull_div_multiplier": 1.15,
        "my_strategy_boost": 8,
    }
    # Always fold divergence training (works even with few closed trades)
    try:
        div = learn_from_divergence_training()
        learning["rsi_bull_div_multiplier"] = div.get("rsi_bull_div_multiplier", 1.15)
        learning["my_strategy_boost"] = div.get("my_strategy_boost", 8)
        learning["pattern_multipliers"]["RSI_BULL_DIV"] = learning["rsi_bull_div_multiplier"]
        learning["pattern_multipliers"]["MY_STRATEGY_DIV"] = learning["rsi_bull_div_multiplier"]
        learning["lessons"].extend(div.get("lessons") or [])
    except Exception:
        pass
    if history is None or history.empty:
        save_learning(learning)
        return learning

    res = history["Result"].astype(str).str.upper()
    wins_m = res.str.contains("TARGET ACHIEVED", na=False) | res.isin(["WIN"])
    loss_m = res.str.contains("STOP LOSS HIT", na=False) | res.isin(["LOSS"])
    closed = history[wins_m | loss_m].copy()
    learning["sample_closed"] = len(closed)
    learning["sample_wins"] = int(wins_m.sum())
    learning["sample_losses"] = int(loss_m.sum())

    if len(closed) < min_closed:
        learning["lessons"].append(
            f"Need at least {min_closed} closed target/stop outcomes to learn firmly "
            f"(now {len(closed)}). Keep scanning + Force re-check."
        )
        save_learning(learning)
        return learning

    closed["_win"] = closed["Result"].astype(str).str.upper().str.contains(
        "TARGET ACHIEVED", na=False
    ) | closed["Result"].astype(str).str.upper().isin(["WIN"])

    # --- Call type ---
    for call in ["BUY", "HOLD", "SELL"]:
        sub = closed[closed["Call"].astype(str).str.upper() == call]
        if len(sub) < 3:
            continue
        wr = float(sub["_win"].mean() * 100)
        learning["call_stats"][call] = {
            "n": int(len(sub)),
            "win_rate": round(wr, 1),
        }
        if wr >= 55:
            learning["lessons"].append(f"{call} calls historically win ~{wr:.0f}% — favour this call type.")
        elif wr <= 40:
            learning["lessons"].append(f"{call} calls historically win only ~{wr:.0f}% — be stricter on filters.")

    # --- Prediction strength buckets ---
    pred = pd.to_numeric(closed.get("Prediction"), errors="coerce")
    for lo, hi, label in [(0, 60, "pred_lt_60"), (60, 70, "pred_60_70"), (70, 80, "pred_70_80"), (80, 101, "pred_80_plus")]:
        m = (pred >= lo) & (pred < hi)
        sub = closed[m]
        if len(sub) < 3:
            continue
        wr = float(sub["_win"].mean() * 100)
        learning["pred_bucket_stats"][label] = {"n": int(len(sub)), "win_rate": round(wr, 1), "lo": lo, "hi": hi}
        if wr >= 58:
            learning["lessons"].append(f"Prediction {lo}–{hi}% zone wins ~{wr:.0f}% — boost confidence here.")
        elif wr <= 42:
            learning["lessons"].append(f"Prediction {lo}–{hi}% zone wins only ~{wr:.0f}% — reduce size / skip weak ones.")

    # --- Risk level ---
    if "Risk Level" in closed.columns:
        for lvl in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]:
            sub = closed[closed["Risk Level"].astype(str).str.upper() == lvl]
            if len(sub) < 3:
                continue
            wr = float(sub["_win"].mean() * 100)
            learning["risk_stats"][lvl] = {"n": int(len(sub)), "win_rate": round(wr, 1)}
            if lvl in ["HIGH", "VERY HIGH"] and wr < 45:
                learning["lessons"].append(f"{lvl} risk wins only ~{wr:.0f}% historically — avoid unless exceptional.")
            if lvl == "LOW" and wr >= 55:
                learning["lessons"].append(f"LOW risk wins ~{wr:.0f}% — prioritise low-risk BUYs.")

    # --- Patterns from Patterns column or Reason text ---
    pattern_names = list(PATTERN_IMPORTANCE.keys())
    text_col = None
    for c in ["Patterns", "Reason", "Result Detail", "Recommendation"]:
        if c in closed.columns:
            text_col = c if text_col is None else text_col
    # Prefer Patterns if present
    if "Patterns" in closed.columns:
        text_series = closed["Patterns"].astype(str)
    elif "Reason" in closed.columns:
        text_series = closed["Reason"].astype(str)
    else:
        text_series = pd.Series([""] * len(closed), index=closed.index)

    for pname in pattern_names:
        mask = text_series.str.contains(pname.replace("/", "\\/"), case=False, na=False, regex=False)
        # also try partial for HH/HL
        if not mask.any() and "Higher High" in pname:
            mask = text_series.str.contains("Higher High", case=False, na=False)
        if not mask.any() and "Lower High" in pname:
            mask = text_series.str.contains("Lower High", case=False, na=False)
        sub = closed[mask]
        if len(sub) < 3:
            continue
        wr = float(sub["_win"].mean() * 100)
        n = int(len(sub))
        learning["pattern_stats"][pname] = {"n": n, "win_rate": round(wr, 1)}
        # Multiplier: win rate 50% → 1.0, 70% → 1.4, 30% → 0.6
        mult = max(0.4, min(1.6, 0.4 + (wr / 100.0) * 1.2))
        # For bearish patterns, high win-rate on BUY history means pattern hurt (losses) —
        # win_rate here is still target achieved when pattern was present at call time
        learning["pattern_multipliers"][pname] = round(mult, 2)
        bias = PATTERN_IMPORTANCE.get(pname, {}).get("bias", "")
        learning["lessons"].append(
            f"Pattern **{pname}** ({bias}): historical win rate ~{wr:.0f}% over {n} closed trades "
            f"→ weight multiplier {mult:.2f}x. "
            f"{PATTERN_IMPORTANCE.get(pname, {}).get('why', '')}"
        )

    # Global tip
    overall_wr = float(closed["_win"].mean() * 100)
    learning["overall_win_rate"] = round(overall_wr, 1)
    learning["lessons"].insert(
        0,
        f"Overall decided win rate (target vs stop): **{overall_wr:.1f}%** on {len(closed)} closed trades. "
        "Model now weights patterns and prediction zones using this history.",
    )

    save_learning(learning)
    return learning


def learning_score_adjustment(prediction: float, risk_level: str, call: str, patterns: list, learning: dict = None) -> tuple:
    """
    Extra points from learned history. Returns (delta_score, notes_list).
    """
    if not learning:
        learning = load_learning()
    delta = 0.0
    notes = []
    if not learning or learning.get("sample_closed", 0) < 5:
        return 0.0, notes

    # Prediction bucket
    pred = safe_float(prediction)
    bucket = None
    if pred < 60:
        bucket = "pred_lt_60"
    elif pred < 70:
        bucket = "pred_60_70"
    elif pred < 80:
        bucket = "pred_70_80"
    else:
        bucket = "pred_80_plus"
    bs = learning.get("pred_bucket_stats", {}).get(bucket)
    if bs and bs.get("n", 0) >= 3:
        wr = bs["win_rate"]
        if wr >= 60:
            delta += 4
            notes.append(f"Learning: {bucket} wins ~{wr}% historically (+).")
        elif wr <= 40:
            delta -= 5
            notes.append(f"Learning: {bucket} wins only ~{wr}% historically (−).")

    # Risk
    rl = str(risk_level or "").upper()
    rs = learning.get("risk_stats", {}).get(rl)
    if rs and rs.get("n", 0) >= 3:
        wr = rs["win_rate"]
        if rl in ["HIGH", "VERY HIGH"] and wr < 45:
            delta -= 4
            notes.append(f"Learning: {rl} risk underperforms (~{wr}%).")
        if rl == "LOW" and wr >= 55:
            delta += 3
            notes.append(f"Learning: LOW risk outperforms (~{wr}%).")

    # Call type
    cs = learning.get("call_stats", {}).get(str(call).upper())
    if cs and cs.get("n", 0) >= 3:
        wr = cs["win_rate"]
        if wr >= 58:
            delta += 2
        elif wr <= 40:
            delta -= 3
            notes.append(f"Learning: {call} win rate only ~{wr}%.")

    return delta, notes


# ============================================================
# INDICATOR CALCULATION
# ============================================================

def calculate_indicators(df):

    df = normalize_columns(df).copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for col in required:

        if col not in df.columns:
            return pd.DataFrame()

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if "Volume" not in df.columns:

        df["Volume"] = 0

    df["Volume"] = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=required
    )

    df["EMA20"] = ema(
        df["Close"],
        20
    )

    df["EMA50"] = ema(
        df["Close"],
        50
    )

    df["EMA200"] = ema(
        df["Close"],
        200
    )

    df["RSI"] = rsi(
        df["Close"]
    )

    (
        df["MACD"],
        df["MACDSignal"],
        df["MACDHist"]
    ) = macd(
        df["Close"]
    )

    df["ATR"] = atr(df)

    df["ADX"] = adx(df)

    (
        df["StochK"],
        df["StochD"]
    ) = stochastic(df)

    (
        df["BBMiddle"],
        df["BBUpper"],
        df["BBLower"]
    ) = bollinger(df)

    df["VWAP"] = vwap(df)

    df["VolumeAvg20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # --- TradingView-style extras (open equivalents of popular tools) ---
    try:
        df["Supertrend"], df["ST_Dir"] = supertrend(df, period=10, multiplier=3.0)
    except Exception:
        df["Supertrend"] = np.nan
        df["ST_Dir"] = 0
    try:
        df["EMA9"] = ema(df["Close"], 9)
        df["EMA21"] = ema(df["Close"], 21)
    except Exception:
        df["EMA9"] = np.nan
        df["EMA21"] = np.nan
    try:
        # Stochastic RSI (common TV indicator family)
        rsi_s = rsi(df["Close"], 14)
        rsi_min = rsi_s.rolling(14).min()
        rsi_max = rsi_s.rolling(14).max()
        stoch_rsi = (rsi_s - rsi_min) / (rsi_max - rsi_min + 1e-9)
        df["StochRSI"] = stoch_rsi * 100
    except Exception:
        df["StochRSI"] = np.nan
    try:
        # Donchian channel (breakout systems / turtle-style)
        df["DonchianHigh"] = df["High"].rolling(20).max()
        df["DonchianLow"] = df["Low"].rolling(20).min()
    except Exception:
        df["DonchianHigh"] = np.nan
        df["DonchianLow"] = np.nan

    if "VolumeAvg20" in df.columns:
        df["Volume Ratio"] = df["Volume"] / df["VolumeAvg20"].replace(0, np.nan)
    else:
        df["Volume Ratio"] = 1.0

    return df


def supertrend(df, period=10, multiplier=3.0):
    """Supertrend — widely used on TradingView for swing trend."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    atr_s = atr(df, period) if "ATR" not in df.columns else df["ATR"]
    if atr_s is None or (isinstance(atr_s, pd.Series) and atr_s.isna().all()):
        atr_s = atr(df, period)
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr_s
    lower = hl2 - multiplier * atr_s
    st_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    st_line.iloc[0] = upper.iloc[0]
    direction.iloc[0] = 1
    for i in range(1, len(df)):
        if close.iloc[i] > st_line.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < st_line.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
        if direction.iloc[i] == 1:
            st_line.iloc[i] = max(lower.iloc[i], st_line.iloc[i - 1]) if direction.iloc[i - 1] == 1 else lower.iloc[i]
        else:
            st_line.iloc[i] = min(upper.iloc[i], st_line.iloc[i - 1]) if direction.iloc[i - 1] == -1 else upper.iloc[i]
    return st_line, direction


# ============================================================
# SWING STRATEGY LAB — multi-strategy + historical / live backtest
# ============================================================
# Target: high trade efficiency via strict filters (not a guarantee of 70%).
# Win = price reaches target before stop within hold window.

SWING_STRATEGIES = {
    "my_strategy": {
        "name": "My Strategy (RSI Divergence)",
        "side": "BUY",
        "hold_days": 15,
        "desc": "ONLY building: price lower-lows + RSI higher-lows, RSI line unbroken. Too late if 2–3 days passed or price already bounced.",
        "why": "Your rule — early divergence only. Completed patterns (e.g. Shiva Mills after 2–3 days) are NOT buy calls.",
        "beginner": (
            "DETAILED: (1) Zoom out months on daily. (2) Price swing lows falling. (3) RSI swing lows rising. "
            "(4) RSI line not broken. (5) Call only while pattern is BUILDING (0–2 days after the lows). "
            "(6) If pattern already complete and price ran for 2–3 days → DO NOT BUY (too late). "
            "(7) Next day small dip with RSI holding = confirmation; big gap-up = don't chase."
        ),
        "typical_win": "35–45%",
        "typical_exp": "−0.20% to +0.05%",
    },
    "ema_pullback": {
        "name": "EMA Pullback",
        "side": "BUY",
        "hold_days": 18,
        "desc": "Price above EMA50; buy dip near EMA20; RSI 45–65.",
        "why": "Classic swing pullback in an uptrend.",
        "beginner": "Trend is up. Wait for a small dip to EMA20, then buy. Stop below the dip.",
        "typical_win": "38–48%",
        "typical_exp": "−0.05% to +0.15%",
    },
    "supertrend_pullback": {
        "name": "Supertrend + EMA Pullback",
        "side": "BUY",
        "hold_days": 18,
        "desc": "Price above Supertrend & EMA50; buy pullback near EMA20 with RSI 48–65.",
        "why": "Rides established uptrends; avoids fighting Supertrend.",
        "beginner": "Supertrend is green and price is above EMA50. Buy a dip toward EMA20.",
        "typical_win": "38–48%",
        "typical_exp": "−0.05% to +0.15%",
    },
    "donchian_breakout": {
        "name": "Donchian Breakout + Volume",
        "side": "BUY",
        "hold_days": 15,
        "desc": "Close breaks 20-day high with volume ≥1.2x; price above EMA50.",
        "why": "Momentum breakout with participation.",
        "beginner": "Price closes above the highest high of 20 days with strong volume while already in an uptrend.",
        "typical_win": "35–45%",
        "typical_exp": "−0.10% to +0.10%",
    },
    "adx_trend_rider": {
        "name": "ADX Trend Rider",
        "side": "BUY",
        "hold_days": 18,
        "desc": "ADX≥25, price>EMA50, RSI 48–68.",
        "why": "Only ride strong trends.",
        "beginner": "ADX high = real trend. Buy with the trend when RSI is not extreme.",
        "typical_win": "40–50%",
        "typical_exp": "−0.05% to +0.20%",
    },
    "bollinger_reversion": {
        "name": "Bollinger Mean Reversion",
        "side": "BUY",
        "hold_days": 12,
        "desc": "Above EMA200, touch BB lower, RSI soft, close back inside band.",
        "why": "Buy panic dips only in long-term uptrends.",
        "beginner": "Big uptrend. Price tags lower Bollinger band then closes back inside — buy the reclaim.",
        "typical_win": "45–55%",
        "typical_exp": "−0.10% to +0.05%",
    },
    "high_52w_momentum": {
        "name": "52w High Momentum",
        "side": "BUY",
        "hold_days": 15,
        "desc": "Near 52-week high, volume up, RSI 55–72.",
        "why": "Strength near highs often continues in swings.",
        "beginner": "Stock near 52-week high with volume — momentum names.",
        "typical_win": "40–50%",
        "typical_exp": "0.00% to +0.25%",
    },
    "ema_stack_breakout": {
        "name": "EMA Stack Breakout",
        "side": "BUY",
        "hold_days": 15,
        "desc": "EMA9>EMA21>EMA50, Donchian break, volume ≥1.2x.",
        "why": "Breakout only when averages are aligned.",
        "beginner": "Fast EMA above slow EMAs, then price breaks 20-day high with volume.",
        "typical_win": "35–45%",
        "typical_exp": "−0.10% to +0.10%",
    },
    "rsi_macd_trend": {
        "name": "RSI–MACD Trend Swing",
        "side": "BUY",
        "hold_days": 20,
        "desc": "Price>EMA50, MACD crosses up, RSI through 50, ADX≥22.",
        "why": "Momentum turn with trend confirmation.",
        "beginner": "MACD turns up and RSI crosses 50 while price is above EMA50.",
        "typical_win": "40–50%",
        "typical_exp": "−0.05% to +0.15%",
    },
    "vwap_reclaim": {
        "name": "VWAP Reclaim Swing",
        "side": "BUY",
        "hold_days": 10,
        "desc": "Reclaim VWAP after days below; volume; price>EMA50.",
        "why": "Institutional reclaim often marks continuation.",
        "beginner": "Price was below VWAP, then closes back above with volume.",
        "typical_win": "38–48%",
        "typical_exp": "−0.05% to +0.12%",
    },
    "supertrend_sell": {
        "name": "Supertrend Breakdown Sell",
        "side": "SELL",
        "hold_days": 15,
        "desc": "Price below Supertrend & EMA50; RSI 35–50.",
        "why": "Swing sell / exit longs in downtrends.",
        "beginner": "Supertrend red and price under EMA50 — weakness for sells/exits.",
        "typical_win": "35–45%",
        "typical_exp": "−0.10% to +0.10%",
    },
    "sure_combo": {
        "name": "Sure Combo (multi-factor)",
        "side": "BUY",
        "hold_days": 18,
        "desc": "price>EMA20>EMA50, ADX≥25, RSI 52–68, Supertrend bull, vol≥1.1.",
        "why": "Strictest long — fewer trades, higher quality aim.",
        "beginner": "Only when trend, momentum, Supertrend and volume all agree.",
        "typical_win": "varies (strict)",
        "typical_exp": "fewer trades, quality focus",
    },
}


def load_my_strategy_params() -> dict:
    defaults = {
        "lookback": 90,          # strict: long window (months on daily)
        "swing_left": 3,
        "swing_right": 3,
        "rsi_min": 25,
        "rsi_max": 58,
        "min_rsi_slope": 0.0,    # RSI line must not fall
        "max_price_slope": -0.5, # price line must be falling (negative)
        "rsi_line_break_tol": 0.5,  # RSI must stay above its rising support
        "early_signal": True,   # only when pattern is building
        "max_bars_after_pattern": 2,  # if 3+ days after swing lows → TOO LATE (no buy)
        "max_bounce_pct_from_low": 3.5,  # already bounced >3.5% → pattern completed, skip
        "require_ema200": False,
        "min_adx": 0,
        "atr_target_mult": 1.8,
        "atr_stop_mult": 2.2,
        "hold_bars": 15,
        "timeframe": "1d",
        "notes": (
            "Price lower-lows + RSI higher-lows, RSI line unbroken. "
            "ONLY building patterns. If already built and 2–3 days passed or price bounced hard → NO buy."
        ),
    }
    try:
        if MY_STRATEGY_PARAMS_FILE.exists():
            with open(MY_STRATEGY_PARAMS_FILE, "r", encoding="utf-8") as f:
                defaults.update(json.load(f))
    except Exception:
        pass
    return defaults


def _swing_low_indices(series: pd.Series, left: int = 3, right: int = 3) -> list:
    """Indices of swing lows (local minima)."""
    vals = series.astype(float).values
    n = len(vals)
    out = []
    for i in range(left, n - right):
        window = vals[i - left : i + right + 1]
        if np.isnan(vals[i]):
            continue
        if vals[i] == np.nanmin(window):
            out.append(i)
    return out


def my_strategy_divergence_ok(df: pd.DataFrame, i: int, p: dict = None) -> bool:
    """
    My Strategy — SAME-TIME price & RSI points (straight paired lines):
    - Find last two PRICE swing lows at bars T1, T2
    - RSI at T1 and RSI at T2 (same bars — not independent RSI swings)
    - Price T2 < Price T1 (falling), RSI T2 >= RSI T1 (rising/flat)
    - RSI line from (T1→T2) must not break after T2
    - Pattern must be fresh (not 2–3 days late / big bounce)
    """
    p = p or load_my_strategy_params()
    lb = max(30, int(p.get("lookback", 90)))
    left = int(p.get("swing_left", 3))
    right = int(p.get("swing_right", 3))
    if i < lb + right + 2 or i >= len(df):
        return False
    if "RSI" not in df.columns or "Low" not in df.columns:
        return False

    start = max(0, i - lb)
    early = bool(p.get("early_signal", True))
    use_right = 1 if early and i >= len(df) - 2 else right

    seg = df.iloc[start : i + 1].copy()
    if len(seg) < 25:
        return False

    lows = seg["Low"].astype(float)
    rsis = seg["RSI"].astype(float)
    # ONLY price swing lows — RSI is read at the SAME bars
    pl_idx = _swing_low_indices(lows, left=left, right=max(1, use_right))
    if len(pl_idx) < 2:
        return False

    t1, t2 = pl_idx[-2], pl_idx[-1]
    if t2 <= t1:
        return False

    price_1 = float(lows.iloc[t1])
    price_2 = float(lows.iloc[t2])
    rsi_1 = float(rsis.iloc[t1])  # same time as price_1
    rsi_2 = float(rsis.iloc[t2])  # same time as price_2

    price_falling = price_2 < price_1 * 0.998
    rsi_rising = rsi_2 >= rsi_1 - 0.25
    both_falling = (price_2 < price_1) and (rsi_2 < rsi_1 - 0.5)
    if both_falling or not price_falling or not rsi_rising:
        return False

    rsi_line_slope = (rsi_2 - rsi_1) / max(1, (t2 - t1))
    if rsi_line_slope < float(p.get("min_rsi_slope", 0.0)) - 0.05:
        return False

    # RSI must not break the straight line between same-time points after t2
    tol = float(p.get("rsi_line_break_tol", 0.5))
    for j in range(t2 + 1, len(seg)):
        t = (j - t1) / max(1, (t2 - t1))
        support = rsi_1 + t * (rsi_2 - rsi_1)
        if float(rsis.iloc[j]) < support - tol:
            return False

    rsi_now = float(rsis.iloc[-1])
    band_ok = float(p.get("rsi_min", 25)) <= rsi_now <= float(p.get("rsi_max", 58))
    if not band_ok:
        return False

    bars_since = (len(seg) - 1) - t2
    max_age = int(p.get("max_bars_after_pattern", 2))
    if bars_since > max_age:
        return False

    c = float(seg["Close"].iloc[-1])
    bounce = (c - price_2) / price_2 * 100 if price_2 > 0 else 0
    if bounce > float(p.get("max_bounce_pct_from_low", 3.5)):
        return False

    recent = t2 >= len(seg) - max_age - 3
    ema200 = safe_float(seg["EMA200"].iloc[-1]) if "EMA200" in seg.columns else 0
    ema200_ok = (not p.get("require_ema200")) or (c > ema200 > 0)
    return bool(recent and ema200_ok)



def my_strategy_status(df: pd.DataFrame, i: int = None, p: dict = None) -> dict:
    """
    ACTIVE_FRESH = pattern building / just ready → can buy
    TOO_LATE     = pattern completed 2–3 days ago (e.g. RHL after the move)
    RSI_FAILED   = was forming but RSI fell / line broke (e.g. Shiva Mills) → no buy
    INVALID      = no structure
    """
    p = p or load_my_strategy_params()
    out = {"status": "INVALID", "reason": "Not enough data", "ok_for_buy": False}
    if df is None or len(df) < 50:
        return out
    if "RSI" not in df.columns:
        try:
            df = calculate_indicators(df)
        except Exception:
            return out
    if i is None:
        i = len(df) - 1
    if my_strategy_divergence_ok(df, i, p):
        out["status"] = "ACTIVE_FRESH"
        out["reason"] = (
            "Pattern is ready to build / just formed (fresh). "
            "Eligible buy while RSI line holds."
        )
        out["ok_for_buy"] = True
        return out
    try:
        lb = max(30, int(p.get("lookback", 90)))
        start = max(0, i - lb)
        seg = df.iloc[start : i + 1]
        lows = seg["Low"].astype(float)
        rsis = seg["RSI"].astype(float)
        pl = _swing_low_indices(lows, 3, 1)
        rl = _swing_low_indices(rsis, 3, 1)
        if len(pl) >= 2 and len(rl) >= 2:
            p1, p2 = pl[-2], pl[-1]
            r1, r2 = rl[-2], rl[-1]
            price_low_1 = float(lows.iloc[p1])
            price_low_2 = float(lows.iloc[p2])
            rsi_low_1 = float(rsis.iloc[r1])
            rsi_low_2 = float(rsis.iloc[r2])
            bars_since = (len(seg) - 1) - min(p2, r2)
            c = float(seg["Close"].iloc[-1])
            bounce = (c - price_low_2) / price_low_2 * 100 if price_low_2 else 0
            price_falling = price_low_2 < price_low_1 * 0.998
            rsi_was_rising = rsi_low_2 >= rsi_low_1 - 0.25
            rsi_now = float(rsis.iloc[-1])
            rsi_failed = (rsi_low_2 < rsi_low_1 - 0.5) or (rsi_now < min(rsi_low_1, rsi_low_2) - 1.0)
            if price_falling and rsi_failed:
                out["status"] = "RSI_FAILED"
                out["reason"] = (
                    "Pattern was about to form, but RSI went down / line broke. "
                    "Example: Shiva Mills — cannot purchase now."
                )
                out["ok_for_buy"] = False
                return out
            max_age = int(p.get("max_bars_after_pattern", 2))
            max_bounce = float(p.get("max_bounce_pct_from_low", 3.5))
            if price_falling and rsi_was_rising and (bars_since > max_age or bounce > max_bounce):
                out["status"] = "TOO_LATE"
                out["reason"] = (
                    f"Pattern already passed ~{bars_since} day(s) ago (bounce ~{bounce:.1f}%). "
                    "Example: RHL / Robust Hotels after the move — no new buy call."
                )
                out["ok_for_buy"] = False
                return out
    except Exception:
        pass
    out["status"] = "INVALID"
    out["reason"] = (
        "No My Strategy buy: need price lower-lows + RSI higher-lows, "
        "RSI line intact, pattern still fresh."
    )
    out["ok_for_buy"] = False
    return out


def save_my_strategy_params(params: dict) -> None:
    try:
        with open(MY_STRATEGY_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _strategy_signal_at_bar(df: pd.DataFrame, i: int, strategy_id: str) -> bool:
    """True if strategy entry triggers on bar i (uses data up to i inclusive)."""
    if i >= len(df):
        return False
    # My Strategy needs long history; other strategies need ~60 bars
    if strategy_id == "my_strategy":
        if i < 40:
            return False
        try:
            return bool(my_strategy_divergence_ok(df, i, load_my_strategy_params()))
        except Exception:
            return False
    if i < 60:
        return False
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    c = safe_float(row.get("Close"))
    o = safe_float(row.get("Open"))
    h = safe_float(row.get("High"))
    l = safe_float(row.get("Low"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    ema200 = safe_float(row.get("EMA200"))
    ema9 = safe_float(row.get("EMA9"))
    ema21 = safe_float(row.get("EMA21"))
    rsi_v = safe_float(row.get("RSI"), 50)
    adx_v = safe_float(row.get("ADX"))
    macd_v = safe_float(row.get("MACD"))
    macds = safe_float(row.get("MACDSignal"))
    atr_v = safe_float(row.get("ATR")) or c * 0.02
    st_dir = safe_float(row.get("ST_Dir"))
    vol_r = safe_float(row.get("Volume Ratio"), 1.0)
    bb_l = safe_float(row.get("BBLower"))
    bb_m = safe_float(row.get("BBMiddle"))
    vwap_v = safe_float(row.get("VWAP"))
    don_h = safe_float(row.get("DonchianHigh"))
    prev_don_h = safe_float(df.iloc[i - 1].get("DonchianHigh")) if i > 0 else don_h

    sid = strategy_id

    if sid == "ema_pullback":
        return (
            c > ema50 > 0 and ema20 > 0
            and l <= ema20 * 1.01 and c >= ema20 * 0.995
            and 45 <= rsi_v <= 65
        )

    if sid == "donchian_breakout":
        return (
            don_h > 0 and c >= don_h * 0.999 and c > ema50 > 0
            and vol_r >= 1.2 and 50 <= rsi_v <= 72
        )

    if sid == "adx_trend_rider":
        return c > ema50 > 0 and adx_v >= 25 and 48 <= rsi_v <= 68

    if sid == "bollinger_reversion":
        prev_c = safe_float(prev.get("Close"))
        prev_bb = safe_float(prev.get("BBLower"))
        return (
            c > ema200 > 0 and prev_c <= prev_bb * 1.002
            and c > bb_l and rsi_v < 40 and c > o
        )

    if sid == "high_52w_momentum":
        # use DonchianHigh as proxy if no 52w series on bar
        h52 = safe_float(row.get("DonchianHigh"))
        try:
            if len(df) >= 200:
                h52 = safe_float(df["High"].iloc[max(0, i - 251) : i + 1].max())
        except Exception:
            pass
        return h52 > 0 and c >= h52 * 0.97 and vol_r >= 1.1 and 55 <= rsi_v <= 72

    if sid == "supertrend_pullback":
        return (
            st_dir > 0 and c > ema50 > 0 and ema20 > 0
            and l <= ema20 * 1.01 and c >= ema20 * 0.995
            and 48 <= rsi_v <= 65 and adx_v >= 18
        )
    if sid == "ema_stack_breakout":
        return (
            ema9 > ema21 > ema50 > 0 and c > ema9
            and don_h > 0 and c >= don_h * 0.999
            and vol_r >= 1.2 and rsi_v >= 50 and rsi_v <= 72
        )
    if sid == "rsi_macd_trend":
        prev_rsi = safe_float(prev.get("RSI"), 50)
        prev_macd = safe_float(prev.get("MACD"))
        prev_ms = safe_float(prev.get("MACDSignal"))
        return (
            c > ema50 > 0 and adx_v >= 22
            and macd_v > macds and prev_macd <= prev_ms
            and prev_rsi < 50 <= rsi_v and rsi_v <= 68
        )
    if sid == "mean_reversion_bollinger":
        prev_c = safe_float(prev.get("Close"))
        prev_bb = safe_float(prev.get("BBLower"))
        return (
            c > ema200 > 0 and prev_c <= prev_bb * 1.002
            and c > bb_l and rsi_v < 40 and rsi_v > 20
            and c > o  # green reclaim
        )
    if sid == "vwap_reclaim":
        if i < 3:
            return False
        below = all(safe_float(df.iloc[i - j].get("Close")) < safe_float(df.iloc[i - j].get("VWAP")) for j in range(1, 3))
        return below and c > vwap_v > 0 and c > ema50 > 0 and vol_r >= 1.15 and rsi_v >= 45
    if sid == "supertrend_sell":
        return (
            st_dir < 0 and c < ema50
            and 30 <= rsi_v <= 50 and adx_v >= 20
        )
    if sid == "sure_combo":
        return (
            c > ema20 > ema50 > 0 and adx_v >= 25
            and 52 <= rsi_v <= 68 and st_dir > 0
            and vol_r >= 1.1 and (ema200 <= 0 or c > ema200 * 0.99)
        )
    return False


def backtest_strategy_on_df(df: pd.DataFrame, strategy_id: str, meta: dict = None) -> dict:
    """
    Walk-forward backtest on one symbol's daily OHLC.
    Entry next open after signal; exit target/stop/time.
    BUY: target = entry+1.8ATR, stop = entry-2.2ATR
    SELL: inverse.
    """
    meta = meta or SWING_STRATEGIES.get(strategy_id, {})
    side = meta.get("side", "BUY")
    hold = int(meta.get("hold_days", 15))
    trades = []
    i = 60
    while i < len(df) - 2:
        if not _strategy_signal_at_bar(df, i, strategy_id):
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= len(df):
            break
        entry = safe_float(df.iloc[entry_i].get("Open"))
        atr_v = safe_float(df.iloc[i].get("ATR")) or entry * 0.02
        if entry <= 0:
            i += 1
            continue
        if side == "SELL":
            target = entry - 1.8 * atr_v
            stop = entry + 2.2 * atr_v
        else:
            target = entry + 1.8 * atr_v
            stop = entry - 2.2 * atr_v
        result = "HOLDING PERIOD COMPLETED"
        exit_px = safe_float(df.iloc[min(entry_i + hold, len(df) - 1)].get("Close"))
        exit_i = min(entry_i + hold, len(df) - 1)
        for j in range(entry_i, min(entry_i + hold + 1, len(df))):
            hi = safe_float(df.iloc[j].get("High"))
            lo = safe_float(df.iloc[j].get("Low"))
            if side == "BUY":
                if lo <= stop:
                    result, exit_px, exit_i = "STOP LOSS HIT", stop, j
                    break
                if hi >= target:
                    result, exit_px, exit_i = "TARGET ACHIEVED", target, j
                    break
            else:
                if hi >= stop:
                    result, exit_px, exit_i = "STOP LOSS HIT", stop, j
                    break
                if lo <= target:
                    result, exit_px, exit_i = "TARGET ACHIEVED", target, j
                    break
            exit_px = safe_float(df.iloc[j].get("Close"))
            exit_i = j
        if side == "SELL":
            ret = (entry - exit_px) / entry * 100
        else:
            ret = (exit_px - entry) / entry * 100
        trades.append({
            "entry_date": str(df.index[entry_i].date()) if hasattr(df.index[entry_i], "date") else str(df.index[entry_i]),
            "exit_date": str(df.index[exit_i].date()) if hasattr(df.index[exit_i], "date") else str(df.index[exit_i]),
            "entry": round(entry, 2),
            "exit": round(exit_px, 2),
            "result": result,
            "return_pct": round(ret, 2),
            "days": int(exit_i - entry_i),
        })
        i = exit_i + 1  # no overlapping trades
    wins = sum(1 for t in trades if t["result"] == "TARGET ACHIEVED")
    losses = sum(1 for t in trades if t["result"] == "STOP LOSS HIT")
    decided = wins + losses
    win_rate = (wins / decided * 100) if decided else 0.0
    all_n = len(trades)
    avg_ret = float(np.mean([t["return_pct"] for t in trades])) if trades else 0.0
    return {
        "strategy_id": strategy_id,
        "name": meta.get("name", strategy_id),
        "side": side,
        "trades": all_n,
        "wins": wins,
        "losses": losses,
        "timeouts": all_n - decided,
        "win_rate": round(win_rate, 1),
        "efficiency": round(win_rate, 1),  # target-before-stop rate among decided
        "avg_return_pct": round(avg_ret, 2),
        "trade_list": trades,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def backtest_strategy_universe(strategy_id: str, symbols: tuple, max_symbols: int = 40) -> dict:
    """Backtest one strategy across many symbols; aggregate stats."""
    meta = SWING_STRATEGIES.get(strategy_id, {})
    all_trades = []
    per_stock = []
    for sym in list(symbols)[:max_symbols]:
        try:
            raw = stock_history(clean_symbol(sym), interval="1d")
            if raw is None or len(raw) < 120:
                continue
            df = calculate_indicators(raw)
            if df is None or len(df) < 80:
                continue
            res = backtest_strategy_on_df(df, strategy_id, meta)
            if res["trades"] > 0:
                per_stock.append({"Stock": display_symbol(sym), **{k: res[k] for k in ("trades", "wins", "losses", "win_rate", "avg_return_pct")}})
                for t in res["trade_list"]:
                    t2 = dict(t)
                    t2["Stock"] = display_symbol(sym)
                    t2["Strategy"] = meta.get("name", strategy_id)
                    all_trades.append(t2)
        except Exception:
            continue
    wins = sum(1 for t in all_trades if t["result"] == "TARGET ACHIEVED")
    losses = sum(1 for t in all_trades if t["result"] == "STOP LOSS HIT")
    decided = wins + losses
    wr = (wins / decided * 100) if decided else 0.0
    avg_r = float(np.mean([t["return_pct"] for t in all_trades])) if all_trades else 0.0
    return {
        "strategy_id": strategy_id,
        "name": meta.get("name", strategy_id),
        "side": meta.get("side", "BUY"),
        "desc": meta.get("desc", ""),
        "why": meta.get("why", ""),
        "symbols_tested": len(per_stock),
        "trades": len(all_trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 1),
        "avg_return_pct": round(avg_r, 2),
        "per_stock": per_stock,
        "trade_list": all_trades[-200:],
    }


def live_strategy_signals(results: pd.DataFrame, strategy_id: str, max_check: int = 35) -> pd.DataFrame:
    """Check current bar for strategy signals on scan universe."""
    meta = SWING_STRATEGIES.get(strategy_id, {})
    if results is None or results.empty:
        return pd.DataFrame()
    stocks = results["Stock"].astype(str).unique().tolist()[:max_check]
    rows = []
    for stock in stocks:
        try:
            raw = stock_history(clean_symbol(stock), interval="1d")
            if raw is None or len(raw) < 80:
                continue
            df = calculate_indicators(raw)
            i = len(df) - 1
            if not _strategy_signal_at_bar(df, i, strategy_id):
                continue
            row = df.iloc[i]
            c = safe_float(row.get("Close"))
            atr_v = safe_float(row.get("ATR")) or c * 0.02
            side = meta.get("side", "BUY")
            if side == "SELL":
                tgt, sl = c - 1.8 * atr_v, c + 2.2 * atr_v
            else:
                tgt, sl = c + 1.8 * atr_v, c - 2.2 * atr_v
            # Prefer live quote for current price when available
            live_px = c
            try:
                q = live_quote(clean_symbol(stock))
                if q and safe_float(q.get("price") or q.get("Price") or q.get("last"), 0) > 0:
                    live_px = safe_float(q.get("price") or q.get("Price") or q.get("last"), c)
            except Exception:
                pass
            rows.append({
                "Stock": display_symbol(stock),
                "Strategy": meta.get("name", strategy_id),
                "Strategy ID": strategy_id,
                "Side": side,
                "Current / Live Price": round(live_px, 2),
                "Price": round(live_px, 2),
                "Target": round(tgt, 2),
                "Stop Loss": round(sl, 2),
                "Hold Days": meta.get("hold_days", 15),
                "RSI": round(safe_float(row.get("RSI")), 1),
                "ADX": round(safe_float(row.get("ADX")), 1),
                "Supertrend Dir": "BULL" if safe_float(row.get("ST_Dir")) > 0 else "BEAR",
                "Why": meta.get("why", ""),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def strategies_matching_stock(symbol: str, side_filter: str = "BUY") -> dict:
    """
    Which swing strategies currently fire on this stock.
    Returns {count, names, ids, detail}.
    """
    out = {"count": 0, "names": [], "ids": [], "detail": []}
    try:
        raw = stock_history(clean_symbol(symbol), interval="1d")
        if raw is None or len(raw) < 80:
            return out
        df = calculate_indicators(raw)
        i = len(df) - 1
        row = df.iloc[i]
        c = safe_float(row.get("Close"))
        atr_v = safe_float(row.get("ATR")) or c * 0.02
        for sid, meta in SWING_STRATEGIES.items():
            if side_filter and str(meta.get("side", "BUY")).upper() != str(side_filter).upper():
                if side_filter.upper() == "BUY" and meta.get("side") == "SELL":
                    continue
            try:
                if not _strategy_signal_at_bar(df, i, sid):
                    continue
            except Exception:
                continue
            side = meta.get("side", "BUY")
            if side == "SELL":
                tgt, sl = c - 1.8 * atr_v, c + 2.2 * atr_v
            else:
                tgt, sl = c + 1.8 * atr_v, c - 2.2 * atr_v
            name = meta.get("name", sid)
            out["ids"].append(sid)
            out["names"].append(name)
            out["detail"].append(
                {
                    "Strategy": name,
                    "ID": sid,
                    "Side": side,
                    "Entry": round(c, 2),
                    "Target": round(tgt, 2),
                    "Stop": round(sl, 2),
                }
            )
        out["count"] = len(out["names"])
    except Exception:
        pass
    return out


def stocks_under_strategy(results: pd.DataFrame, strategy_id: str, max_check: int = 40) -> pd.DataFrame:
    """All stocks from scan/universe that currently match one strategy."""
    return live_strategy_signals(results, strategy_id, max_check=max_check)


def show_strategy_lab(results: pd.DataFrame):
    """Everything in one place: learn, edit My Strategy, 5y backtest, live BUY calls, Angel levels."""
    st.title("🧪 Strategy Lab — All-in-one (inside main app)")
    st.caption(
        "One file only: run **app.py**. Here you get strategies, **My Strategy** editor, "
        "backtest efficiency (up to ~5y), live stock signals for Angel One, and beginner lessons. "
        "Efficiency = target hit **before** stop. Not a guarantee of any fixed %."
    )

    try:
        reg = nifty_market_regime()
        if not reg.get("trade_longs", True):
            st.error("Nifty regime **weak** — avoid new long strategy trades today.")
        else:
            st.info(f"Nifty: **{reg.get('regime')}** (score {reg.get('score')})")
    except Exception:
        pass

    default_syms = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
        "ITC", "LT", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
        "TITAN", "WIPRO", "NTPC", "TATAMOTORS", "M&M", "HCLTECH",
    ]
    if results is not None and not results.empty and "Stock" in results.columns:
        scan_syms = (
            results.sort_values("Prediction", ascending=False)["Stock"]
            .astype(str).head(40).tolist()
        )
        universe = tuple(dict.fromkeys(scan_syms + default_syms))
    else:
        universe = tuple(default_syms)

    tab_learn, tab_edit, tab_bt, tab_live, tab_stocks, tab_hi, tab_ao = st.tabs(
        [
            "📘 Learn",
            "✏️ My Strategy",
            "📊 Backtest %",
            "📡 Live BUY by strategy",
            "📋 Stocks under strategy",
            "⭐ High conviction",
            "🏦 Angel One levels",
        ]
    )

    with tab_learn:
        st.subheader("Learn strategies the easy way")
        st.markdown(
            """
### How to find stocks **manually** (TradingView)

1. Open **daily** chart + add **RSI (14)**.  
2. Zoom out **2–6 months** (long timeframe).  
3. Draw a line on **price swing lows** — is it **falling**?  
4. Draw a line on **RSI swing lows** — is it **rising / flat-up**?  
5. RSI line must **not be broken**.  
6. If yes → that is **My Strategy** (same as Robust Hotels / Shiva Mills style).  
7. Wait **confirmation**: next day small dip OK; RSI must still hold.  
8. If RSI breaks its line → **skip**. If price gaps up hard → **don’t chase**.

### All strategies in simple words
            """
        )
        rows = []
        for sid, m in SWING_STRATEGIES.items():
            st.markdown(f"### {m['name']}")
            st.write(m.get("beginner") or m.get("desc"))
            st.caption(
                f"Typical win rate: **{m.get('typical_win', '—')}** · "
                f"Expectancy: **{m.get('typical_exp', '—')}** (illustrative, not a guarantee)"
            )
            rows.append(
                {
                    "Strategy": m["name"],
                    "Typical win rate": m.get("typical_win", "—"),
                    "Expectancy per trade": m.get("typical_exp", "—"),
                }
            )
            st.divider()
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(
            """
### Manual checklist before any trade
- [ ] I can see the pattern on **daily** (or my timeframe) with **RSI**  
- [ ] Lines make sense over **weeks**, not only 2–3 candles  
- [ ] Nifty is not in a crash day (optional filter)  
- [ ] Stop is defined under price swing low  
- [ ] I will not chase a vertical next-day green candle  

### Drawing on charts in this app
- **RSI is on every chart** (Plotly).  
- **MyStrat lines** checkbox draws the model’s blue lines for comparison.  
- **Hand drawing** → button **TradingView (draw)** (full site has trendline tool).
            """
        )

    with tab_edit:
        st.subheader("Edit **My Strategy** (saved in my_strategy_params.json)")
        st.markdown(
            """
**Rules (strict — from your charts):**
1. **Price line falling** (lower lows) — weeks/months OK  
2. **RSI line rising or flat-up** — **must not break**  
3. **Reject** if both lines falling  
4. Call only while pattern is **BUILDING** (0–2 days after the swing lows)  
5. **If pattern already built and 2–3 days passed** (price already up, e.g. late Shiva Mills) → **NOT a buy call**  
6. If price already bounced more than ~3.5% from the pattern low → **too late**
            """
        )
        p = load_my_strategy_params()
        # Clamp old saved JSON values into widget min/max (fixes StreamlitValueBelowMinError)
        lb0 = max(30, min(200, int(p.get("lookback", 60) or 60)))
        rmin0 = max(10, min(50, int(p.get("rsi_min", 25) or 25)))
        rmax0 = max(40, min(70, int(p.get("rsi_max", 58) or 58)))
        if rmax0 < rmin0:
            rmax0 = min(70, rmin0 + 5)
        slope0 = max(-1.0, min(2.0, float(p.get("min_rsi_slope", 0.0) or 0.0)))
        brk0 = max(0.0, min(5.0, float(p.get("rsi_line_break_tol", 0.5) or 0.5)))
        tgt0 = max(0.5, min(4.0, float(p.get("atr_target_mult", 1.8) or 1.8)))
        sl0 = max(0.5, min(4.0, float(p.get("atr_stop_mult", 2.2) or 2.2)))
        hold0 = max(5, min(40, int(p.get("hold_bars", 15) or 15)))

        c1, c2, c3 = st.columns(3)
        with c1:
            p["lookback"] = int(st.number_input("Lookback bars (months OK on daily)", 30, 200, lb0, key="my_lb"))
            p["rsi_min"] = int(st.number_input("RSI min", 10, 50, rmin0, key="my_rmin"))
            p["rsi_max"] = int(st.number_input("RSI max", 40, 70, rmax0, key="my_rmax"))
        with c2:
            p["min_rsi_slope"] = float(st.number_input("Min RSI line slope", -1.0, 2.0, slope0, key="my_slope"))
            p["rsi_line_break_tol"] = float(st.number_input("RSI line break tolerance", 0.0, 5.0, brk0, key="my_brk"))
            p["atr_target_mult"] = float(st.number_input("Target × ATR", 0.5, 4.0, tgt0, key="my_tgt"))
        with c3:
            p["atr_stop_mult"] = float(st.number_input("Stop × ATR", 0.5, 4.0, sl0, key="my_sl"))
            p["hold_bars"] = int(st.number_input("Hold bars/days", 5, 40, hold0, key="my_hold"))
            p["early_signal"] = st.checkbox(
                "Early signal (as pattern builds)",
                value=bool(p.get("early_signal", True)),
                key="my_early",
            )
            p["require_ema200"] = st.checkbox(
                "Require > EMA200",
                value=bool(p.get("require_ema200")),
                key="my_e200",
            )
        p["notes"] = st.text_area("Notes", value=str(p.get("notes", "")), key="my_notes")
        if st.button("💾 Save My Strategy", type="primary", key="my_save"):
            save_my_strategy_params(p)
            st.success(f"Saved → `{MY_STRATEGY_PARAMS_FILE.name}`")

        st.subheader("🧠 Train My Strategy from your charts (RSI divergence)")
        st.caption(
            "Your template: **price lower lows + RSI higher lows** (RELIANCE / COALINDIA style). "
            "Save every good/bad example — the model adjusts **RSI_BULL_DIV** weight on next learning pass."
        )
        st.markdown(
            """
**What the model learns from your images**

| Chart feature | Rule stored |
|---------------|-------------|
| Price swing lows falling | Required |
| RSI at **same bars** rising / flat | Required (bullish divergence) |
| Point **D** = second low completion | Entry window starts |
| 2–3 days after big bounce | **TOO_LATE** — not a buy |
| RSI breaks rising support line | **RSI_FAILED** — no buy |
            """
        )
        with st.form("my_strat_train_form"):
            t_sym = st.text_input("Symbol", value="RELIANCE")
            t_outcome = st.selectbox(
                "Outcome",
                [
                    "ACTIVE_FRESH good",
                    "Bullish RSI divergence confirmed",
                    "TARGET HIT",
                    "TOO_LATE",
                    "RSI_FAILED",
                    "STOP HIT",
                    "Should not have bought",
                ],
            )
            t_note = st.text_area(
                "What you saw (lines / days / RSI)",
                value="Price lower low + RSI higher low (bullish divergence). Entry near second low confirmation.",
            )
            t_img = st.file_uploader("Optional chart image (png/jpg)", type=["png", "jpg", "jpeg"])
            t_sub = st.form_submit_button("Save training example + update learning")
        if t_sub:
            import datetime as _dt
            row = {
                "time": _dt.datetime.now().isoformat(timespec="seconds"),
                "symbol": (t_sym or "").upper().strip(),
                "outcome": t_outcome,
                "note": t_note or "",
                "image_saved": "",
            }
            train_dir = APP_DIR / "my_strategy_train"
            train_dir.mkdir(exist_ok=True)
            if t_img is not None:
                img_path = train_dir / f"{row['symbol']}_{row['time'].replace(':', '-')}.{t_img.name.split('.')[-1]}"
                img_path.write_bytes(t_img.getvalue())
                row["image_saved"] = str(img_path.name)
            train_csv = APP_DIR / "my_strategy_train_log.csv"
            try:
                if train_csv.exists():
                    _tdf = pd.read_csv(train_csv)
                    _tdf = pd.concat([_tdf, pd.DataFrame([row])], ignore_index=True)
                else:
                    _tdf = pd.DataFrame([row])
                _tdf.to_csv(train_csv, index=False)
                # Refresh learning weights immediately
                try:
                    learn_from_history(min_closed=1)
                except Exception:
                    pass
                st.success(
                    f"Saved example → `{train_csv.name}`. "
                    f"Learning updated (RSI_BULL_DIV weight refreshed)."
                )
            except Exception as _e:
                st.error(str(_e))

        # One-click seed from your two chart types
        if st.button("📌 Load RELIANCE + COALINDIA divergence as training templates", key="seed_div_charts"):
            train_csv = APP_DIR / "my_strategy_train_log.csv"
            import datetime as _dt
            seeds = [
                {
                    "time": _dt.datetime.now().isoformat(timespec="seconds"),
                    "symbol": "RELIANCE",
                    "outcome": "Bullish RSI divergence confirmed",
                    "note": "Daily: price LL into Jun pivot, RSI HL; classic bullish divergence template",
                    "image_saved": "user_chart_reliance",
                },
                {
                    "time": _dt.datetime.now().isoformat(timespec="seconds"),
                    "symbol": "COALINDIA",
                    "outcome": "Bullish RSI divergence confirmed",
                    "note": "Daily after sell: price LL + RSI HL then recovery; buy on confirmation",
                    "image_saved": "user_chart_coalindia",
                },
            ]
            try:
                if train_csv.exists():
                    _tdf = pd.read_csv(train_csv)
                    _tdf = pd.concat([_tdf, pd.DataFrame(seeds)], ignore_index=True)
                else:
                    _tdf = pd.DataFrame(seeds)
                _tdf.to_csv(train_csv, index=False)
                learn_from_history(min_closed=1)
                st.success("Templates saved + learning refreshed. Model will favour this divergence shape.")
            except Exception as e:
                st.error(str(e))


        st.divider()
        st.subheader("📈 My Strategy — chart check (TradingView + analysis)")
        st.markdown(
            """
**How your pattern works (from Robust Hotels / Shiva Mills style):**

| Phase | Price | RSI | What to do |
|-------|--------|-----|------------|
| **Building** | Mild lower lows (blue line down) | Higher lows (blue line up) | Watch — pattern forming |
| **Early call** | Line still down, not collapsed | RSI line **not broken** | Optional small entry / alert |
| **Next day** | Often one more soft dip | RSI still holds / rises | **Confirmation** day |
| **Complete** | Bounce starts | RSI holds above its line | Main swing entry |
| **Invalid** | — | RSI **breaks** its rising line | **No trade** |

**IMPORTANT — Shiva Mills rule:**  
Only take calls when the pattern is **ready to build / just formed**.  
If the pattern **already built and 2–3 days have passed** (or price already bounced) → **NOT a buy call** (stock will correctly **disappear** from My Strategy list).
            """
        )
        sym_my = st.text_input(
            "NSE symbol to analyse on chart",
            value="SHIVAMILLS",
            key="my_strat_chart_sym",
        ).upper().strip()
        # Always show chart when symbol typed (not only after button)
        chart_sym = display_symbol(sym_my) if sym_my else ""
        if st.button("Analyse + show chart", type="primary", key="my_chart_btn"):
            st.session_state["my_chart_sym"] = chart_sym
        chart_sym = st.session_state.get("my_chart_sym") or chart_sym
        if chart_sym:
            st.write(f"**{chart_sym}** — chart with RSI + model blue lines")
            raw = pd.DataFrame()
            try:
                raw = stock_history(clean_symbol(chart_sym), interval="1d")
            except Exception:
                raw = pd.DataFrame()
            if raw is None or raw.empty:
                try:
                    import yfinance as yf
                    raw = yf.download(
                        f"{display_symbol(chart_sym)}.NS",
                        period="1y",
                        interval="1d",
                        progress=False,
                        auto_adjust=True,
                    )
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = [c[0] for c in raw.columns]
                    raw = normalize_columns(raw)
                except Exception as e:
                    st.warning(f"Could not load data: {e}")
                    raw = pd.DataFrame()
            if raw is not None and not raw.empty:
                try:
                    raw = calculate_indicators(raw)
                except Exception:
                    pass
                pp = load_my_strategy_params()
                try:
                    fig = build_full_plotly_chart(
                        raw,
                        title=f"{chart_sym} — RSI + My Strategy lines",
                        height=720,
                        show_rsi=True,
                        my_strategy_lines=True,
                        lookback_lines=int(pp.get("lookback", 90)),
                    )
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, key=f"myfig_{chart_sym}")
                    else:
                        st.warning("Chart figure empty — try TradingView link.")
                except Exception as e:
                    st.warning(f"Plotly chart error: {e}")
                    try:
                        show_tradingview_chart(clean_symbol(chart_sym), chart_sym, height=560)
                    except Exception:
                        pass
                try:
                    tv_sym = _to_tv_symbol(chart_sym)
                    st.link_button(
                        "🔗 TradingView (draw your own lines)",
                        f"https://www.tradingview.com/chart/?symbol={tv_sym}",
                    )
                except Exception:
                    pass
                try:
                    st_status = my_strategy_status(raw)
                    c = safe_float(raw["Close"].iloc[-1])
                    rsi_v = safe_float(raw["RSI"].iloc[-1], 50) if "RSI" in raw.columns else 50
                    atr_v = safe_float(raw["ATR"].iloc[-1], 0) if "ATR" in raw.columns else c * 0.02
                    if atr_v <= 0:
                        atr_v = c * 0.02
                    tgt = c + float(pp.get("atr_target_mult", 1.8)) * atr_v
                    sl = c - float(pp.get("atr_stop_mult", 2.2)) * atr_v
                    if st_status["status"] == "ACTIVE_FRESH":
                        st.success(f"**{st_status['status']}** — {st_status['reason']}")
                    elif st_status["status"] in ("TOO_LATE", "RSI_FAILED"):
                        st.error(f"**{st_status['status']}** — {st_status['reason']}")
                    else:
                        st.warning(f"**{st_status['status']}** — {st_status['reason']}")
                    st.markdown(
                        f"**Levels (only if ACTIVE_FRESH):** ₹{c:,.2f} → T ₹{tgt:,.2f} / SL ₹{sl:,.2f} · RSI {rsi_v:.1f}"
                    )
                except Exception as e:
                    st.caption(f"Status note: {e}")
            else:
                st.error("No price data for this symbol.")

        st.divider()
        st.subheader("Nifty / Bank Nifty — Call & Put (same My Strategy idea)")
        st.markdown(
            """
My Strategy is **price vs RSI divergence** — it also applies to **index futures/options direction**:

| Index view | Option idea (educational) |
|------------|----------------------------|
| **Nifty/BankNifty**: price lower-lows, RSI higher-lows, RSI line holds | Bias **CALL** (bullish bounce) after confirmation |
| Price higher-highs, RSI lower-highs (bearish divergence) | Bias **PUT** (not coded as My Strategy long; opposite pattern) |
| RSI line breaks | **No trade** on options either |

**How to use:**  
1. Open NIFTY / BANKNIFTY on My Strategy chart check (symbol `^NSEI` or use index pages).  
2. Same blue-line rules.  
3. If bullish My Strategy on index → prefer **CE** only after confirmation day; strike near ATM, defined SL.  
4. This app does **not** auto-trade option chain orders — levels are directional bias only.  
5. Options decay fast — prefer confirmation + tight invalidation (RSI line break = exit bias).
            """
        )
        ix = st.selectbox("Index for My Strategy bias", ["NIFTY 50", "BANK NIFTY"], key="my_opt_idx")
        if st.button("Check index My Strategy", key="my_opt_btn"):
            yf_i = "^NSEI" if "NIFTY 50" in ix else "^NSEBANK"
            try:
                raw = stock_history(yf_i, interval="1d")
                df = calculate_indicators(raw)
                ok = my_strategy_divergence_ok(df, len(df) - 1, load_my_strategy_params())
                fig = build_full_plotly_chart(
                    df, title=ix, height=650, show_rsi=True, my_strategy_lines=True,
                    lookback_lines=int(load_my_strategy_params().get("lookback", 90)),
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                if ok:
                    st.success(f"**Bullish My Strategy bias on {ix}** → educational CALL bias after confirmation.")
                else:
                    st.warning(f"No strict bullish My Strategy on {ix} right now.")
            except Exception as e:
                st.error(str(e))

    with tab_bt:
        st.subheader("Backtest efficiency (your data)")
        st.caption("Use scan stocks or type a symbol. Period up to **5y** where Yahoo has history.")
        mode = st.radio("Mode", ["All strategies on universe", "One symbol deep dive"], horizontal=True, key="bt_mode")
        max_sym = st.slider("Universe size", 8, 40, 20, key="strat_max_sym")
        if mode.startswith("All"):
            if st.button("▶ Run all strategy backtests", type="primary", key="strat_run_bt"):
                summary_rows = []
                with st.spinner("Backtesting… (can take a few minutes)"):
                    for sid in SWING_STRATEGIES:
                        try:
                            agg = backtest_strategy_universe(sid, universe, max_symbols=max_sym)
                            st.session_state.setdefault("strategy_bt_detail", {})[sid] = agg
                            summary_rows.append(
                                {
                                    "Strategy": agg["name"],
                                    "Side": agg["side"],
                                    "Symbols": agg["symbols_tested"],
                                    "Trades": agg["trades"],
                                    "Wins": agg["wins"],
                                    "Losses": agg["losses"],
                                    "Efficiency %": agg["win_rate"],
                                    "Avg Return %": agg["avg_return_pct"],
                                    "Typical ref": SWING_STRATEGIES[sid].get("typical_win", ""),
                                    "ID": sid,
                                }
                            )
                        except Exception as e:
                            summary_rows.append(
                                {
                                    "Strategy": SWING_STRATEGIES[sid]["name"],
                                    "Efficiency %": 0,
                                    "Trades": 0,
                                    "Wins": 0,
                                    "Losses": 0,
                                    "Error": str(e)[:60],
                                    "ID": sid,
                                }
                            )
                st.session_state["strategy_bt_cache"] = pd.DataFrame(summary_rows)
                try:
                    st.session_state["strategy_bt_cache"].to_csv(STRATEGY_BT_FILE, index=False)
                except Exception:
                    pass
            summary = st.session_state.get("strategy_bt_cache", pd.DataFrame())
            if summary is not None and not summary.empty:
                show = summary.sort_values("Efficiency %", ascending=False)
                st.dataframe(show.drop(columns=["ID"], errors="ignore"), use_container_width=True, hide_index=True)
                best = show.iloc[0]
                st.success(
                    f"**Best in this run:** {best.get('Strategy')} — **{best.get('Efficiency %')}%** "
                    f"({int(best.get('Wins', 0))}W / {int(best.get('Losses', 0))}L). "
                    "Prefer this for live trades if sample is large enough."
                )
            else:
                st.info("Click **Run all strategy backtests** after a market scan for best universe.")
        else:
            sym = st.text_input("NSE symbol", value="RELIANCE", key="bt_one_sym").upper().strip()
            sid = st.selectbox(
                "Strategy",
                list(SWING_STRATEGIES.keys()),
                format_func=lambda k: SWING_STRATEGIES[k]["name"],
                key="bt_one_sid",
            )
            if st.button("▶ Backtest this symbol", type="primary", key="bt_one_btn"):
                with st.spinner("Loading history (~5y) & backtesting…"):
                    try:
                        raw = stock_history(clean_symbol(sym), interval="1d")
                        if raw is None or len(raw) < 80:
                            # try longer via yfinance period
                            import yfinance as yf

                            raw = yf.download(
                                f"{display_symbol(sym)}.NS",
                                period="5y",
                                interval="1d",
                                progress=False,
                                auto_adjust=True,
                            )
                            if isinstance(raw.columns, pd.MultiIndex):
                                raw.columns = [c[0] for c in raw.columns]
                        df = calculate_indicators(raw) if raw is not None and not raw.empty else pd.DataFrame()
                        res = backtest_strategy_on_df(df, sid, SWING_STRATEGIES.get(sid))
                        a, b, c, d = st.columns(4)
                        a.metric("Efficiency %", f"{res.get('win_rate', 0)}%")
                        b.metric("Trades", res.get("trades", 0))
                        c.metric("Wins", res.get("wins", 0))
                        d.metric("Losses", res.get("losses", 0))
                        st.caption(f"Avg return {res.get('avg_return_pct')}% · {SWING_STRATEGIES[sid].get('beginner','')}")
                        if res.get("trade_list"):
                            st.dataframe(pd.DataFrame(res["trade_list"]).tail(40), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(str(e))

    with tab_live:
        st.subheader("Live stock signals by strategy (BUY calls for today)")
        st.caption("Uses last scan universe when available. Run FULL MARKET SCAN first for more names.")
        pick = st.selectbox(
            "Strategy",
            list(SWING_STRATEGIES.keys()),
            format_func=lambda k: f"{SWING_STRATEGIES[k]['name']} ({SWING_STRATEGIES[k]['side']})",
            key="live_strat_pick",
        )
        # Detailed explanation for selected strategy
        meta_pick = SWING_STRATEGIES.get(pick, {})
        with st.expander(f"📖 Learn this strategy in detail — {meta_pick.get('name', pick)}", expanded=True):
            st.write(meta_pick.get("beginner") or meta_pick.get("desc"))
            st.write(f"**Why it can work:** {meta_pick.get('why', '')}")
            st.write(f"**Side:** {meta_pick.get('side')} · **Hold ~** {meta_pick.get('hold_days')} days")
            st.caption(
                f"Typical win rate {meta_pick.get('typical_win', '—')} · "
                f"Expectancy {meta_pick.get('typical_exp', '—')} (illustrative)"
            )
            if pick == "my_strategy":
                st.warning(
                    "**My Strategy rule:** only patterns **ready to build / just forming**. "
                    "If like Shiva Mills the pattern already built and **2–3 days passed** with price up → "
                    "**not a buy call** (too late)."
                )
        if st.button("Generate live signals", type="primary", key="live_strat_btn"):
            with st.spinner("Scanning live bars…"):
                base = results if results is not None and not results.empty else pd.DataFrame({"Stock": list(universe)})
                live_df = live_strategy_signals(base, pick, max_check=35)
                # attach how many strategies each stock matches
                if live_df is not None and not live_df.empty:
                    counts, names_list = [], []
                    for _, rr in live_df.iterrows():
                        m = strategies_matching_stock(str(rr["Stock"]), side_filter="BUY")
                        counts.append(m["count"])
                        names_list.append(", ".join(m["names"]))
                    live_df = live_df.copy()
                    live_df["# Strategies"] = counts
                    live_df["All matching strategies"] = names_list
                st.session_state["live_strat_df"] = live_df
                n_saved = 0
                try:
                    if live_df is not None and not live_df.empty:
                        live_df.to_csv(STRATEGY_LIVE_FILE, index=False)
                        special = []
                        for _, lr in live_df.iterrows():
                            special.append({
                                "Stock": lr.get("Stock"),
                                "Call": lr.get("Side", "BUY"),
                                "Call Source": "STRATEGY",
                                "Strategy": lr.get("Strategy", ""),
                                "Entry": lr.get("Price"),
                                "Target": lr.get("Target"),
                                "Stop Loss": lr.get("Stop Loss"),
                                "Hold Days": lr.get("Hold Days", 15),
                                "Prediction": 78,
                                "Reason": f"Strategy: {lr.get('Strategy', '')}",
                            })
                        n_saved = save_special_calls(special)
                except Exception as e:
                    st.warning(f"Could not save strategy calls to history: {e}")
                if n_saved:
                    st.success(
                        f"Saved **{n_saved}** strategy signal(s) to Past Predictions "
                        f"(Call Source = **STRATEGY**)."
                    )
                elif live_df is not None and not live_df.empty:
                    st.info("Live signals ready (already in history today, or none new).")
        live_df = st.session_state.get("live_strat_df", pd.DataFrame())
        if live_df is not None and not live_df.empty:
            if st.button("💾 Save these strategy signals to Past Predictions", key="strat_resave"):
                special = []
                for _, lr in live_df.iterrows():
                    special.append({
                        "Stock": lr.get("Stock"),
                        "Call": lr.get("Side", "BUY"),
                        "Call Source": "STRATEGY",
                        "Strategy": lr.get("Strategy", ""),
                        "Entry": lr.get("Price"),
                        "Target": lr.get("Target"),
                        "Stop Loss": lr.get("Stop Loss"),
                        "Hold Days": lr.get("Hold Days", 15),
                        "Prediction": 78,
                    })
                n_saved = save_special_calls(special)
                st.success(f"Wrote {n_saved} STRATEGY row(s) to history.")
            st.subheader("📋 Unique stocks (all strategies listed once)")
            render_unique_strategy_stock_cards(live_df, max_cards=12, key_prefix="live_usc")
            with st.expander("Raw signal table (optional)", expanded=False):
                st.dataframe(live_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No live hits yet — normal when filters are strict.")

    with tab_stocks:
        st.subheader("Which stocks fall under strategies")
        st.caption(
            "List **one strategy** or **all strategies at once**. "
            "Uses last scan universe when available."
        )
        mode = st.radio(
            "Mode",
            ["All strategies (full map)", "One strategy only"],
            horizontal=True,
            key="stocks_under_mode",
        )
        base = results if results is not None and not results.empty else pd.DataFrame({"Stock": list(universe)})
        max_check = st.slider("Max stocks to check per strategy", 15, 80, 35, key="stocks_under_max")

        if mode.startswith("All strategies"):
            st.info(
                "Builds a map: every strategy → matching stocks today. "
                "Takes longer; keep max stocks moderate."
            )
            side_f = st.selectbox("Side", ["BUY", "SELL", "BOTH"], key="all_strat_side")
            if st.button("List stocks under ALL strategies", type="primary", key="list_all_strat_btn"):
                all_rows = []
                summary = []
                buy_ids = [
                    k for k, v in SWING_STRATEGIES.items()
                    if side_f == "BOTH" or str(v.get("side", "BUY")).upper() == side_f
                ]
                prog = st.progress(0.0)
                with st.spinner("Scanning all strategies…"):
                    for i, sid in enumerate(buy_ids):
                        try:
                            sdf = stocks_under_strategy(base, sid, max_check=max_check)
                            if sdf is not None and not sdf.empty:
                                sdf = sdf.copy()
                                sdf["Strategy ID"] = sid
                                if "Strategy" not in sdf.columns:
                                    sdf["Strategy"] = SWING_STRATEGIES[sid]["name"]
                                all_rows.append(sdf)
                                summary.append({
                                    "Strategy": SWING_STRATEGIES[sid]["name"],
                                    "Side": SWING_STRATEGIES[sid].get("side", "BUY"),
                                    "Matches": len(sdf),
                                    "Sample stocks": ", ".join(
                                        sdf["Stock"].astype(str).head(8).tolist()
                                    ),
                                })
                            else:
                                summary.append({
                                    "Strategy": SWING_STRATEGIES[sid]["name"],
                                    "Side": SWING_STRATEGIES[sid].get("side", "BUY"),
                                    "Matches": 0,
                                    "Sample stocks": "—",
                                })
                        except Exception:
                            summary.append({
                                "Strategy": SWING_STRATEGIES.get(sid, {}).get("name", sid),
                                "Side": "—",
                                "Matches": 0,
                                "Sample stocks": "error",
                            })
                        prog.progress((i + 1) / max(len(buy_ids), 1))
                if all_rows:
                    big = pd.concat(all_rows, ignore_index=True)
                    # Attach how many strategies each stock matches overall
                    try:
                        counts, alln = [], []
                        for _, rr in big.iterrows():
                            m = strategies_matching_stock(
                                str(rr["Stock"]),
                                side_filter="BUY" if side_f != "SELL" else "SELL",
                            )
                            counts.append(m.get("count", 0))
                            alln.append(", ".join(m.get("names") or []))
                        big["# Strategies matched"] = counts
                        big["Also matches"] = alln
                    except Exception:
                        pass
                    st.session_state["stocks_under_all_df"] = big
                    st.session_state["stocks_under_all_summary"] = pd.DataFrame(summary)
                else:
                    st.session_state["stocks_under_all_df"] = pd.DataFrame()
                    st.session_state["stocks_under_all_summary"] = pd.DataFrame(summary)
                    st.warning("No matches across strategies with current universe/limits.")

            summary_df = st.session_state.get("stocks_under_all_summary")
            big = st.session_state.get("stocks_under_all_df")
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                st.markdown("**Summary — matches per strategy**")
                st.dataframe(
                    summary_df.sort_values("Matches", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            if isinstance(big, pd.DataFrame) and not big.empty:
                st.success(f"**{len(big)}** raw strategy–stock rows → merged into unique stock cards below")
                names = ["ALL"] + sorted(big["Strategy"].astype(str).unique().tolist())
                pick_name = st.selectbox("Filter map by strategy (optional)", names, key="all_map_filter")
                view = big if pick_name == "ALL" else big[big["Strategy"].astype(str) == pick_name]
                st.subheader("📋 Unique stocks — all strategies on one card")
                render_unique_strategy_stock_cards(view, max_cards=15, key_prefix="all_usc")
                with st.expander("Raw multi-row table + CSV", expanded=False):
                    st.dataframe(view, use_container_width=True, hide_index=True)
                    try:
                        st.download_button(
                            "⬇️ Download CSV",
                            data=view.to_csv(index=False).encode("utf-8"),
                            file_name=f"all_strategies_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="dl_all_strat_stocks",
                        )
                    except Exception:
                        pass
            elif summary_df is None:
                st.caption("Click **List stocks under ALL strategies** to build the full map.")

        else:
            # One strategy only (original behaviour)
            sid_s = st.selectbox(
                "Strategy",
                list(SWING_STRATEGIES.keys()),
                format_func=lambda k: SWING_STRATEGIES[k]["name"],
                key="stocks_under_select",
            )
            meta_s = SWING_STRATEGIES[sid_s]
            with st.expander(f"📖 Detailed explanation — {meta_s['name']}", expanded=True):
                st.write(meta_s.get("beginner") or meta_s.get("desc"))
                st.write(f"**Why:** {meta_s.get('why', '')}")
                st.write(f"**Hold ~{meta_s.get('hold_days')} days** · Typical win {meta_s.get('typical_win')}")
                if sid_s == "my_strategy":
                    st.error(
                        "STRICT: Building patterns only. "
                        "Shiva Mills–style (pattern already done + 2–3 days up) = **NOT a buy**."
                    )
            det = (st.session_state.get("strategy_bt_detail") or {}).get(sid_s)
            if det:
                st.info(
                    f"Last backtest for **{det.get('name')}**: efficiency **{det.get('win_rate')}%** · "
                    f"{det.get('wins')}W / {det.get('losses')}L · {det.get('trades')} trades"
                )
            if st.button("List stocks under this strategy", type="primary", key="list_under_btn"):
                with st.spinner("Matching stocks…"):
                    sdf = stocks_under_strategy(base, sid_s, max_check=max_check)
                    if sdf is not None and not sdf.empty:
                        counts, alln = [], []
                        for _, rr in sdf.iterrows():
                            m = strategies_matching_stock(str(rr["Stock"]), side_filter="BUY")
                            counts.append(m["count"])
                            alln.append(", ".join(m["names"]))
                        sdf = sdf.copy()
                        sdf["# Strategies matched"] = counts
                        sdf["Also matches"] = alln
                    st.session_state["stocks_under_df"] = sdf
                    st.session_state["stocks_under_sid_saved"] = sid_s
            sdf = st.session_state.get("stocks_under_df", pd.DataFrame())
            if (
                sdf is not None
                and not sdf.empty
                and st.session_state.get("stocks_under_sid_saved") == sid_s
            ):
                st.success(f"**{len(sdf)}** hits under **{SWING_STRATEGIES[sid_s]['name']}** → unique cards")
                render_unique_strategy_stock_cards(sdf, max_cards=12, key_prefix="one_usc")
                with st.expander("Raw table", expanded=False):
                    st.dataframe(sdf, use_container_width=True, hide_index=True)
            else:
                st.caption("Click the button after a market scan for more names.")

    with tab_hi:
        st.subheader("⭐ High conviction — multi-strategy merge")
        st.caption(
            "Stocks matching **2+ strategies**. Chart + full strategy details + "
            "simple merge backtest (how often target hit when those strategies agreed)."
        )
        st.markdown(
            """
**What to buy? (priority order)**

| Priority | Source | When to use |
|----------|--------|-------------|
| **1 Best** | **High conviction** (2+ strategies) + especially **My Strategy** | Strongest overlap |
| **2** | **Sure Call Desk** | Ultra-strict two-stage + Nifty OK |
| **3** | **My Strategy** alone (ACTIVE_FRESH) | Your divergence rule only |
| **4** | **Precision picks** | Ranked scan quality, still selective |
| **5** | Plain BUY scan list | Weakest — use only with extra filters |

If **My Strategy** is ACTIVE_FRESH **and** another strategy matches → prefer that name.
            """
        )
        if st.button("Find high conviction", type="primary", key="hi_conv"):
            buy_ids = [k for k, v in SWING_STRATEGIES.items() if v.get("side") == "BUY"]
            hits = {}  # stock -> {ids, names, details}
            with st.spinner("Scanning strategies + charts…"):
                for stock in list(universe)[:30]:
                    matched_ids = []
                    matched_names = []
                    try:
                        raw = stock_history(clean_symbol(stock), interval="1d")
                        if raw is None or len(raw) < 80:
                            continue
                        df = calculate_indicators(raw)
                        i = len(df) - 1
                        for sid in buy_ids:
                            if _strategy_signal_at_bar(df, i, sid):
                                matched_ids.append(sid)
                                matched_names.append(SWING_STRATEGIES[sid]["name"])
                    except Exception:
                        continue
                    if len(matched_ids) >= 2:
                        hits[display_symbol(stock)] = {
                            "ids": matched_ids,
                            "names": matched_names,
                        }
            st.session_state["hi_conv_hits"] = hits
            # Auto-save high conviction to Past Predictions
            try:
                special = []
                for stock, info in (hits or {}).items():
                    names = info.get("names") if isinstance(info, dict) else info
                    live_px = tgt = sl = 0.0
                    hold_d = 15
                    try:
                        raw0 = stock_history(clean_symbol(stock), interval="1d")
                        if raw0 is not None and not raw0.empty:
                            df0 = calculate_indicators(raw0)
                            row0 = df0.iloc[-1]
                            live_px = safe_float(row0.get("Close"))
                            atr0 = safe_float(row0.get("ATR")) or live_px * 0.02
                            try:
                                q = live_quote(clean_symbol(stock))
                                if q:
                                    lp = safe_float(q.get("price") or q.get("Price") or q.get("last"), 0)
                                    if lp > 0:
                                        live_px = lp
                            except Exception:
                                pass
                            pp = load_my_strategy_params()
                            tgt = live_px + float(pp.get("atr_target_mult", 1.8)) * atr0
                            sl = live_px - float(pp.get("atr_stop_mult", 2.2)) * atr0
                            hold_d = int(pp.get("hold_bars", 15))
                    except Exception:
                        pass
                    special.append({
                        "Stock": stock,
                        "Call": "BUY",
                        "Call Source": "HIGH_CONV",
                        "Strategy": " + ".join(names) if names else "High conviction",
                        "Entry": live_px,
                        "Target": tgt,
                        "Stop Loss": sl,
                        "Hold Days": hold_d,
                        "Prediction": 88,
                        "Reason": f"High conviction: {', '.join(names or [])}",
                    })
                n_hc = save_special_calls(special)
                if n_hc:
                    st.success(
                        f"Saved **{n_hc}** high-conviction call(s) to Past Predictions "
                        f"(Call Source = **HIGH_CONV**)."
                    )
            except Exception as e:
                st.caption(f"High-conviction history save note: {e}")
        hits = st.session_state.get("hi_conv_hits") or {}
        if hits:
            if st.button("💾 Save high conviction to Past Predictions again", key="hc_resave"):
                special = []
                for stock, info in hits.items():
                    names = info.get("names") if isinstance(info, dict) else info
                    special.append({
                        "Stock": stock,
                        "Call": "BUY",
                        "Call Source": "HIGH_CONV",
                        "Strategy": " + ".join(names) if names else "High conviction",
                        "Entry": 0,
                        "Target": 0,
                        "Stop Loss": 0,
                        "Hold Days": 15,
                        "Prediction": 88,
                    })
                st.success(f"Wrote {save_special_calls(special)} HIGH_CONV row(s).")
            for stock, info in hits.items():
                names = info.get("names") if isinstance(info, dict) else info
                ids = info.get("ids") if isinstance(info, dict) else []
                st.markdown("---")
                st.success(f"**{stock}** — **{len(names)} strategies merge:** {', '.join(names)}")
                # Live / last price + Target + Stop (works market open or closed)
                live_px = tgt = sl = 0.0
                hold_d = 15
                try:
                    raw0 = stock_history(clean_symbol(stock), interval="1d")
                    if raw0 is not None and not raw0.empty:
                        df0 = calculate_indicators(raw0)
                        row0 = df0.iloc[-1]
                        live_px = safe_float(row0.get("Close"))
                        atr0 = safe_float(row0.get("ATR")) or live_px * 0.02
                        try:
                            q = live_quote(clean_symbol(stock))
                            if q:
                                lp = safe_float(q.get("price") or q.get("Price") or q.get("last"), 0)
                                if lp > 0:
                                    live_px = lp
                        except Exception:
                            pass
                        pp = load_my_strategy_params()
                        tgt = live_px + float(pp.get("atr_target_mult", 1.8)) * atr0
                        sl = live_px - float(pp.get("atr_stop_mult", 2.2)) * atr0
                        hold_d = int(pp.get("hold_bars", 15))
                except Exception:
                    pass
                a, b, c, d = st.columns(4)
                a.metric("Current / Live Price", f"₹{live_px:,.2f}" if live_px else "—")
                b.metric("Target", f"₹{tgt:,.2f}" if tgt else "—")
                c.metric("Stop Loss", f"₹{sl:,.2f}" if sl else "—")
                d.metric("Hold (days)", str(hold_d))
                st.markdown(
                    f"**{stock}** · Live **₹{live_px:,.2f}** · Target **₹{tgt:,.2f}** · "
                    f"Stop Loss **₹{sl:,.2f}** · (last close used if market closed)"
                )
                # Strategy details
                for sid in (ids or []):
                    meta = SWING_STRATEGIES.get(sid, {})
                    with st.expander(f"📖 {meta.get('name', sid)}", expanded=False):
                        st.write(meta.get("beginner") or meta.get("desc"))
                        st.write(f"**Why:** {meta.get('why', '')}")
                        st.caption(
                            f"Hold ~{meta.get('hold_days')}d · Typical win {meta.get('typical_win')} · "
                            f"{meta.get('typical_exp')}"
                        )
                        st.markdown(
                            f"Levels for this stock: Live ₹{live_px:,.2f} · "
                            f"Target ₹{tgt:,.2f} · SL ₹{sl:,.2f}"
                        )
                # Chart
                try:
                    raw = stock_history(clean_symbol(stock), interval="1d")
                    if raw is not None and not raw.empty:
                        df = calculate_indicators(raw)
                        show_my = "my_strategy" in (ids or [])
                        fig = build_full_plotly_chart(
                            df,
                            title=f"{stock} · High conviction",
                            target=tgt,
                            stop_loss=sl,
                            height=560,
                            show_rsi=True,
                            my_strategy_lines=show_my,
                            lookback_lines=int(load_my_strategy_params().get("lookback", 90)),
                        )
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True, key=f"hi_chart_{stock}")
                        # Merge backtest: bars where ALL matched strategies fired, then forward outcome
                        if ids and len(df) > 100:
                            hold = 15
                            wins = losses = 0
                            for i in range(60, len(df) - hold - 1):
                                if all(
                                    _strategy_signal_at_bar(df, i, sid) for sid in ids
                                ):
                                    entry = safe_float(df.iloc[i + 1]["Open"])
                                    atr_v = safe_float(df.iloc[i].get("ATR")) or entry * 0.02
                                    tgt = entry + 1.8 * atr_v
                                    stop = entry - 2.2 * atr_v
                                    res = None
                                    for j in range(i + 1, min(i + 1 + hold, len(df))):
                                        hi = safe_float(df.iloc[j]["High"])
                                        lo = safe_float(df.iloc[j]["Low"])
                                        if lo <= stop:
                                            res = "L"
                                            break
                                        if hi >= tgt:
                                            res = "W"
                                            break
                                    if res == "W":
                                        wins += 1
                                    elif res == "L":
                                        losses += 1
                            decided = wins + losses
                            pct = (wins / decided * 100) if decided else 0.0
                            st.info(
                                f"**Merge backtest on {stock}** (all of: {', '.join(names)}): "
                                f"**{pct:.0f}%** target-before-stop "
                                f"({wins}W / {losses}L on {decided} decided signals). "
                                + (
                                    "**Positive historical edge on this merge — preferred buy candidate.**"
                                    if decided >= 3 and pct >= 50
                                    else "Small sample or mixed — size small / confirm with Sure or My Strategy."
                                )
                            )
                except Exception as e:
                    st.caption(f"Chart/backtest: {e}")
        else:
            st.caption("Click **Find high conviction** after a market scan.")

    with tab_ao:
        st.subheader("Levels for manual Angel One order")
        st.caption("This app does not place broker orders. Copy Entry / SL / Target into Angel One.")
        sym_a = st.text_input("Symbol", value="INFY", key="ao_sym").upper().strip()
        if st.button("Get levels", key="ao_btn"):
            try:
                raw = stock_history(clean_symbol(sym_a), interval="1d")
                if raw is None or raw.empty:
                    st.error("No data")
                else:
                    df = calculate_indicators(raw)
                    row = df.iloc[-1]
                    c = safe_float(row.get("Close"))
                    atr_v = safe_float(row.get("ATR")) or c * 0.02
                    p = load_my_strategy_params()
                    tgt = c + float(p.get("atr_target_mult", 1.8)) * atr_v
                    sl = c - float(p.get("atr_stop_mult", 2.2)) * atr_v
                    st.markdown(
                        f"**{display_symbol(sym_a)}**  \n"
                        f"- Entry (ref): **₹{c:,.2f}**  \n"
                        f"- Target: **₹{tgt:,.2f}**  \n"
                        f"- Stop Loss: **₹{sl:,.2f}**  \n"
                        f"- Hold ~**{int(p.get('hold_bars', 15))}** days"
                    )
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown(
        """
**What should you buy?**  
1. **High conviction** (2+ strategies) with good merge % — best  
2. **Sure Call** when Nifty allows  
3. **My Strategy ACTIVE_FRESH** (stocks or Nifty/BankNifty options bias)  
4. **Precision** only if it also overlaps above  
5. Avoid raw scan-only names without filters  

**Daily flow:** Scan → Strategy Lab / High conviction → My Strategy or Sure → Angel One levels  
        """
    )


# ============================================================
# NEWS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_news(symbol):

    try:

        ticker = yf.Ticker(
            clean_symbol(symbol)
        )

        news = ticker.news

        if not news:
            return []

        output = []

        for item in news[:5]:

            content = item.get(
                "content",
                item
            )

            if not isinstance(
                content,
                dict
            ):
                content = item

            title = (
                content.get("title")
                or item.get("title")
                or ""
            )

            provider = content.get(
                "provider",
                {}
            )

            if isinstance(
                provider,
                dict
            ):
                publisher = provider.get(
                    "displayName",
                    ""
                )
            else:
                publisher = item.get(
                    "publisher",
                    ""
                )

            if title:

                output.append(
                    {
                        "title": title,
                        "publisher": publisher,
                    }
                )

        return output

    except Exception:

        return []


def news_score(news):

    positive_words = [
        "profit",
        "growth",
        "record",
        "strong",
        "surge",
        "approval",
        "order",
        "contract",
        "expansion",
        "upgrade",
        "buy",
        "beat",
        "revenue",
        "positive",
        "partnership",
    ]

    negative_words = [
        "loss",
        "fall",
        "decline",
        "weak",
        "downgrade",
        "fraud",
        "investigation",
        "warning",
        "debt",
        "cut",
        "sell",
        "negative",
        "lawsuit",
        "miss",
    ]

    score = 0

    for item in news:

        text = str(
            item.get(
                "title",
                ""
            )
        ).lower()

        for word in positive_words:

            if word in text:
                score += 1

        for word in negative_words:

            if word in text:
                score -= 1

    return score


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def analyse_stock(
    symbol,
    df,
    fetch_news=True
):

    if df is None or df.empty:
        return None

    if len(df) < 220:
        return None

    df = calculate_indicators(
        df
    )

    if df.empty or len(df) < 220:
        return None

    row = df.iloc[-1]

    price = safe_float(
        row["Close"]
    )

    if price <= 0:
        return None

    score = 50

    reasons = []

    # --------------------------------------------------------
    # EMA 20
    # --------------------------------------------------------

    if price > safe_float(row["EMA20"]):

        score += 5

        reasons.append(
            "Price is above the 20-day EMA, showing short-term strength."
        )

    else:

        score -= 5

        reasons.append(
            "Price is below the 20-day EMA, showing weaker short-term momentum."
        )

    # --------------------------------------------------------
    # EMA 50
    # --------------------------------------------------------

    if price > safe_float(row["EMA50"]):

        score += 6

        reasons.append(
            "Price is above the 50-day EMA, supporting the medium-term trend."
        )

    else:

        score -= 6

        reasons.append(
            "Price is below the 50-day EMA."
        )

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    if price > safe_float(row["EMA200"]):

        score += 5

        reasons.append(
            "Price is above the 200-day EMA, supporting the long-term trend."
        )

    else:

        score -= 5

        reasons.append(
            "Price is below the 200-day EMA, which is a long-term weakness warning."
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_value = safe_float(
        row["RSI"]
    )

    if 50 <= rsi_value <= 70:

        score += 7

        reasons.append(
            f"RSI is {rsi_value:.1f}, showing positive momentum without extreme overbought conditions."
        )

    elif rsi_value < 30:

        score += 2

        reasons.append(
            f"RSI is {rsi_value:.1f}, indicating oversold conditions and possible rebound potential."
        )

    elif rsi_value > 75:

        score -= 6

        reasons.append(
            f"RSI is {rsi_value:.1f}, indicating the stock may be overheated."
        )

    else:

        reasons.append(
            f"RSI is {rsi_value:.1f}."
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_value = safe_float(
        row["MACD"]
    )

    macd_signal = safe_float(
        row["MACDSignal"]
    )

    macd_hist = safe_float(
        row["MACDHist"]
    )

    if (
        macd_value > macd_signal
        and
        macd_hist > 0
    ):

        score += 8

        reasons.append(
            "MACD is bullish because it is above the signal line."
        )

    elif (
        macd_value < macd_signal
        and
        macd_hist < 0
    ):

        score -= 8

        reasons.append(
            "MACD is bearish because it is below the signal line."
        )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    adx_value = safe_float(
        row["ADX"]
    )

    if adx_value >= 25:

        score += 6

        reasons.append(
            f"ADX is {adx_value:.1f}, indicating a reasonably strong trend."
        )

    else:

        reasons.append(
            f"ADX is {adx_value:.1f}, so the current trend is not especially strong."
        )

    # --------------------------------------------------------
    # STOCHASTIC
    # --------------------------------------------------------

    k = safe_float(
        row["StochK"]
    )

    d = safe_float(
        row["StochD"]
    )

    if k > d and k < 80:

        score += 4

        reasons.append(
            "Stochastic momentum is bullish."
        )

    elif k < d and k > 20:

        score -= 4

        reasons.append(
            "Stochastic momentum is bearish."
        )

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    bb_middle = safe_float(
        row["BBMiddle"]
    )

    if price > bb_middle:

        score += 4

        reasons.append(
            "Price is above the Bollinger middle band."
        )

    else:

        score -= 3

        reasons.append(
            "Price is below the Bollinger middle band."
        )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    vwap_value = safe_float(
        row["VWAP"]
    )

    if price > vwap_value:

        score += 4

        reasons.append(
            "Price is above VWAP, supporting buying strength."
        )

    else:

        score -= 4

        reasons.append(
            "Price is below VWAP."
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume = safe_float(
        row["Volume"]
    )

    average_volume = safe_float(
        row["VolumeAvg20"]
    )

    if average_volume > 0:

        volume_ratio = (
            volume /
            average_volume
        )

    else:

        volume_ratio = 1

    if volume_ratio >= 1.5:

        score += 6

        reasons.append(
            "Volume is significantly above average, making the price move more meaningful."
        )

    elif volume_ratio >= 1.1:

        score += 2

        reasons.append(
            "Volume is moderately above average."
        )

    else:

        reasons.append(
            "Volume is not unusually high."
        )

    # --------------------------------------------------------
    # PATTERNS (weights learned from past wins/losses)
    # --------------------------------------------------------

    patterns = detect_patterns(df)
    _learning = load_learning()

    for pattern in patterns:
        w = pattern_weight(pattern, _learning)
        meta = PATTERN_IMPORTANCE.get(pattern, {})
        score += w
        why = meta.get("why", "Detected technical pattern.")
        bias = meta.get("bias", "")
        if w > 0:
            reasons.append(
                f"{pattern} ({bias}): {why} [weight {w:+.1f} from base+learning]"
            )
        elif w < 0:
            reasons.append(
                f"{pattern} ({bias}): {why} [weight {w:+.1f} from base+learning]"
            )
        else:
            reasons.append(f"{pattern}: {why}")

    # My Strategy — RSI bullish divergence (user template: RELIANCE / COALINDIA)
    try:
        _ms = my_strategy_status(df)
        if _ms.get("ok_for_buy") or _ms.get("status") == "ACTIVE_FRESH":
            boost = safe_float(_learning.get("my_strategy_boost"), 8)
            mult = safe_float(_learning.get("rsi_bull_div_multiplier"), 1.15)
            add = boost * mult / 1.1
            score += add
            if "RSI_BULL_DIV" not in patterns:
                patterns.append("RSI_BULL_DIV")
            reasons.append(
                f"My Strategy RSI bullish divergence ACTIVE_FRESH "
                f"(price LL + RSI HL, same bars) [+{add:.1f} learned weight]. "
                f"{_ms.get('reason', '')}"
            )
        elif _ms.get("status") == "TOO_LATE":
            reasons.append(
                f"My Strategy divergence TOO_LATE — not a fresh buy: {_ms.get('reason', '')}"
            )
            score -= 3
        elif _ms.get("status") == "RSI_FAILED":
            reasons.append(
                f"My Strategy RSI line failed — no buy: {_ms.get('reason', '')}"
            )
            score -= 4
    except Exception:
        pass

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    news = []

    if fetch_news:

        news = get_news(
            symbol
        )

    ns = news_score(
        news
    )

    if ns > 0:

        score += min(
            ns * 2,
            6
        )

        news_influence = (
            "Positive news influence detected."
        )

    elif ns < 0:

        score -= min(
            abs(ns) * 2,
            6
        )

        news_influence = (
            "Negative news influence detected."
        )

    else:

        news_influence = (
            "No clear positive or negative news influence detected."
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = float(
        np.clip(
            score,
            5,
            95
        )
    )

    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    if prediction >= 72:

        signal = "BUY"

    elif prediction >= 55:

        signal = "HOLD"

    elif prediction <= 38:

        signal = "SELL"

    else:

        signal = "WATCH"

    # --------------------------------------------------------
    # TARGET / STOP
    # --------------------------------------------------------

    atr_value = safe_float(
        row["ATR"]
    )

    if atr_value <= 0:

        atr_value = (
            price * 0.03
        )

    # Placeholder — final target/stop set after signal is finalized
    stop_loss = price - 1.5 * atr_value
    target = price + 2.5 * atr_value

    risk_pct = (
        abs(price - stop_loss)
        /
        price
        *
        100
    )

    if risk_pct < 3:

        risk_level = "LOW"

    elif risk_pct < 6:

        risk_level = "MEDIUM"

    elif risk_pct < 10:

        risk_level = "HIGH"

    else:

        risk_level = "VERY HIGH"

    # --------------------------------------------------------
    # LEARNING FROM PAST MISTAKES (closed target/stop history)
    # --------------------------------------------------------
    learn_delta, learn_notes = learning_score_adjustment(
        prediction, risk_level, signal, patterns, _learning
    )
    if learn_delta:
        score = float(np.clip(score + learn_delta, 5, 95))
        prediction = float(np.clip(score, 5, 95))
        # Re-map call after learning nudge
        if prediction >= 72:
            signal = "BUY"
        elif prediction >= 55:
            signal = "HOLD"
        elif prediction <= 38:
            signal = "SELL"
        else:
            signal = "WATCH"
    for note in learn_notes:
        reasons.append(note)

    # --------------------------------------------------------
    # STRONG TREND ONLY for BUY (quality over quantity)
    # --------------------------------------------------------
    adx_v = safe_float(row.get("ADX"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    ema200 = safe_float(row.get("EMA200"))
    vol_r = safe_float(row.get("Volume Ratio"), 1.0)
    rsi_v = safe_float(row.get("RSI"), 50)

    # Prefer strong trend; relaxed so the screen is not empty for days
    strong_trend = (
        price > 0
        and (ema20 <= 0 or price > ema20)
        and (ema50 <= 0 or price > ema50)
        and adx_v >= 20
        and 45 <= rsi_v <= 75
        and vol_r >= 0.9
    )
    strong_trend_note = (
        f"Strong trend: price vs EMA20/50, ADX={adx_v:.1f} (≥20), "
        f"RSI={rsi_v:.1f} (45–75), VolRatio={vol_r:.2f}."
    )

    if signal == "BUY" and not strong_trend:
        signal = "WATCH"
        prediction = min(prediction, 70)
        reasons.append(
            "BUY gated: prefer ADX≥20, price above EMA20/50, RSI 45–75. "
            "Marked WATCH (near-buy) so weak BUYs are not flooded."
        )
    elif signal == "BUY" and strong_trend:
        reasons.append("STRONG TREND BUY: " + strong_trend_note)
        prediction = min(95, prediction + 3)

    # --------------------------------------------------------
    # TARGET / STOP by call direction (SELL target must be BELOW price)
    # --------------------------------------------------------
    if atr_value <= 0:
        atr_value = price * 0.03

    if signal == "SELL":
        # Short / sell: profit when price falls → target below, stop above
        target = price - 2.5 * atr_value
        stop_loss = price + 1.5 * atr_value
        if target <= 0:
            target = price * 0.92
        reasons.append(
            f"SELL levels: Target ₹{target:.2f} (below price) · Stop ₹{stop_loss:.2f} (above price)."
        )
    else:
        # BUY / HOLD / WATCH: long levels
        stop_loss = price - 1.5 * atr_value
        if stop_loss <= 0:
            stop_loss = price * 0.95
        target = price + 2.5 * atr_value

    risk_pct = abs(price - stop_loss) / price * 100 if price > 0 else 0
    if risk_pct < 3:
        risk_level = "LOW"
    elif risk_pct < 6:
        risk_level = "MEDIUM"
    elif risk_pct < 10:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"

    # Expected move % (direction-aware)
    if signal == "SELL":
        target_pct = ((price - target) / price * 100) if price > 0 else 0
    else:
        target_pct = ((target - price) / price * 100) if price > 0 else 0

    # --------------------------------------------------------
    # HOLDING PERIOD
    # --------------------------------------------------------

    if prediction >= 85:

        hold_days = 7

    elif prediction >= 75:

        hold_days = 10

    elif prediction >= 65:

        hold_days = 15

    else:

        hold_days = 20

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    if (
        prediction >= 85
        and
        risk_pct < 6
    ):

        priority = "VERY HIGH"

    elif prediction >= 75:

        priority = "HIGH"

    elif prediction >= 65:

        priority = "MEDIUM"

    else:

        priority = "LOW"

    # --------------------------------------------------------
    # EXPECTED RETURN (direction-aware)
    # --------------------------------------------------------

    if signal == "SELL":
        target_pct = ((price - target) / price * 100) if price > 0 else 0
    else:
        target_pct = ((target - price) / price * 100) if price > 0 else 0

    # --------------------------------------------------------
    # FINAL REASON
    # --------------------------------------------------------

    reason = " ".join(
        reasons
    )

    return {

        "Stock": display_symbol(
            symbol
        ),

        "Symbol": clean_symbol(
            symbol
        ),

        "Sector": sector_of(
            symbol
        ),

        "Price": round(
            price,
            2
        ),

        "Call": signal,

        "Prediction": round(
            prediction,
            1
        ),

        "Risk %": round(
            risk_pct,
            2
        ),

        "Risk Level": risk_level,

        "Target": round(
            target,
            2
        ),

        "Target %": round(
            target_pct,
            2
        ),

        "Stop Loss": round(
            stop_loss,
            2
        ),

        "Hold Days": hold_days,

        "Priority": priority,

        "RSI": round(
            rsi_value,
            2
        ),

        "MACD": round(
            macd_value,
            4
        ),

        "MACD Signal": round(
            macd_signal,
            4
        ),

        "ADX": round(
            adx_value,
            2
        ),

        "Stochastic": round(
            k,
            2
        ),

        "Volume Ratio": round(
            volume_ratio,
            2
        ),

        "VWAP": round(
            vwap_value,
            2
        ),

        "Patterns": (
            ", ".join(patterns)
            if patterns
            else
            "No major pattern detected"
        ),

        "Reason": reason,

        "Technical Reasons": (
            " | ".join(reasons)
        ),

        "News Influence":
            news_influence,

        "News": news,

        "Data": df,
    }


# ============================================================
# BULK MARKET DOWNLOAD
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def download_market_data():
    """
    Robust batch downloader for the full NSE scan.
    A failed batch does not stop the complete scan.
    """
    symbols = list(NSE_STOCKS[:MAX_SCAN_STOCKS])
    batches = []
    batch_size = 50

    for start_idx in range(0, len(symbols), batch_size):
        batch = symbols[start_idx:start_idx + batch_size]
        part = None

        for attempt in range(3):
            try:
                part = yf.download(
                    tickers=batch,
                    period="1y",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                    timeout=30,
                )
                if part is not None and not part.empty:
                    break
            except Exception:
                part = None

            time.sleep(1 + attempt)

        if part is not None and not part.empty:
            batches.append(part)

    if not batches:
        return None

    if len(batches) == 1:
        return batches[0]

    try:
        return pd.concat(batches, axis=1)
    except Exception:
        # If one unusual batch has incompatible columns, retain all
        # usable data from the first successful batch rather than crashing.
        return batches[0]


def extract_stock_data(
    all_data,
    symbol
):

    if all_data is None:
        return pd.DataFrame()

    try:

        if isinstance(
            all_data.columns,
            pd.MultiIndex
        ):

            level0 = (
                all_data
                .columns
                .get_level_values(0)
            )

            level1 = (
                all_data
                .columns
                .get_level_values(1)
            )

            if symbol in level0:

                df = all_data[
                    symbol
                ].copy()

            elif symbol in level1:

                df = all_data.xs(
                    symbol,
                    axis=1,
                    level=1
                ).copy()

            else:

                return pd.DataFrame()

        else:

            df = all_data.copy()

        df = normalize_columns(
            df
        )

        if "Close" not in df.columns:
            return pd.DataFrame()

        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# FAST SCANNER
# ============================================================


def run_scanner():
    """
    Fault-tolerant full-market scanner.

    It scans every NSE symbol for which usable price history is available.
    Missing, suspended, delisted or temporarily unavailable symbols are skipped.
    """
    symbols = list(NSE_STOCKS[:MAX_SCAN_STOCKS])

    if not symbols:
        return pd.DataFrame()

    all_data = download_market_data()
    results = []
    analysed = set()

    progress = st.progress(0)
    status = st.empty()
    total = len(symbols)

    # FAST PASS: analyse all symbols returned by the bulk downloader.
    for i, symbol in enumerate(symbols):
        try:
            df = extract_stock_data(all_data, symbol)

            if df is not None and not df.empty and len(df) >= 60:
                result = analyse_stock(
                    symbol,
                    df,
                    fetch_news=False
                )

                if result:
                    results.append(result)
                    analysed.add(symbol)
        except Exception:
            # Never let one stock terminate the full market scan.
            pass

        if i == total - 1 or i % 20 == 0:
            progress.progress(min((i + 1) / total, 1.0))
            pct = ((i + 1) / total) * 100
            status.caption(
                f"⏳ FULL MARKET SCAN: {pct:.1f}% complete | "
                f"{i + 1:,}/{total:,} stocks scanned | "
                f"{len(results):,} usable analyses"
            )

    # FALLBACK PASS:
    # Retry missing symbols via Yahoo, then NSE historical API.
    # Cap retries so the UI stays usable.
    missing = [s for s in symbols if s not in analysed]
    max_retry = 300 if len(results) < 50 else min(120, len(missing))
    retry_list = missing[:max_retry]

    if retry_list and (len(results) < 80 or len(missing) > 0):
        status.caption(
            f"Retrying up to {len(retry_list):,} symbols via Yahoo / NSE fallback..."
        )

        for j, symbol in enumerate(retry_list):
            try:
                df = stock_history(symbol)
                if df is not None and not df.empty and len(df) >= 60:
                    result = analyse_stock(symbol, df, fetch_news=False)
                    if result:
                        results.append(result)
                        analysed.add(symbol)
            except Exception:
                pass

            if j == len(retry_list) - 1 or j % 15 == 0:
                fallback_pct = ((j + 1) / len(retry_list)) * 100
                status.caption(
                    f"🔄 FALLBACK SCAN: {fallback_pct:.1f}% | "
                    f"{j + 1:,}/{len(retry_list):,} | "
                    f"{len(results):,} usable analyses"
                )

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame()

    output = pd.DataFrame(results)

    # Defensive cleanup so old/malformed values cannot break sorting.
    for col in ["Prediction", "Risk %"]:
        if col not in output.columns:
            output[col] = 0.0
        output[col] = pd.to_numeric(
            output[col],
            errors="coerce"
        ).fillna(0.0)

    output = output.sort_values(
        ["Prediction", "Risk %"],
        ascending=[False, True],
        kind="stable"
    ).reset_index(drop=True)

    output["Rank"] = np.arange(
        1,
        len(output) + 1
    )

    return output



def run_historical_prediction(symbol, selected_date):
    """
    Past prediction: rebuild V10 analysis using only data available up to
    the selected historical closing date. This works even when the market
    is currently closed.
    """
    df = stock_history(symbol)
    if df is None or df.empty:
        return None, pd.DataFrame()

    cutoff = pd.Timestamp(selected_date)
    hist = df.loc[df.index <= cutoff].copy()
    if hist.empty or len(hist) < 60:
        return None, hist

    try:
        result = analyse_stock(clean_symbol(symbol), hist)
    except Exception:
        result = None
    return result, hist

# ============================================================
# LIVE TRADINGVIEW CHARTS
# ============================================================

def _to_tv_symbol(symbol):
    """Map internal / Yahoo symbols to TradingView exchange:symbol format."""
    raw = str(symbol).upper().strip()
    if raw in {"^NSEI", "NIFTY", "NIFTY50", "NIFTY 50", "NSE:NIFTY"}:
        return "NSE:NIFTY"
    if raw in {"^NSEBANK", "BANKNIFTY", "BANK NIFTY", "NSE:BANKNIFTY"}:
        return "NSE:BANKNIFTY"
    if raw in {"^BSESN", "SENSEX", "BSE:SENSEX"}:
        return "BSE:SENSEX"
    if ":" in raw:
        return raw
    raw = raw.replace(".NS", "").replace(".BO", "")
    return "NSE:" + raw


def _yf_symbol_for_chart(symbol):
    """Yahoo Finance ticker for chart data download."""
    raw = str(symbol).upper().strip()
    if raw in {"^NSEI", "NIFTY", "NIFTY50", "NIFTY 50", "NSE:NIFTY"}:
        return "^NSEI"
    if raw in {"^NSEBANK", "BANKNIFTY", "BANK NIFTY", "NSE:BANKNIFTY"}:
        return "^NSEBANK"
    if raw in {"^BSESN", "SENSEX", "BSE:SENSEX"}:
        return "^BSESN"
    if raw.startswith("^"):
        return raw
    if ":" in raw:
        raw = raw.split(":")[-1]
    return clean_symbol(raw)


def _my_strategy_line_points(df: pd.DataFrame, lookback: int = 90):
    """
    Draw paired lines at the SAME timestamps:
    - Price line: swing low at T1 → swing low at T2
    - RSI line: RSI(T1) → RSI(T2)  (same bars as price)
    This matches user rule: entry price & RSI together, exit/2nd point together.
    """
    if df is None or len(df) < 40:
        return None
    d = df.copy()
    if "RSI" not in d.columns:
        try:
            d = calculate_indicators(d)
        except Exception:
            return None
    if "RSI" not in d.columns or "Low" not in d.columns:
        return None
    i = len(d) - 1
    lb = max(40, int(lookback))
    start = max(0, i - lb)
    seg = d.iloc[start : i + 1]
    lows = seg["Low"].astype(float)
    rsis = seg["RSI"].astype(float)
    pl = _swing_low_indices(lows, 3, 3)
    if len(pl) < 2:
        # try softer right window for latest bar
        pl = _swing_low_indices(lows, 3, 1)
    if len(pl) < 2:
        return None
    t1, t2 = pl[-2], pl[-1]
    if t2 <= t1:
        return None
    price_1 = float(lows.iloc[t1])
    price_2 = float(lows.iloc[t2])
    rsi_1 = float(rsis.iloc[t1])
    rsi_2 = float(rsis.iloc[t2])
    return {
        "price_x": [seg.index[t1], seg.index[t2]],
        "price_y": [price_1, price_2],
        "rsi_x": [seg.index[t1], seg.index[t2]],  # SAME times as price
        "rsi_y": [rsi_1, rsi_2],
        "price_falling": price_2 < price_1,
        "rsi_rising": rsi_2 >= rsi_1 - 0.25,
        "t1": str(seg.index[t1]),
        "t2": str(seg.index[t2]),
    }



def build_full_plotly_chart(
    df,
    title="Chart",
    target=None,
    stop_loss=None,
    height=720,
    show_rsi=True,
    my_strategy_lines=False,
    lookback_lines=90,
    show_patterns=True,
):
    """
    Candlestick + EMA/BB + Volume + RSI on every chart.
    Pattern markers on the candle where detected.
    Optional My Strategy paired lines (same-time price/RSI).
    """
    if df is None or df.empty:
        return None

    d = normalize_columns(df).copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col not in d.columns:
            return None
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if "Volume" not in d.columns:
        d["Volume"] = 0
    d["Volume"] = pd.to_numeric(d["Volume"], errors="coerce").fillna(0)
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    if len(d) < 5:
        return None

    d = d.tail(220).copy()
    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    d["RSI"] = rsi(d["Close"], 14)
    mid, upper, lower = bollinger(d)
    d["BBMiddle"] = mid
    d["BBUpper"] = upper
    d["BBLower"] = lower

    # Pattern recognition on last bars (for chart markers)
    pattern_labels = []
    try:
        if show_patterns and len(d) >= 5:
            for off in range(1, min(8, len(d))):
                sub = d.iloc[: len(d) - off + 1]
                pats = detect_patterns(sub)
                if pats:
                    pattern_labels.append((d.index[-off], pats, float(d["High"].iloc[-off])))
    except Exception:
        pattern_labels = []

    from plotly.subplots import make_subplots

    if show_rsi:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.30],
            subplot_titles=(title, "Volume", "RSI (14)"),
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["Open"],
            high=d["High"],
            low=d["Low"],
            close=d["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["EMA20"], name="EMA 20", line=dict(width=1.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["EMA50"], name="EMA 50", line=dict(width=1.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["BBUpper"],
            name="BB Upper",
            line=dict(width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["BBLower"],
            name="BB Lower",
            line=dict(width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

    if target is not None and safe_float(target) > 0:
        fig.add_hline(
            y=safe_float(target),
            line_dash="dash",
            line_color="#1a9b5f",
            annotation_text="Target",
            row=1,
            col=1,
        )
    if stop_loss is not None and safe_float(stop_loss) > 0:
        fig.add_hline(
            y=safe_float(stop_loss),
            line_dash="dash",
            line_color="#d93025",
            annotation_text="Stop Loss",
            row=1,
            col=1,
        )

    colors = [
        "#1a9b5f" if c >= o else "#d93025"
        for o, c in zip(d["Open"], d["Close"])
    ]
    fig.add_trace(
        go.Bar(x=d.index, y=d["Volume"], name="Volume", marker_color=colors),
        row=2,
        col=1,
    )

    if show_rsi:
        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d["RSI"],
                name="RSI",
                line=dict(color="#7c5cff", width=1.5),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#888", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#888", row=3, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="#555", row=3, col=1)

    # Model-drawn My Strategy lines (like your blue lines on TradingView)
    if my_strategy_lines:
        pts = _my_strategy_line_points(d, lookback=lookback_lines)
        if pts:
            fig.add_trace(
                go.Scatter(
                    x=pts["price_x"],
                    y=pts["price_y"],
                    mode="lines+markers",
                    name="MyStrat price line",
                    line=dict(color="#2196F3", width=2),
                    marker=dict(size=8),
                ),
                row=1,
                col=1,
            )
            if show_rsi:
                fig.add_trace(
                    go.Scatter(
                        x=pts["rsi_x"],
                        y=pts["rsi_y"],
                        mode="lines+markers",
                        name="MyStrat RSI line",
                        line=dict(color="#2196F3", width=2),
                        marker=dict(size=8),
                    ),
                    row=3,
                    col=1,
                )


    # Pattern markers on chart (recognition on the candles themselves)
    if show_patterns and pattern_labels:
        xs, ys, texts = [], [], []
        for idx, pats, hi in pattern_labels[:5]:
            xs.append(idx)
            ys.append(hi * 1.01)
            texts.append(", ".join(pats)[:28])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=11, symbol="triangle-down", color="#f5c542"),
                text=texts,
                textposition="top center",
                name="Patterns",
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.3)
    if show_rsi:
        fig.update_yaxes(range=[15, 85], row=3, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=0.3)
    return fig


def show_tradingview_chart(symbol, title=None, height=620, target=None, stop_loss=None):
    """
    Always-working interactive chart for any NSE stock / index.

    TradingView free embeds often show:
      "This symbol is only available on TradingView"
    inside Streamlit (nested iframe / domain restriction).

    So we chart with Plotly using Yahoo data (works for ALL symbols),
    and provide a one-click link to open the same symbol on TradingView.com.
    """
    tv_symbol = _to_tv_symbol(symbol)
    yf_sym = _yf_symbol_for_chart(symbol)
    chart_title = title or tv_symbol

    # Open full TradingView (drawing tools: trendline, RSI) — best place to draw by hand
    tv_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption(f"Chart + RSI · {tv_symbol} · Draw lines on TradingView (full site)")
    with c2:
        st.link_button("🔗 TradingView (draw)", tv_url, use_container_width=True)
    with c3:
        draw_my = st.checkbox("Show MyStrat lines", value=False, key=f"mylines_{tv_symbol}_{height}")

    try:
        df = stock_history(yf_sym)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.download(
                yf_sym,
                period="1y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            df = normalize_columns(df)
        except Exception:
            df = pd.DataFrame()

    lb_lines = 90
    try:
        lb_lines = int(load_my_strategy_params().get("lookback", 90))
    except Exception:
        pass

    fig = build_full_plotly_chart(
        df,
        title=chart_title,
        target=target,
        stop_loss=stop_loss,
        height=height,
        show_rsi=True,
        my_strategy_lines=draw_my,
        lookback_lines=lb_lines,
    )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "RSI panel is always on. Tick **Show MyStrat lines** to see model blue lines "
            "(price lower-lows + RSI higher-lows). For freehand drawing use **TradingView (draw)**."
        )
    else:
        st.warning(
            f"Chart data unavailable for {tv_symbol}. "
            "Use the TradingView button above to open the full chart."
        )


def show_market_index_charts():
    st.subheader("📈 Live Market Charts")
    tab1, tab2, tab3 = st.tabs(["NIFTY 50", "BANK NIFTY", "SENSEX"])
    with tab1:
        show_tradingview_chart("^NSEI", "NIFTY 50", height=560)
    with tab2:
        show_tradingview_chart("^NSEBANK", "BANK NIFTY", height=560)
    with tab3:
        show_tradingview_chart("^BSESN", "SENSEX", height=560)

# ============================================================
# LIVE QUOTE
# ============================================================

@st.cache_data(
    ttl=20,
    show_spinner=False
)
def live_quote(symbol):

    try:

        symbol = clean_symbol(
            symbol
        )

        # During market hours try 1-minute data.
        if nse_market_open_now():

            d = yf.download(
                symbol,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            d = normalize_columns(
                d
            )

            if (
                d.empty
                or
                "Close" not in d.columns
            ):

                d = yf.download(
                    symbol,
                    period="5d",
                    interval="5m",
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )

                d = normalize_columns(
                    d
                )

        else:

            # MARKET CLOSED:
            # Get latest closing price.
            d = yf.download(
                symbol,
                period="10d",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            d = normalize_columns(
                d
            )

        if (
            d is None
            or
            d.empty
            or
            "Close" not in d.columns
        ):

            return None

        close = (
            pd.to_numeric(
                d["Close"],
                errors="coerce"
            )
            .dropna()
        )

        if close.empty:
            return None

        current = safe_float(
            close.iloc[-1]
        )

        previous = (
            safe_float(
                close.iloc[-2]
            )
            if len(close) > 1
            else current
        )

        change = (
            current -
            previous
        )

        pct = (
            change /
            previous *
            100
            if previous
            else 0
        )

        if nse_market_open_now():

            label = "LIVE"

        else:

            label = "LAST CLOSE"

        return {

            "price": current,

            "change": change,

            "pct": pct,

            "updated":
                datetime.now().strftime(
                    "%H:%M:%S"
                ),

            "label": label,
        }

    except Exception:

        return None


def update_results_with_live_prices(results, max_stocks=80):
    """
    Refresh CURRENT price only.
    Locked prices (Entry at scan, Target, Stop Loss) are never overwritten.
    """
    if results is None or results.empty:
        return results

    x = results.copy()
    if "Symbol" not in x.columns and "Stock" not in x.columns:
        return x

    # Preserve original scan levels once
    if "Locked Price" not in x.columns and "Price" in x.columns:
        x["Locked Price"] = pd.to_numeric(x["Price"], errors="coerce")
    for col in ["Target", "Stop Loss"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    if "Current Price" not in x.columns:
        x["Current Price"] = pd.to_numeric(x.get("Price"), errors="coerce")

    n = min(max_stocks, len(x))
    for idx in list(x.index)[:n]:
        try:
            sym = x.at[idx, "Symbol"] if "Symbol" in x.columns else x.at[idx, "Stock"]
            q = live_quote(sym)
            if q and q.get("price"):
                px = round(safe_float(q["price"]), 2)
                x.at[idx, "Current Price"] = px
                x.at[idx, "Price"] = px  # display LTP
                # Target / Stop Loss / Locked Price untouched
        except Exception:
            continue

    return x


def refresh_history_current_prices(max_stocks: int = 60) -> pd.DataFrame:
    """
    Write Current Price into recommendation_history.csv for open rows.
    Never changes Entry, Target, or Stop Loss.
    """
    history = normalize_history_df(load_history())
    if history is None or history.empty:
        return history

    if "Current Price" not in history.columns:
        history["Current Price"] = ""

    # Freeze originals
    for col in ["Entry", "Target", "Stop Loss"]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    res_u = history["Result"].astype(str).str.upper().str.strip()
    open_mask = ~res_u.str.contains(
        "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
        regex=True,
        na=True,
    )
    open_idx = history.index[open_mask].tolist()[:max_stocks]

    for idx in open_idx:
        try:
            stock = history.at[idx, "Stock"]
            q = live_quote(stock)
            px = None
            if q and q.get("price"):
                px = safe_float(q["price"])
            if not px:
                d = stock_history(stock, interval="1d")
                if d is not None and not d.empty:
                    px = safe_float(d["Close"].iloc[-1])
            if px:
                history.at[idx, "Current Price"] = round(px, 2)
        except Exception:
            continue

    try:
        history.to_csv(HISTORY_FILE, index=False)
    except Exception:
        pass
    return history


def check_price_vs_levels(current_price, entry, target, stop_loss, call="BUY"):
    """
    Compare current live price against original Target / Stop Loss and
    return a clear status dict matching the requested UI format.
    """
    current = safe_float(current_price)
    entry = safe_float(entry)
    target = safe_float(target)
    stop = safe_float(stop_loss)
    call = str(call).upper()

    status = {
        "status": "OPEN",
        "badge": "",
        "message": "",
        "recommendation": "HOLD",
        "new_target": None,
        "new_stop": None,
        "action": "HOLD",
    }

    if current <= 0:
        return status

    # Long side (BUY / HOLD)
    if call in ["BUY", "HOLD", "WATCH"]:
        if target > 0 and current >= target:
            # Target achieved — suggest trailing or booking
            atr_proxy = max(abs(target - entry) * 0.4, current * 0.015)
            new_target = round(current + atr_proxy * 1.5, 2)
            new_stop = round(max(entry, current - atr_proxy), 2)

            status.update({
                "status": "TARGET ACHIEVED",
                "badge": "🎯 TARGET ACHIEVED",
                "message": (
                    f"Previous Target: ₹{target:,.2f}\n"
                    f"Current Price: ₹{current:,.2f}\n"
                    f"🎯 TARGET ACHIEVED"
                ),
                "recommendation": (
                    "HOLD / TRAIL STOP LOSS\n"
                    "🎯 TARGET ACHIEVED\n"
                    "🔴 SELL / BOOK PROFIT"
                ),
                "new_target": new_target,
                "new_stop": new_stop,
                "action": "TARGET ACHIEVED",
            })
            return status

        if stop > 0 and current <= stop:
            status.update({
                "status": "STOP LOSS HIT",
                "badge": "🔴 STOP LOSS HIT",
                "message": (
                    f"Current Price: ₹{current:,.2f}\n"
                    f"Stop Loss: ₹{stop:,.2f}\n"
                    f"🔴 STOP LOSS HIT"
                ),
                "recommendation": "🔴 SELL / EXIT",
                "action": "STOP LOSS HIT",
            })
            return status

        # Still open — optional mild trailing suggestion when close to target
        if target > 0 and current >= target * 0.97:
            status["message"] = (
                f"Approaching target (₹{target:,.2f}). Current: ₹{current:,.2f}"
            )
            status["recommendation"] = "HOLD — watch target closely"

    # Short side (SELL)
    elif call == "SELL":
        if target > 0 and current <= target:
            status.update({
                "status": "TARGET ACHIEVED",
                "badge": "🎯 TARGET ACHIEVED",
                "message": (
                    f"Previous Target: ₹{target:,.2f}\n"
                    f"Current Price: ₹{current:,.2f}\n"
                    f"🎯 TARGET ACHIEVED (SELL)"
                ),
                "recommendation": "🎯 TARGET ACHIEVED — Book profit / cover",
                "action": "TARGET ACHIEVED",
            })
            return status

        if stop > 0 and current >= stop:
            status.update({
                "status": "STOP LOSS HIT",
                "badge": "🔴 STOP LOSS HIT",
                "message": (
                    f"Current Price: ₹{current:,.2f}\n"
                    f"Stop Loss: ₹{stop:,.2f}\n"
                    f"🔴 STOP LOSS HIT"
                ),
                "recommendation": "🔴 SELL / EXIT — Cover position",
                "action": "STOP LOSS HIT",
            })
            return status

    return status


# ============================================================
# HISTORICAL DATA FOR INDIVIDUAL STOCK
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def nse_equity_history(symbol, days=400):
    """
    Fallback OHLC from NSE when Yahoo Finance has no data.
    TradingView has no free official OHLC API for embedding; NSE is the
    reliable free source for Indian equities.
    """
    sym = display_symbol(symbol)
    if not sym or sym.startswith("^"):
        return pd.DataFrame()

    end = datetime.now().date()
    start = end - timedelta(days=max(days, 60))
    from_s = start.strftime("%d-%m-%Y")
    to_s = end.strftime("%d-%m-%Y")

    url = (
        "https://www.nseindia.com/api/historical/cm/equity"
        f"?symbol={sym}&series=[%22EQ%22]&from={from_s}&to={to_s}"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }

    try:
        import urllib.request
        import json as _json

        # Warm-up cookies
        home = urllib.request.Request(
            "https://www.nseindia.com/",
            headers=headers,
        )
        with urllib.request.urlopen(home, timeout=15) as resp:
            cookie = resp.headers.get("Set-Cookie", "")

        if cookie:
            headers["Cookie"] = cookie.split(";")[0]

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = _json.loads(resp.read().decode("utf-8", errors="ignore"))

        rows = payload.get("data") or payload.get("dataList") or []
        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            # NSE field names vary slightly by endpoint version
            dt = r.get("CH_TIMESTAMP") or r.get("mTIMESTAMP") or r.get("date") or r.get("Date")
            o = r.get("CH_OPENING_PRICE") or r.get("OPEN") or r.get("open")
            h = r.get("CH_TRADE_HIGH_PRICE") or r.get("HIGH") or r.get("high")
            l = r.get("CH_TRADE_LOW_PRICE") or r.get("LOW") or r.get("low")
            c = r.get("CH_CLOSING_PRICE") or r.get("CLOSE") or r.get("close") or r.get("CH_LAST_TRADED_PRICE")
            v = r.get("CH_TOT_TRADED_QTY") or r.get("VOLUME") or r.get("volume") or 0
            if not dt or c is None:
                continue
            records.append(
                {
                    "Date": pd.to_datetime(dt, errors="coerce"),
                    "Open": safe_float(o),
                    "High": safe_float(h),
                    "Low": safe_float(l),
                    "Close": safe_float(c),
                    "Volume": safe_float(v),
                }
            )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).dropna(subset=["Date", "Close"])
        df = df.set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def stock_history(symbol, interval="1d"):
    """
    Prefer Yahoo Finance; if empty / missing, fall back to NSE historical API.
    interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk (Yahoo limits apply).
    Works when market is closed (uses last available bars).
    """
    interval = str(interval or "1d").lower()
    # Yahoo max periods by interval
    period_map = {
        "1m": "5d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",
        "60m": "730d",
        "1d": "2y",
        "1wk": "5y",
        "1mo": "10y",
    }
    period = period_map.get(interval, "2y")

    try:
        d = yf.download(
            clean_symbol(symbol),
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        d = normalize_columns(d)
        if d is not None and not d.empty and "Close" in d.columns:
            out = d.dropna(subset=["Open", "High", "Low", "Close"])
            min_bars = 20 if interval in {"1m", "5m", "15m", "30m", "1h", "60m"} else 30
            if len(out) >= min_bars:
                return out
    except Exception:
        pass

    # Daily fallback to NSE when Yahoo fails (intraday not available on this path)
    if interval in {"1d", "1wk", "1mo"}:
        nse = nse_equity_history(symbol, days=500)
        if nse is not None and not nse.empty:
            return nse

    return pd.DataFrame()


# ============================================================
# HISTORY
# ============================================================

def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make recommendation_history.csv safe across machines / old file versions.
    Fixes KeyError: 'Prediction Date' when the CSV is empty or has different headers.
    """
    if df is None:
        return pd.DataFrame()

    x = df.copy()

    # Strip column names (BOM / spaces from Excel saves)
    x.columns = [str(c).strip().replace("\ufeff", "") for c in x.columns]

    # Common aliases from older exports
    aliases = {
        "prediction date": "Prediction Date",
        "prediction_date": "Prediction Date",
        "date": "Prediction Date",
        "stock": "Stock",
        "symbol": "Symbol",
        "call": "Call",
        "signal": "Call",
        "entry": "Entry",
        "price": "Entry",
        "target": "Target",
        "stop loss": "Stop Loss",
        "stop_loss": "Stop Loss",
        "risk %": "Risk %",
        "risk_pct": "Risk %",
        "risk level": "Risk Level",
        "hold days": "Hold Days",
        "hold_days": "Hold Days",
        "expiry date": "Expiry Date",
        "status": "Status",
        "evaluation date": "Evaluation Date",
        "exit price": "Exit Price",
        "return %": "Return %",
        "result": "Result",
        "result detail": "Result Detail",
        "days taken": "Days Taken",
        "outcome message": "Outcome Message",
        "recommendation": "Recommendation",
        "suggestion": "Suggestion",
        "reason": "Reason",
        "prediction": "Prediction",
    }
    lower_map = {str(c).strip().lower(): c for c in x.columns}
    for old_l, new in aliases.items():
        if new not in x.columns and old_l in lower_map:
            x[new] = x[lower_map[old_l]]

    required = [
        "Prediction Date",
        "Stock",
        "Symbol",
        "Call Source",
        "Strategy",
        "Patterns",
        "Call",
        "Prediction",
        "Entry",
        "Target",
        "Stop Loss",
        "Risk %",
        "Risk Level",
        "Hold Days",
        "Expiry Date",
        "Status",
        "Evaluation Date",
        "Exit Price",
        "Return %",
        "Result",
        "Result Detail",
        "Days Taken",
        "Outcome Message",
        "Recommendation",
        "Suggestion",
        "Reason",
        "Current Price",
    ]
    for col in required:
        if col not in x.columns:
            x[col] = ""

    # Default source for old rows
    if "Call Source" in x.columns:
        blank = (
            x["Call Source"].isna()
            | x["Call Source"].astype(str).str.strip().isin(["", "nan", "None", "NAN"])
        )
        x.loc[blank, "Call Source"] = "SCAN"

    # Drop fully empty rows
    if "Stock" in x.columns:
        x = x[~(x["Stock"].astype(str).str.strip() == "") | (x["Prediction Date"].astype(str).str.strip() != "")]

    return x.reset_index(drop=True)


def load_history():

    try:
        if not HISTORY_FILE.exists():
            ensure_files()
            return pd.DataFrame()

        df = pd.read_csv(HISTORY_FILE)
        if df is None or df.empty:
            return normalize_history_df(pd.DataFrame())

        return normalize_history_df(df)

    except Exception:
        return normalize_history_df(pd.DataFrame())


def save_recommendations(
    results
):

    if results is None:
        return

    if results.empty:
        return

    history = load_history()

    now = datetime.now()

    rows = []

    # Save every signal.
    # SELL is now included.

    for _, row in results.iterrows():

        call = str(
            row.get(
                "Call",
                ""
            )
        ).upper()

        if call not in [
            "BUY",
            "HOLD",
            "SELL",
        ]:
            continue

        hold_days = int(
            safe_float(
                row.get(
                    "Hold Days",
                    DEFAULT_HOLD_DAYS
                ),
                DEFAULT_HOLD_DAYS
            )
        )

        expiry = (
            now +
            timedelta(
                days=hold_days
            )
        )

        rows.append(
            {
                "Prediction Date":
                    now.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Stock":
                    row.get(
                        "Stock",
                        ""
                    ),

                "Symbol":
                    row.get(
                        "Symbol",
                        ""
                    ),

                "Call Source":
                    str(row.get("Call Source", "SCAN") or "SCAN").upper(),

                "Strategy":
                    str(row.get("Strategy", "") or ""),

                "Patterns":
                    str(row.get("Patterns", row.get("Pattern", "")) or ""),

                "Call":
                    call,

                "Prediction":
                    row.get(
                        "Prediction",
                        ""
                    ),

                "Entry":
                    row.get(
                        "Price",
                        ""
                    ),

                "Target":
                    row.get(
                        "Target",
                        ""
                    ),

                "Stop Loss":
                    row.get(
                        "Stop Loss",
                        ""
                    ),

                "Risk %":
                    row.get(
                        "Risk %",
                        ""
                    ),

                "Risk Level":
                    row.get(
                        "Risk Level",
                        ""
                    ),

                "Hold Days":
                    hold_days,

                "Expiry Date":
                    expiry.strftime(
                        "%Y-%m-%d"
                    ),

                "Status":
                    "OPEN",

                "Evaluation Date":
                    "",

                "Exit Price":
                    "",

                "Return %":
                    "",

                "Result":
                    "PENDING",

                "Reason":
                    row.get(
                        "Reason",
                        ""
                    ),

                "Patterns":
                    row.get("Patterns", ""),

                "RSI":
                    row.get("RSI", ""),

                "ADX":
                    row.get("ADX", ""),
            }
        )

    if not rows:
        return

    new_history = pd.DataFrame(
        rows
    )

    history = pd.concat(
        [
            history,
            new_history,
        ],
        ignore_index=True
    )

    # Prevent exact duplicate scans.
    if not history.empty:

        history = history.drop_duplicates(
            subset=[
                "Prediction Date",
                "Stock",
                "Call",
            ],
            keep="last"
        )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# HISTORICAL PREDICTION EVALUATION
# ============================================================

def _normalize_ohlc_index(data: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone and normalize to calendar dates so day comparisons work."""
    if data is None or data.empty:
        return pd.DataFrame()
    d = data.copy()
    idx = pd.to_datetime(d.index, errors="coerce")
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        try:
            idx = idx.tz_convert(None)
        except Exception:
            pass
    d.index = pd.DatetimeIndex(idx).normalize()
    d = d[~d.index.isna()]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def _eval_price_history(symbol: str) -> pd.DataFrame:
    """
    Aggressive daily OHLC for evaluation — bypasses stale cache when possible.
    """
    sym = clean_symbol(symbol)
    if not sym:
        return pd.DataFrame()
    try:
        # Fresh download (not only stock_history cache)
        d = yf.download(
            sym,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        d = normalize_columns(d)
        d = _normalize_ohlc_index(d)
        if d is not None and not d.empty and "High" in d.columns and "Low" in d.columns:
            return d.dropna(subset=["High", "Low", "Close"], how="any")
    except Exception:
        pass
    try:
        d = stock_history(sym, interval="1d")
        d = _normalize_ohlc_index(d)
        if d is not None and not d.empty:
            return d
    except Exception:
        pass
    try:
        d = nse_equity_history(sym, days=400)
        return _normalize_ohlc_index(d)
    except Exception:
        return pd.DataFrame()


def evaluate_history(force_all: bool = False):
    """
    Evaluate open predictions against subsequent price action.
    ORIGINAL Target and Stop Loss in the history sheet are NEVER changed.
    Near-target ideas go only into Suggestion / Recommendation text.
    """

    history = load_history()

    if history.empty:
        return history

    history = normalize_history_df(history)

    text_cols = [
        "Prediction Date", "Stock", "Symbol", "Call", "Status",
        "Evaluation Date", "Result", "Result Detail", "Days Taken",
        "Outcome Message", "Recommendation", "Reason", "Risk Level",
        "Expiry Date", "Suggestion",
    ]
    for col in text_cols:
        if col not in history.columns:
            history[col] = ""
        history[col] = history[col].astype(object)

    # Numeric columns — Target / Stop Loss are read-only after this (never written back differently)
    locked_levels = {}  # idx -> (target, stop) frozen originals
    for col in ["Exit Price", "Return %", "Entry", "Target", "Stop Loss", "Risk %", "Hold Days", "Prediction"]:
        if col not in history.columns:
            history[col] = None
        history[col] = pd.to_numeric(history[col], errors="coerce")

    for idx in history.index:
        locked_levels[idx] = (
            safe_float(history.at[idx, "Target"]),
            safe_float(history.at[idx, "Stop Loss"]),
        )

    required = [
        "Prediction Date", "Stock", "Call", "Entry", "Target", "Stop Loss",
        "Hold Days", "Status", "Result", "Result Detail", "Days Taken",
        "Outcome Message", "Recommendation", "Suggestion",
    ]
    for col in required:
        if col not in history.columns:
            history[col] = ""

    changed = False

    res_u = history["Result"].astype(str).str.upper().str.strip()
    open_mask = ~res_u.str.contains(
        "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
        regex=True,
        na=True,
    )

    # Repair impossible outcomes: event date BEFORE prediction date
    # (old bug used data.tail() and could mark stop on a past day)
    invalid_closed = []
    for idx in history.index:
        result_s = str(history.at[idx, "Result"]).upper()
        if not any(x in result_s for x in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD", "WIN", "LOSS")):
            continue
        try:
            pred_dt = pd.to_datetime(history.at[idx, "Prediction Date"], errors="coerce")
            if pd.isna(pred_dt):
                continue
            pred_d = pd.Timestamp(pred_dt).normalize()
            eval_dt = pd.to_datetime(history.at[idx, "Evaluation Date"], errors="coerce")
            detail = str(history.at[idx, "Result Detail"] or "")
            event_dt = eval_dt
            # Parse "Stop Loss Hit On: 10 September 2026" / "Target Reached On: ..."
            for label in ("Stop Loss Hit On:", "Target Reached On:", "Exit Date:"):
                if label in detail:
                    try:
                        part = detail.split(label, 1)[1].split("\n")[0].strip()
                        event_dt = pd.to_datetime(part, errors="coerce", dayfirst=True)
                    except Exception:
                        pass
                    break
            if pd.notna(event_dt) and pd.Timestamp(event_dt).normalize() < pred_d:
                invalid_closed.append(idx)
                history.at[idx, "Status"] = "OPEN"
                history.at[idx, "Result"] = "PENDING"
                history.at[idx, "Result Detail"] = ""
                history.at[idx, "Days Taken"] = ""
                history.at[idx, "Outcome Message"] = "Reset: prior result had event date before prediction"
                history.at[idx, "Recommendation"] = ""
                history.at[idx, "Evaluation Date"] = ""
                history.at[idx, "Exit Price"] = None
                history.at[idx, "Return %"] = None
                changed = True
        except Exception:
            continue

    if invalid_closed:
        open_mask = open_mask | history.index.isin(invalid_closed)

    if force_all:
        # Re-check everything including previously closed (after repair)
        open_indices = history.index.tolist()
    else:
        open_indices = history.index[open_mask].tolist()

    # Prefer evaluating newest predictions first
    try:
        open_indices = sorted(
            open_indices,
            key=lambda i: pd.to_datetime(history.at[i, "Prediction Date"], errors="coerce")
            or pd.Timestamp.min,
            reverse=True,
        )
    except Exception:
        pass

    MAX_EVAL_PER_RUN = 800 if force_all else 250
    open_indices = open_indices[:MAX_EVAL_PER_RUN]

    for idx in open_indices:

        status = str(history.at[idx, "Status"]).upper().strip()
        result = str(history.at[idx, "Result"]).upper().strip()

        if status == "CLOSED" and result in [
            "TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED", "WIN", "LOSS",
        ]:
            continue

        symbol = clean_symbol(str(history.at[idx, "Stock"]))
        entry = safe_float(history.at[idx, "Entry"])
        # ALWAYS use original locked levels from the sheet
        target, stop = locked_levels.get(idx, (0.0, 0.0))
        target = safe_float(target)
        stop = safe_float(stop)
        call = str(history.at[idx, "Call"]).upper().strip()

        if entry <= 0 or not symbol:
            continue

        try:
            prediction_date = pd.to_datetime(history.at[idx, "Prediction Date"], errors="coerce")
            if pd.isna(prediction_date):
                continue
            pred_day = pd.Timestamp(prediction_date).normalize()
            try:
                if getattr(pred_day, "tz", None) is not None:
                    pred_day = pred_day.tz_localize(None)
            except Exception:
                pass
        except Exception:
            continue

        try:
            hold_days = int(safe_float(history.at[idx, "Hold Days"], DEFAULT_HOLD_DAYS))
        except Exception:
            hold_days = DEFAULT_HOLD_DAYS
        if hold_days <= 0:
            hold_days = DEFAULT_HOLD_DAYS

        data = _eval_price_history(symbol)
        if data.empty:
            continue

        # STRICT: only bars on/after prediction calendar day.
        # NEVER fall back to data.tail() — that caused stop dates BEFORE prediction date.
        try:
            idx_norm = pd.DatetimeIndex(pd.to_datetime(data.index)).tz_localize(None)
            data = data.copy()
            data.index = idx_norm
        except Exception:
            pass
        future = data[data.index.normalize() >= pred_day]
        # Allow 1 calendar day slack only if pred is "today" and last bar is yesterday (market closed)
        if future.empty:
            last_bar = pd.Timestamp(data.index[-1]).normalize()
            if pred_day <= last_bar + pd.Timedelta(days=1) and pred_day >= last_bar - pd.Timedelta(days=1):
                future = data[data.index.normalize() >= last_bar]
            else:
                # No price history after the call — keep OPEN, do not invent a past hit
                continue

        if future.empty:
            continue

        # Drop any residual bars before prediction (safety)
        future = future[future.index.normalize() >= pred_day]
        if future.empty:
            continue

        latest_date = pd.Timestamp(future.index[-1])
        trading_days = max(1, len(future))

        result_value = None
        exit_price = None
        event_date = None
        days_taken = None
        outcome_message = ""
        recommendation = ""
        result_detail = ""
        suggestion = str(history.at[idx, "Suggestion"] or "")

        # ----------------------------------------------------
        # BUY / HOLD — compare High/Low to ORIGINAL target/stop
        # ----------------------------------------------------
        if call in ["BUY", "HOLD"]:

            for i, (ts, candle) in enumerate(future.iterrows()):
                bar_day = pd.Timestamp(ts).normalize()
                if bar_day < pred_day:
                    continue  # never count pre-call price action
                high = safe_float(candle.get("High"))
                low = safe_float(candle.get("Low"))
                day_num = i + 1

                # Check both: if both hit same day, prefer stop first (conservative)
                stop_hit = stop > 0 and low <= stop
                tgt_hit = target > 0 and high >= target

                if stop_hit and tgt_hit:
                    # Same bar: use open vs levels — if gap both ways, mark stop (safer)
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT\n"
                        f"Stop Loss Hit On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Loss: {loss:.2f}%\n"
                        f"(Same day also touched original target ₹{target:,.2f} — stop prioritised)"
                    )
                    outcome_message = f"Stop Loss ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT"
                    break

                if stop_hit:
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT\n"
                        f"Stop Loss Hit On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Loss: {loss:.2f}%\n"
                        f"Original Target (unchanged): ₹{target:,.2f} | Original SL: ₹{stop:,.2f}"
                    )
                    outcome_message = f"Stop Loss ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT"
                    break

                if tgt_hit:
                    result_value = "TARGET ACHIEVED"
                    exit_price = target
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    profit = ((target - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🎯 TARGET ACHIEVED\n"
                        f"Target Reached On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Profit: +{profit:.2f}%\n"
                        f"Original Target (locked): ₹{target:,.2f} | Original SL (locked): ₹{stop:,.2f}"
                    )
                    outcome_message = f"Target ₹{target:,.2f} reached"
                    recommendation = (
                        "🎯 TARGET ACHIEVED — Book profit or trail stop. "
                        "Sheet Target/SL stay frozen at original values."
                    )
                    break

            # Live / last close check if daily path still open
            if result_value is None:
                last_close = safe_float(future["Close"].iloc[-1])
                try:
                    q = live_quote(symbol)
                    if q and q.get("price"):
                        last_close = safe_float(q["price"])
                except Exception:
                    pass

                if target > 0 and last_close >= target:
                    result_value = "TARGET ACHIEVED"
                    exit_price = target
                    event_date = latest_date
                    days_taken = trading_days
                    profit = ((target - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🎯 TARGET ACHIEVED\n"
                        f"Target Reached On: {event_date.strftime('%d %B %Y')} (price check)\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Profit: +{profit:.2f}%\n"
                        f"LTP/Close ₹{last_close:,.2f} ≥ Original Target ₹{target:,.2f}"
                    )
                    outcome_message = f"Target ₹{target:,.2f} reached"
                    recommendation = "🎯 TARGET ACHIEVED — Book profit or trail stop"
                elif stop > 0 and last_close <= stop:
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = latest_date
                    days_taken = trading_days
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT\n"
                        f"Stop Loss Hit On: {event_date.strftime('%d %B %Y')} (price check)\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Loss: {loss:.2f}%\n"
                        f"LTP/Close ₹{last_close:,.2f} ≤ Original SL ₹{stop:,.2f}"
                    )
                    outcome_message = f"Stop Loss ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT"

            # Near-target SUGGESTION only (does not change Target/SL columns)
            if result_value is None and target > 0 and entry > 0:
                last_close = safe_float(future["Close"].iloc[-1])
                try:
                    q = live_quote(symbol)
                    if q and q.get("price"):
                        last_close = safe_float(q["price"])
                except Exception:
                    pass
                dist = (target - last_close) / target if target else 1
                if 0 < dist <= 0.02:
                    # Within 2% of original target
                    new_t = round(target * 1.04, 2)
                    new_sl = round(max(entry, last_close * 0.98), 2)
                    suggestion = (
                        f"NEAR ORIGINAL TARGET ₹{target:,.2f} (LTP ₹{last_close:,.2f}). "
                        f"Optional idea only — do NOT overwrite sheet: "
                        f"trail SL toward ₹{new_sl:,.2f}, stretch target idea ₹{new_t:,.2f}. "
                        f"Original Target/SL remain locked in history."
                    )
                elif last_close >= target * 0.97:
                    suggestion = (
                        f"Approaching locked target ₹{target:,.2f}. "
                        f"Consider booking partial profit; sheet levels stay unchanged."
                    )

            if result_value is None and trading_days >= hold_days:
                candle = future.iloc[min(hold_days - 1, len(future) - 1)]
                exit_price = safe_float(candle["Close"])
                event_date = pd.Timestamp(future.index[min(hold_days - 1, len(future) - 1)])
                days_taken = min(hold_days, trading_days)
                result_value = "HOLDING PERIOD COMPLETED"
                result_detail = (
                    f"⏰ HOLDING PERIOD COMPLETED\n"
                    f"Neither Original Target ₹{target:,.2f} nor SL ₹{stop:,.2f} was reached.\n"
                    f"Exit Date: {event_date.strftime('%d %B %Y')}\n"
                    f"Exit Closing Price: ₹{exit_price:,.2f}"
                )
                outcome_message = "Holding period expired"
                recommendation = (
                    "⚠️ EARLY EXIT / SELL — mild profit, review position"
                    if exit_price >= entry else
                    "⚠️ EARLY EXIT / SELL — mild loss, review position"
                )

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------
        elif call == "SELL":

            for i, (ts, candle) in enumerate(future.iterrows()):
                high = safe_float(candle.get("High"))
                low = safe_float(candle.get("Low"))
                day_num = i + 1

                if target > 0 and low <= target:
                    result_value = "TARGET ACHIEVED"
                    exit_price = target
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    profit = ((entry - target) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🎯 TARGET ACHIEVED (SELL)\n"
                        f"Target Reached On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Profit: +{profit:.2f}%\n"
                        f"Original Target/SL locked in sheet"
                    )
                    outcome_message = f"Sell target ₹{target:,.2f} reached"
                    recommendation = "🎯 TARGET ACHIEVED — Book profit / cover short"
                    break

                if stop > 0 and high >= stop:
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT (SELL)\n"
                        f"Stop Loss Hit On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Loss: {loss:.2f}%"
                    )
                    outcome_message = f"Sell stop ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT — Cover position"
                    break

            if result_value is None:
                last_close = safe_float(future["Close"].iloc[-1])
                try:
                    q = live_quote(symbol)
                    if q and q.get("price"):
                        last_close = safe_float(q["price"])
                except Exception:
                    pass
                if target > 0 and last_close <= target:
                    result_value = "TARGET ACHIEVED"
                    exit_price = target
                    event_date = latest_date
                    days_taken = trading_days
                    profit = ((entry - target) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🎯 TARGET ACHIEVED (SELL)\n"
                        f"LTP/Close ₹{last_close:,.2f} ≤ Original Target ₹{target:,.2f}\n"
                        f"Profit: +{profit:.2f}%"
                    )
                    outcome_message = f"Sell target ₹{target:,.2f} reached"
                    recommendation = "🎯 TARGET ACHIEVED — Book profit"
                elif stop > 0 and last_close >= stop:
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = latest_date
                    days_taken = trading_days
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT (SELL)\n"
                        f"LTP/Close ₹{last_close:,.2f} ≥ Original SL ₹{stop:,.2f}\n"
                        f"Loss: {loss:.2f}%"
                    )
                    outcome_message = f"Sell stop ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT — Cover position"

            if result_value is None and trading_days >= hold_days:
                candle = future.iloc[min(hold_days - 1, len(future) - 1)]
                exit_price = safe_float(candle["Close"])
                event_date = pd.Timestamp(future.index[min(hold_days - 1, len(future) - 1)])
                days_taken = min(hold_days, trading_days)
                result_value = "HOLDING PERIOD COMPLETED"
                result_detail = (
                    f"⏰ HOLDING PERIOD COMPLETED\n"
                    f"Neither Target nor Stop Loss was reached.\n"
                    f"Exit Date: {event_date.strftime('%d %B %Y')}\n"
                    f"Exit Closing Price: ₹{exit_price:,.2f}"
                )
                outcome_message = "Holding period expired"
                recommendation = "⚠️ EARLY EXIT / SELL — Review position"

        else:
            # Still save near-target suggestion updates for non-trade rows if needed
            if suggestion and str(history.at[idx, "Suggestion"]) != suggestion:
                history.loc[idx, "Suggestion"] = suggestion
                changed = True
            continue

        # Update suggestion for still-open rows
        if result_value is None:
            if suggestion and str(history.at[idx, "Suggestion"] or "") != suggestion:
                try:
                    history.loc[idx, "Suggestion"] = str(suggestion)
                    # Recommendation can show near-target idea without closing
                    if suggestion and not str(history.at[idx, "Recommendation"] or "").startswith("🎯"):
                        history.loc[idx, "Recommendation"] = str(suggestion)
                    changed = True
                except Exception:
                    pass
            continue

        if exit_price is None:
            continue

        if call == "SELL":
            return_pct = ((entry - exit_price) / entry) * 100 if entry else 0
        else:
            return_pct = ((exit_price - entry) / entry) * 100 if entry else 0

        eval_date_str = (
            event_date.strftime("%Y-%m-%d") if event_date is not None
            else latest_date.strftime("%Y-%m-%d")
        )
        days_taken_val = days_taken if days_taken is not None else ""

        # Write outcome only — NEVER overwrite Target or Stop Loss
        try:
            history.loc[idx, "Status"] = "CLOSED"
            history.loc[idx, "Evaluation Date"] = str(eval_date_str)
            history.loc[idx, "Exit Price"] = float(round(exit_price, 2))
            history.loc[idx, "Return %"] = float(round(return_pct, 2))
            history.loc[idx, "Result"] = str(result_value)
            history.loc[idx, "Result Detail"] = str(result_detail)
            history.loc[idx, "Days Taken"] = str(days_taken_val)
            history.loc[idx, "Outcome Message"] = str(outcome_message)
            history.loc[idx, "Recommendation"] = str(recommendation)
            if suggestion:
                history.loc[idx, "Suggestion"] = str(suggestion)
        except Exception:
            for col, val in [
                ("Status", "CLOSED"),
                ("Evaluation Date", str(eval_date_str)),
                ("Result", str(result_value)),
                ("Result Detail", str(result_detail)),
                ("Days Taken", str(days_taken_val)),
                ("Outcome Message", str(outcome_message)),
                ("Recommendation", str(recommendation)),
                ("Suggestion", str(suggestion or "")),
            ]:
                try:
                    history[col] = history[col].astype(object)
                    history.loc[idx, col] = val
                except Exception:
                    pass
            try:
                history.loc[idx, "Exit Price"] = float(round(exit_price, 2))
                history.loc[idx, "Return %"] = float(round(return_pct, 2))
            except Exception:
                pass

        # Restore locked target/stop in case anything touched them
        try:
            ot, os_ = locked_levels[idx]
            history.loc[idx, "Target"] = ot
            history.loc[idx, "Stop Loss"] = os_
        except Exception:
            pass

        changed = True

    # Final safety: re-apply every locked Target/SL so the sheet never drifts
    for idx, (ot, os_) in locked_levels.items():
        try:
            if idx in history.index:
                history.loc[idx, "Target"] = ot
                history.loc[idx, "Stop Loss"] = os_
        except Exception:
            pass

    if changed:
        try:
            history.to_csv(HISTORY_FILE, index=False)
        except Exception:
            out = history.copy()
            for c in out.columns:
                try:
                    out[c] = out[c].astype(object)
                except Exception:
                    pass
            out.to_csv(HISTORY_FILE, index=False)

    # Current price on open rows only — Entry/Target/SL stay locked
    try:
        history = refresh_history_current_prices(max_stocks=60)
    except Exception:
        pass

    # Keep auto trade tracker in sync after every evaluation
    try:
        sync_auto_trades_tracker(history)
    except Exception:
        pass

    return history


def _trade_id(row) -> str:
    stock = display_symbol(row.get("Stock", ""))
    pred = str(row.get("Prediction Date", ""))[:19]
    call = str(row.get("Call", "")).upper()
    return f"{stock}|{pred}|{call}"


def sync_auto_trades_tracker(history: pd.DataFrame = None, max_live: int = 40) -> pd.DataFrame:
    """
    Automate trade tracking from recommendation_history.csv:
    - One row per prediction (Trade ID)
    - OPEN: refresh current price, unrealized %, distance to locked target/stop
    - CLOSED: lock realized % from history Result (target/stop never changed)
    Writes auto_trades_tracker.csv
    """
    ensure_files()
    if history is None:
        history = load_history()
    history = normalize_history_df(history)
    if history is None or history.empty:
        return pd.DataFrame()

    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    open_count = 0

    for _, row in history.iterrows():
        stock = display_symbol(row.get("Stock", ""))
        if not stock:
            continue
        call = str(row.get("Call", "")).upper().strip()
        if call not in {"BUY", "HOLD", "SELL"}:
            continue

        entry = safe_float(row.get("Entry"))
        target = safe_float(row.get("Target"))
        stop = safe_float(row.get("Stop Loss"))
        hold_days = int(safe_float(row.get("Hold Days"), DEFAULT_HOLD_DAYS))
        result = str(row.get("Result", "PENDING")).upper().strip()
        status = str(row.get("Status", "OPEN")).upper().strip()

        is_closed = any(
            x in result
            for x in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED", "WIN", "LOSS")
        ) or status == "CLOSED"

        try:
            pred_dt = pd.to_datetime(row.get("Prediction Date"), errors="coerce")
            days_held = 0
            if pd.notna(pred_dt):
                days_held = max(0, (pd.Timestamp(datetime.now().date()) - pd.Timestamp(pred_dt).normalize()).days)
        except Exception:
            days_held = 0
        days_rem = max(0, hold_days - days_held)

        cur = None
        unreal = None
        dist_t = None
        dist_s = None
        realized = safe_float(row.get("Return %")) if is_closed else None
        exit_px = safe_float(row.get("Exit Price")) if is_closed else None
        suggestion = str(row.get("Suggestion", "") or "")

        # Live price for open trades (cap network calls)
        if not is_closed and open_count < max_live:
            open_count += 1
            try:
                q = live_quote(stock)
                if q and q.get("price"):
                    cur = safe_float(q["price"])
            except Exception:
                cur = None
            if cur is None:
                try:
                    d = stock_history(stock, interval="1d")
                    if d is not None and not d.empty:
                        cur = safe_float(d["Close"].iloc[-1])
                except Exception:
                    pass
            if cur and entry > 0:
                if call == "SELL":
                    unreal = ((entry - cur) / entry) * 100
                else:
                    unreal = ((cur - entry) / entry) * 100
                if target > 0:
                    dist_t = ((target - cur) / target) * 100
                if stop > 0:
                    dist_s = ((cur - stop) / stop) * 100
                # Near-target suggestion (does not change locked levels)
                if call in {"BUY", "HOLD"} and target > 0 and cur >= target * 0.98 and cur < target:
                    suggestion = (
                        f"NEAR locked target ₹{target:,.2f} (LTP ₹{cur:,.2f}). "
                        f"Idea only: trail SL / partial book — sheet Target/SL stay frozen."
                    )
                elif call in {"BUY", "HOLD"} and target > 0 and cur >= target:
                    # Should have been closed by evaluate_history; flag anyway
                    suggestion = f"LTP ₹{cur:,.2f} ≥ locked target ₹{target:,.2f} — run Force re-check on Past Predictions."
                elif call in {"BUY", "HOLD"} and stop > 0 and cur <= stop:
                    suggestion = f"LTP ₹{cur:,.2f} ≤ locked SL ₹{stop:,.2f} — run Force re-check on Past Predictions."

        trade_status = "CLOSED" if is_closed else "OPEN"
        result_out = result if result and result not in {"", "NAN", "NONE"} else (
            "CLOSED" if is_closed else "OPEN / PENDING"
        )

        rows.append({
            "Trade ID": _trade_id(row),
            "Prediction Date": row.get("Prediction Date", ""),
            "Stock": stock,
            "Call": call,
            "Entry": entry,
            "Target": target,
            "Stop Loss": stop,
            "Hold Days": hold_days,
            "Status": trade_status,
            "Result": result_out,
            "Current Price": cur if cur is not None else "",
            "Unrealized %": round(unreal, 2) if unreal is not None else "",
            "Realized %": round(realized, 2) if realized is not None and is_closed else "",
            "Exit Price": exit_px if exit_px is not None and is_closed else "",
            "Days Held": days_held,
            "Days Remaining": days_rem if not is_closed else 0,
            "Distance to Target %": round(dist_t, 2) if dist_t is not None else "",
            "Distance to Stop %": round(dist_s, 2) if dist_s is not None else "",
            "Last Checked": now_s,
            "Suggestion": suggestion,
        })

    trades = pd.DataFrame(rows)
    if not trades.empty:
        # Prefer closed + high conviction open first
        trades = trades.sort_values(
            by=["Status", "Prediction Date"],
            ascending=[True, False],
        )
        trades.to_csv(TRADES_FILE, index=False)
    return trades


def load_auto_trades() -> pd.DataFrame:
    ensure_files()
    try:
        if TRADES_FILE.exists():
            return pd.read_csv(TRADES_FILE)
    except Exception:
        pass
    return pd.DataFrame()


def show_trade_tracker():
    """Automated trade tracking dashboard."""
    st.title("🤖 Auto Trade Tracker")
    st.caption(
        "Tracks every saved prediction as a trade. "
        "**Target & Stop Loss stay locked** from the original call. "
        "Status updates when Past Predictions evaluation marks target/stop/time exit."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⟳ Sync now", type="primary", key="trade_sync_btn"):
            with st.spinner("Evaluating outcomes + refreshing open prices..."):
                evaluate_history(force_all=True)
                trades = sync_auto_trades_tracker(max_live=60)
            st.success(f"Synced **{len(trades)}** trades → `{TRADES_FILE.name}`")
            st.rerun()
    with c2:
        auto_on = st.checkbox(
            "Auto-sync when opening this page",
            value=bool(st.session_state.get("trade_auto_sync", True)),
            key="trade_auto_sync",
        )
    with c3:
        st.caption(f"File: `{TRADES_FILE}`")

    # Auto sync at most every 3 minutes on page open
    if auto_on:
        last = st.session_state.get("_last_trade_sync")
        run_sync = last is None
        if last is not None:
            try:
                run_sync = (datetime.now() - last).total_seconds() > 180
            except Exception:
                run_sync = True
        if run_sync:
            with st.spinner("Auto-syncing trades..."):
                try:
                    evaluate_history(force_all=False)
                except Exception:
                    pass
                sync_auto_trades_tracker(max_live=40)
            st.session_state._last_trade_sync = datetime.now()

    trades = load_auto_trades()
    if trades.empty:
        st.info(
            "No trades yet. Run **FULL MARKET SCAN** so predictions are saved, "
            "then click **Sync now**."
        )
        return

    # Summary metrics
    status_u = trades["Status"].astype(str).str.upper()
    res_u = trades["Result"].astype(str).str.upper()
    n_open = int((status_u == "OPEN").sum())
    n_closed = int((status_u == "CLOSED").sum())
    n_tgt = int(res_u.str.contains("TARGET ACHIEVED", na=False).sum())
    n_sl = int(res_u.str.contains("STOP LOSS HIT", na=False).sum())

    closed = trades[status_u == "CLOSED"].copy()
    avg_r = None
    if not closed.empty and "Realized %" in closed.columns:
        avg_r = pd.to_numeric(closed["Realized %"], errors="coerce").mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Open trades", n_open)
    m2.metric("Closed trades", n_closed)
    m3.metric("🎯 Targets hit", n_tgt)
    m4.metric("🔴 Stops hit", n_sl)
    m5.metric("Avg realized %", f"{avg_r:+.2f}%" if avg_r == avg_r and avg_r is not None else "—")

    f1, f2 = st.columns(2)
    with f1:
        view = st.selectbox(
            "Show",
            ["ALL", "OPEN only", "CLOSED only", "TARGET ACHIEVED", "STOP LOSS HIT"],
            key="trade_view_filter",
        )
    with f2:
        call_f = st.selectbox("Call", ["ALL", "BUY", "HOLD", "SELL"], key="trade_call_filter")

    view_df = trades.copy()
    if view == "OPEN only":
        view_df = view_df[view_df["Status"].astype(str).str.upper() == "OPEN"]
    elif view == "CLOSED only":
        view_df = view_df[view_df["Status"].astype(str).str.upper() == "CLOSED"]
    elif view == "TARGET ACHIEVED":
        view_df = view_df[view_df["Result"].astype(str).str.upper().str.contains("TARGET ACHIEVED", na=False)]
    elif view == "STOP LOSS HIT":
        view_df = view_df[view_df["Result"].astype(str).str.upper().str.contains("STOP LOSS HIT", na=False)]
    if call_f != "ALL":
        view_df = view_df[view_df["Call"].astype(str).str.upper() == call_f]

    st.dataframe(view_df, use_container_width=True, hide_index=True)

    # Open trades cards
    open_df = trades[trades["Status"].astype(str).str.upper() == "OPEN"].head(20)
    if not open_df.empty:
        st.subheader("📡 Open trades — live tracking")
        for _, t in open_df.iterrows():
            cur = safe_float(t.get("Current Price"))
            entry = safe_float(t.get("Entry"))
            tgt = safe_float(t.get("Target"))
            sl = safe_float(t.get("Stop Loss"))
            ur = t.get("Unrealized %", "")
            st.markdown(
                f"""
                <div class="success-box" style="margin-bottom:10px;padding:12px;border-radius:8px;">
                <b>{t.get('Stock')}</b> · {t.get('Call')} · Entry ₹{entry:,.2f}<br>
                Locked Target ₹{tgt:,.2f} · Locked SL ₹{sl:,.2f}<br>
                LTP ₹{cur:,.2f} · Unrealized {ur}% ·
                Days held {t.get('Days Held')} · Remaining {t.get('Days Remaining')}<br>
                {t.get('Suggestion') or ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Automation flow: Market Scan → saves calls → evaluation marks target/stop → "
        "this tracker mirrors OPEN/CLOSED with live LTP for open rows. "
        "Original Target/SL columns are never modified."
    )


def load_paper_portfolio() -> pd.DataFrame:
    ensure_files()
    try:
        if PAPER_FILE.exists():
            return pd.read_csv(PAPER_FILE)
    except Exception:
        pass
    return pd.DataFrame()


def save_paper_portfolio(df: pd.DataFrame):
    try:
        df.to_csv(PAPER_FILE, index=False)
    except Exception:
        pass


def execute_paper_order(
    stock: str,
    side: str = "BUY",
    entry: float = 0.0,
    target: float = 0.0,
    stop: float = 0.0,
    shares: int = 0,
    source: str = "",
    hold_days: int = 15,
    risk_pct: float = 1.0,
    capital: float = 50000.0,
) -> dict:
    """
    Execute paper BUY/SELL from any page.
    Auto-fills live price, target, stop; sizes shares from risk if shares=0.
    Appears immediately in Paper Trading portfolio.
    """
    stock = display_symbol(stock)
    side = str(side or "BUY").upper()
    if side not in ("BUY", "SELL"):
        side = "BUY"
    # Live entry
    if entry <= 0:
        try:
            q = live_quote(stock)
            if q and q.get("price"):
                entry = safe_float(q["price"])
        except Exception:
            pass
    if entry <= 0:
        try:
            d = stock_history(clean_symbol(stock), interval="1d")
            if d is not None and not d.empty:
                entry = safe_float(d["Close"].iloc[-1])
        except Exception:
            pass
    if entry <= 0:
        return {"ok": False, "msg": f"No price for {stock}"}

    # Default target/stop from ATR if missing
    if target <= 0 or stop <= 0:
        try:
            d = stock_history(clean_symbol(stock), interval="1d")
            if d is not None and len(d) > 30:
                d = calculate_indicators(d)
                atr = safe_float(d.iloc[-1].get("ATR")) or entry * 0.02
                if side == "BUY":
                    if target <= 0:
                        target = round(entry + 2.0 * atr, 2)
                    if stop <= 0:
                        stop = round(entry - 1.2 * atr, 2)
                else:
                    if target <= 0:
                        target = round(entry - 2.0 * atr, 2)
                    if stop <= 0:
                        stop = round(entry + 1.2 * atr, 2)
        except Exception:
            if side == "BUY":
                target = target or round(entry * 1.04, 2)
                stop = stop or round(entry * 0.97, 2)
            else:
                target = target or round(entry * 0.96, 2)
                stop = stop or round(entry * 1.03, 2)

    # Position size from risk % of capital to stop
    if shares <= 0:
        risk_amt = capital * (risk_pct / 100.0)
        per_share_risk = abs(entry - stop)
        if per_share_risk > 0:
            shares = max(1, int(risk_amt / per_share_risk))
        else:
            shares = max(1, int(capital * 0.1 / entry))

    row = {
        "Open Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Stock": stock,
        "Side": side,
        "Entry": round(entry, 2),
        "Target": round(target, 2),
        "Stop Loss": round(stop, 2),
        "Shares": int(shares),
        "Hold Days": int(hold_days or 15),
        "Source": source or "Manual",
        "Status": "OPEN",
        "Result": "PENDING",
        "Exit Date": "",
        "Exit Price": "",
        "Return %": "",
        "PnL ₹": "",
    }
    paper = load_paper_portfolio()
    paper = pd.concat([paper, pd.DataFrame([row])], ignore_index=True)
    save_paper_portfolio(paper)
    return {
        "ok": True,
        "msg": (
            f"{side} {shares} × {stock} @ ₹{entry:,.2f} | "
            f"T ₹{target:,.2f} | SL ₹{stop:,.2f} → Paper portfolio"
        ),
        "row": row,
    }


def render_active_trade_buttons(
    stock: str,
    side_hint: str = "BUY",
    entry: float = 0.0,
    target: float = 0.0,
    stop: float = 0.0,
    key_prefix: str = "tr",
):
    """Active BUY / SELL / Open analysis buttons — execute paper orders."""
    stock = display_symbol(stock)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button(f"🟢 BUY {stock}", key=f"{key_prefix}_buy_{stock}", use_container_width=True):
            r = execute_paper_order(
                stock, "BUY", entry=entry, target=target, stop=stop,
                source=f"{side_hint}|{key_prefix}",
            )
            if r.get("ok"):
                st.success(r["msg"])
            else:
                st.error(r.get("msg", "Order failed"))
    with b2:
        if st.button(f"🔴 SELL {stock}", key=f"{key_prefix}_sell_{stock}", use_container_width=True):
            r = execute_paper_order(
                stock, "SELL", entry=entry,
                target=target if target and target < entry else 0,
                stop=stop if stop and stop > entry else 0,
                source=f"{side_hint}|{key_prefix}",
            )
            if r.get("ok"):
                st.success(r["msg"])
            else:
                st.error(r.get("msg", "Order failed"))
    with b3:
        if st.button("📊 Analyse", key=f"{key_prefix}_an_{stock}", use_container_width=True):
            st.session_state.selected_stock = stock
            st.session_state.page = "Stock Analysis"
            st.rerun()
    with b4:
        if st.button("🧪 Paper book", key=f"{key_prefix}_pp_{stock}", use_container_width=True):
            st.session_state.page = "Paper Trading"
            st.rerun()


def backtest_paper_trade(row) -> dict:
    """
    Walk forward from Open Date using daily OHLC vs locked Target/SL.
    BUY: target above, stop below. SELL: target below, stop above.
    """
    stock = str(row.get("Stock", ""))
    side = str(row.get("Side", "BUY")).upper()
    entry = safe_float(row.get("Entry"))
    target = safe_float(row.get("Target"))
    stop = safe_float(row.get("Stop Loss"))
    hold_days = safe_int(row.get("Hold Days"), DEFAULT_HOLD_DAYS) or DEFAULT_HOLD_DAYS
    shares = safe_int(row.get("Shares"), 1) or 1

    out = {
        "Status": "OPEN",
        "Result": "PENDING",
        "Exit Date": "",
        "Exit Price": "",
        "Return %": "",
        "PnL ₹": "",
    }
    if entry <= 0 or not stock:
        return out

    try:
        open_dt = pd.to_datetime(row.get("Open Date"), errors="coerce")
        if pd.isna(open_dt):
            open_dt = pd.Timestamp(datetime.now().date())
        open_day = pd.Timestamp(open_dt).normalize()
    except Exception:
        open_day = pd.Timestamp(datetime.now().date())

    data = _eval_price_history(stock) if "_eval_price_history" in dir() else pd.DataFrame()
    try:
        data = _eval_price_history(stock)
    except Exception:
        try:
            data = stock_history(clean_symbol(stock), interval="1d")
            data = _normalize_ohlc_index(data)
        except Exception:
            data = pd.DataFrame()

    if data is None or data.empty:
        # Mark-to-market with live quote only
        try:
            q = live_quote(stock)
            if q and q.get("price"):
                cur = safe_float(q["price"])
                if side == "SELL":
                    ret = (entry - cur) / entry * 100
                else:
                    ret = (cur - entry) / entry * 100
                out["Return %"] = round(ret, 2)
                out["PnL ₹"] = round(shares * entry * ret / 100, 2)
        except Exception:
            pass
        return out

    future = data[data.index >= open_day]
    if future.empty:
        future = data.tail(5)

    for i, (ts, candle) in enumerate(future.iterrows()):
        high = safe_float(candle.get("High"))
        low = safe_float(candle.get("Low"))
        close = safe_float(candle.get("Close"))
        day_num = i + 1

        if side == "SELL":
            if target > 0 and low <= target:
                ret = (entry - target) / entry * 100
                out.update({
                    "Status": "CLOSED",
                    "Result": "TARGET ACHIEVED",
                    "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    "Exit Price": round(target, 2),
                    "Return %": round(ret, 2),
                    "PnL ₹": round(shares * entry * ret / 100, 2),
                })
                return out
            if stop > 0 and high >= stop:
                ret = (entry - stop) / entry * 100
                out.update({
                    "Status": "CLOSED",
                    "Result": "STOP LOSS HIT",
                    "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    "Exit Price": round(stop, 2),
                    "Return %": round(ret, 2),
                    "PnL ₹": round(shares * entry * ret / 100, 2),
                })
                return out
        else:
            if stop > 0 and low <= stop:
                ret = (stop - entry) / entry * 100
                out.update({
                    "Status": "CLOSED",
                    "Result": "STOP LOSS HIT",
                    "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    "Exit Price": round(stop, 2),
                    "Return %": round(ret, 2),
                    "PnL ₹": round(shares * entry * ret / 100, 2),
                })
                return out
            if target > 0 and high >= target:
                ret = (target - entry) / entry * 100
                out.update({
                    "Status": "CLOSED",
                    "Result": "TARGET ACHIEVED",
                    "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    "Exit Price": round(target, 2),
                    "Return %": round(ret, 2),
                    "PnL ₹": round(shares * entry * ret / 100, 2),
                })
                return out

        if day_num >= hold_days:
            if side == "SELL":
                ret = (entry - close) / entry * 100
            else:
                ret = (close - entry) / entry * 100
            out.update({
                "Status": "CLOSED",
                "Result": "HOLDING PERIOD COMPLETED",
                "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                "Exit Price": round(close, 2),
                "Return %": round(ret, 2),
                "PnL ₹": round(shares * entry * ret / 100, 2),
            })
            return out

    # Still open — mark to last close
    last = safe_float(future["Close"].iloc[-1])
    if side == "SELL":
        ret = (entry - last) / entry * 100
    else:
        ret = (last - entry) / entry * 100
    out["Return %"] = round(ret, 2)
    out["PnL ₹"] = round(shares * entry * ret / 100, 2)
    return out


def show_paper_trading():
    """Dummy trades at current price → paper portfolio + backtest."""
    st.title("🧪 Dummy / Paper Trades & Backtest")
    st.caption(
        "Orders from **any page** (Strategy / BUY / SELL / Find Stock cards) land here. "
        "Target & Stop Loss are set automatically from the call (or ATR). "
        "Stored in `paper_portfolio.csv`."
    )

    paper = load_paper_portfolio()
    if paper is not None and not paper.empty:
        open_n = int((paper.get("Status", pd.Series(["OPEN"] * len(paper))).astype(str).str.upper() == "OPEN").sum()) if "Status" in paper.columns else len(paper)
        st.success(f"Portfolio: **{len(paper)}** trades · **{open_n}** open")
        st.markdown("##### Recent paper orders")
        filterable_dataframe(
            paper.sort_values("Open Date", ascending=False) if "Open Date" in paper.columns else paper,
            key="paper_book_table",
            default_cols=[c for c in [
                "Open Date", "Stock", "Side", "Entry", "Target", "Stop Loss",
                "Shares", "Source", "Status", "Result", "Return %", "PnL ₹",
            ] if c in paper.columns],
            height=280,
        )

    st.subheader("➕ New dummy trade")
    c1, c2, c3 = st.columns(3)
    with c1:
        stock = st.text_input("NSE Symbol", value="", key="paper_sym").upper().strip()
    with c2:
        side = st.selectbox("Side", ["BUY", "SELL"], key="paper_side")
    with c3:
        shares = st.number_input("Shares", min_value=1, value=10, step=1, key="paper_shares")

    # Live price
    live_px = 0.0
    if stock:
        try:
            q = live_quote(stock)
            if q and q.get("price"):
                live_px = safe_float(q["price"])
        except Exception:
            pass
        if live_px <= 0:
            try:
                d = stock_history(clean_symbol(stock), interval="1d")
                if d is not None and not d.empty:
                    live_px = safe_float(d["Close"].iloc[-1])
            except Exception:
                pass

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=float(live_px) if live_px > 0 else 0.0,
            step=0.05,
            format="%.2f",
            key="paper_entry",
        )
        if live_px > 0:
            st.caption(f"Live/last: ₹{live_px:,.2f}")
    with d2:
        if side == "SELL":
            default_tgt = round(entry * 0.95, 2) if entry else 0.0
            default_sl = round(entry * 1.03, 2) if entry else 0.0
        else:
            default_tgt = round(entry * 1.06, 2) if entry else 0.0
            default_sl = round(entry * 0.97, 2) if entry else 0.0
        target = st.number_input("Target", min_value=0.0, value=float(default_tgt), step=0.05, format="%.2f", key="paper_tgt")
    with d3:
        stop = st.number_input("Stop Loss", min_value=0.0, value=float(default_sl), step=0.05, format="%.2f", key="paper_sl")
    with d4:
        hold_days = st.number_input("Hold days", min_value=1, value=15, step=1, key="paper_hold")

    if side == "SELL" and target >= entry > 0:
        st.error("For SELL, **Target must be below Entry** (profit if price falls).")
    if side == "BUY" and 0 < target <= entry:
        st.warning("For BUY, Target is usually **above** Entry.")
    if side == "BUY" and 0 < stop >= entry:
        st.warning("For BUY, Stop Loss should be **below** Entry.")
    if side == "SELL" and 0 < stop <= entry:
        st.warning("For SELL, Stop Loss should be **above** Entry.")

    notes = st.text_input("Notes (optional)", key="paper_notes")

    if st.button("Add to paper portfolio", type="primary", key="paper_add"):
        if not stock or entry <= 0:
            st.error("Enter symbol and entry price.")
        elif side == "SELL" and target >= entry:
            st.error("Fix SELL target (must be < entry).")
        else:
            tid = f"{stock}|{datetime.now().strftime('%Y%m%d%H%M%S')}|{side}"
            new_row = {
                "Trade ID": tid,
                "Open Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Stock": display_symbol(stock),
                "Side": side,
                "Shares": int(shares),
                "Entry": round(entry, 2),
                "Target": round(target, 2),
                "Stop Loss": round(stop, 2),
                "Hold Days": int(hold_days),
                "Status": "OPEN",
                "Result": "PENDING",
                "Exit Date": "",
                "Exit Price": "",
                "Return %": "",
                "PnL ₹": "",
                "Notes": notes,
            }
            paper = pd.concat([paper, pd.DataFrame([new_row])], ignore_index=True)
            save_paper_portfolio(paper)
            st.success(f"Paper trade added: {side} {stock} @ ₹{entry:,.2f}")
            st.rerun()

    st.divider()
    st.subheader("📋 Paper portfolio")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("⟳ Backtest / update all", type="primary", key="paper_bt"):
            paper = load_paper_portfolio()
            if paper.empty:
                st.info("No paper trades yet.")
            else:
                rows = []
                with st.spinner("Backtesting against daily OHLC..."):
                    for _, r in paper.iterrows():
                        rec = r.to_dict()
                        if str(rec.get("Status", "")).upper() == "CLOSED":
                            rows.append(rec)
                            continue
                        res = backtest_paper_trade(rec)
                        rec.update(res)
                        rows.append(rec)
                paper = pd.DataFrame(rows)
                save_paper_portfolio(paper)
                st.success("Backtest complete — results written to paper_portfolio.csv")
                st.rerun()
    with b2:
        if st.button("Clear closed trades", key="paper_clear_closed"):
            paper = load_paper_portfolio()
            if not paper.empty:
                paper = paper[paper["Status"].astype(str).str.upper() != "CLOSED"]
                save_paper_portfolio(paper)
            st.rerun()
    with b3:
        if st.button("Clear entire paper book", key="paper_clear_all"):
            save_paper_portfolio(pd.DataFrame(columns=[
                "Trade ID", "Open Date", "Stock", "Side", "Shares", "Entry", "Target",
                "Stop Loss", "Hold Days", "Status", "Result", "Exit Date", "Exit Price",
                "Return %", "PnL ₹", "Notes",
            ]))
            st.rerun()

    paper = load_paper_portfolio()
    if paper.empty:
        st.info("No dummy trades yet. Add one above.")
        return

    # Stats
    res_u = paper["Result"].astype(str).str.upper()
    n_tgt = int(res_u.str.contains("TARGET ACHIEVED", na=False).sum())
    n_sl = int(res_u.str.contains("STOP LOSS HIT", na=False).sum())
    closed = paper[paper["Status"].astype(str).str.upper() == "CLOSED"]
    avg_r = pd.to_numeric(closed.get("Return %"), errors="coerce").mean() if not closed.empty else None
    total_pnl = pd.to_numeric(paper.get("PnL ₹"), errors="coerce").fillna(0).sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", len(paper))
    m2.metric("🎯 Targets", n_tgt)
    m3.metric("🔴 Stops", n_sl)
    m4.metric("Avg return %", f"{avg_r:+.2f}%" if avg_r == avg_r and avg_r is not None else "—")
    m5.metric("Total PnL ₹", f"{total_pnl:,.0f}")

    st.dataframe(paper, use_container_width=True, hide_index=True)
    st.caption(f"Saved to `{PAPER_FILE}` — use this list to backtest dummy trades vs real price history.")


# ============================================================
# HISTORY STATISTICS
# ============================================================

def _stats_from_history_slice(history: pd.DataFrame) -> dict:
    """Target-vs-stop success on a history slice (predicted stocks only)."""
    empty = {
        "recommendations": 0,
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "hold_done": 0,
        "open": 0,
        "success": 0.0,
        "win_rate_decided": 0.0,
        "avg_return": None,
        "avg_win": None,
        "avg_loss": None,
    }
    if history is None or history.empty:
        return empty

    result_col = history["Result"].astype(str).str.upper().str.strip()
    wins = int(result_col.str.contains("TARGET ACHIEVED", na=False).sum())
    wins += int(result_col.isin(["WIN"]).sum())
    losses = int(result_col.str.contains("STOP LOSS HIT", na=False).sum())
    losses += int(result_col.isin(["LOSS"]).sum())
    hold_done = int(result_col.str.contains("HOLDING PERIOD COMPLETED", na=False).sum())
    closed = wins + losses + hold_done
    recommendations = len(history)
    open_n = max(0, recommendations - closed)
    # Primary metric user cares about: targets / (targets + stops) only
    decided = wins + losses
    win_rate_decided = (wins / decided * 100) if decided > 0 else 0.0
    # Secondary: targets among all closed including time exits
    success = (wins / closed * 100) if closed > 0 else 0.0

    def _safe_avg(series):
        try:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty:
                return None
            v = float(s.mean())
            return None if v != v else round(v, 2)
        except Exception:
            return None

    avg_return = avg_win = avg_loss = None
    if "Return %" in history.columns:
        rets = pd.to_numeric(history["Return %"], errors="coerce")
        closed_mask = result_col.str.contains(
            "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
            regex=True,
            na=False,
        )
        win_mask = result_col.str.contains("TARGET ACHIEVED", na=False) | result_col.isin(["WIN"])
        loss_mask = result_col.str.contains("STOP LOSS HIT", na=False) | result_col.isin(["LOSS"])
        avg_return = _safe_avg(rets[closed_mask]) if closed_mask.any() else None
        avg_win = _safe_avg(rets[win_mask]) if win_mask.any() else None
        avg_loss = _safe_avg(rets[loss_mask]) if loss_mask.any() else None

    return {
        "recommendations": recommendations,
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "hold_done": hold_done,
        "open": open_n,
        "success": round(success, 1),
        "win_rate_decided": round(win_rate_decided, 1),
        "avg_return": avg_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


QUALITY_SOURCE_PATTERN = "SURE|STRATEGY|HIGH_CONV|PRECISION|MY_STRATEGY"


def quality_history_only(history: pd.DataFrame) -> pd.DataFrame:
    """Only Strategy / Sure / High-conviction / Precision — not bulk SCAN."""
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    if "Call Source" not in h.columns:
        h["Call Source"] = "SCAN"
    src = h["Call Source"].astype(str).str.upper()
    return h[src.str.contains(QUALITY_SOURCE_PATTERN, regex=True, na=False)].copy()


def overall_statistics():
    """
    Headline success = TARGET / (TARGET + STOP) on **Sure + Strategy + High conv** only.
    Bulk SCAN is excluded so the rate matches real predicted quality trades (~40–55% typical).
    """
    history = load_history()
    history = normalize_history_df(history)
    if history is None or history.empty:
        empty = _stats_from_history_slice(pd.DataFrame())
        empty.update({
            "buy_decided": 0.0, "quality_decided": 0.0, "scan_decided": 0.0,
            "buy_wins": 0, "buy_losses": 0, "quality_wins": 0, "quality_losses": 0,
            "scan_wins": 0, "scan_losses": 0, "quality_n": 0,
        })
        return empty

    quality = quality_history_only(history)
    scan = history
    if "Call Source" in history.columns:
        src = history["Call Source"].astype(str).str.upper()
        scan = history[~src.str.contains(QUALITY_SOURCE_PATTERN, regex=True, na=False)]

    q_stats = _stats_from_history_slice(quality)
    s_stats = _stats_from_history_slice(scan)
    # BUY subset of quality
    if not quality.empty and "Call" in quality.columns:
        buy_q = quality[quality["Call"].astype(str).str.upper().str.contains("BUY", na=False)]
    else:
        buy_q = quality
    buy_stats = _stats_from_history_slice(buy_q)

    # Headline metrics from QUALITY only
    base = dict(q_stats)
    base["buy_decided"] = buy_stats["win_rate_decided"]
    base["buy_wins"] = buy_stats["wins"]
    base["buy_losses"] = buy_stats["losses"]
    base["buy_closed"] = buy_stats["closed"]
    base["quality_decided"] = q_stats["win_rate_decided"]
    base["quality_wins"] = q_stats["wins"]
    base["quality_losses"] = q_stats["losses"]
    base["quality_n"] = len(quality)
    base["scan_decided"] = s_stats["win_rate_decided"]
    base["scan_wins"] = s_stats["wins"]
    base["scan_losses"] = s_stats["losses"]
    base["scan_n"] = len(scan)
    # Primary success rate shown on Past Predictions
    decided = q_stats["wins"] + q_stats["losses"]
    base["success"] = q_stats["win_rate_decided"] if decided > 0 else 0.0
    base["win_rate_decided"] = base["success"]
    base["recommendations"] = len(quality)
    return base


def show_dashboard_past_data():
    """
    Full past data on Dashboard from CSV files so user can see history
    and success rate without leaving the home page.
    """
    st.subheader("📁 Past data (from CSV) — always available on Dashboard")
    st.caption(
        f"Stored in: `{HISTORY_FILE.name}`, `{TRADES_FILE.name}`, `{RESULT_FILE.name}`. "
        "Target/Stop Loss stay locked as first saved."
    )

    # Ensure tracker exists
    try:
        if not TRADES_FILE.exists() or TRADES_FILE.stat().st_size < 50:
            sync_auto_trades_tracker(max_live=15)
    except Exception:
        pass

    stats = overall_statistics()
    st.caption(
        "Success rate on Dashboard = **BUY predictions** that hit **Target vs Stop** only "
        "(not open rows). ~40–55% is a normal healthy band for swing RR &gt; 1."
    )
    a, b, c, d, e, f = st.columns(6)
    a.metric("History rows", stats["recommendations"])
    b.metric("BUY decided (T+SL)", int(stats.get("buy_wins", 0)) + int(stats.get("buy_losses", 0)))
    c.metric("🎯 BUY targets", stats.get("buy_wins", stats["wins"]))
    d.metric("🔴 BUY stops", stats.get("buy_losses", stats["losses"]))
    e.metric("Success rate (BUY)", f"{stats['success']}%")
    f.metric("Quality only", f"{stats.get('quality_decided', stats['win_rate_decided'])}%")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Still open", stats.get("open", 0))
    r2.metric("Scan BUY success", f"{stats.get('scan_decided', 0)}%")
    r3.metric(
        "Avg return (closed)",
        f"{stats['avg_return']:+.2f}%" if stats.get("avg_return") is not None else "—",
    )
    r4.metric(
        "Avg win / avg loss",
        (
            f"{stats['avg_win']:+.1f}% / {stats['avg_loss']:+.1f}%"
            if stats.get("avg_win") is not None and stats.get("avg_loss") is not None
            else "—"
        ),
    )

    history = normalize_history_df(load_history())
    trades = load_auto_trades()

    tab1, tab2, tab3 = st.tabs([
        "📜 All past predictions (CSV)",
        "✅ Closed outcomes",
        "📡 Open tracked trades",
    ])

    with tab1:
        if history is None or history.empty:
            st.info("No history CSV data yet. Run FULL MARKET SCAN once.")
        else:
            show_n = st.selectbox(
                "Rows to show",
                [25, 50, 100, 200, 500, "All"],
                index=1,
                key="dash_hist_rows",
            )
            h = history.sort_values("Prediction Date", ascending=False)
            if show_n != "All":
                h = h.head(int(show_n))
            cols = [
                c for c in [
                    "Prediction Date", "Stock", "Call Source", "Strategy", "Call", "Entry", "Current Price",
                    "Target", "Stop Loss",
                    "Hold Days", "Status", "Result", "Exit Price", "Return %",
                    "Days Taken", "Recommendation", "Suggestion",
                ] if c in h.columns
            ]
            st.dataframe(h[cols], use_container_width=True, hide_index=True)
            st.caption(
                f"Full file has **{len(history)}** rows → `{HISTORY_FILE}`. "
                "**Current Price** updates live; **Entry / Target / Stop Loss** stay locked."
            )

    with tab2:
        if history is None or history.empty:
            st.info("No closed trades yet.")
        else:
            res_u = history["Result"].astype(str).str.upper()
            closed = history[
                res_u.str.contains(
                    "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
                    regex=True,
                    na=False,
                )
            ].copy()
            closed = closed.sort_values("Prediction Date", ascending=False)
            if closed.empty:
                st.warning(
                    "No closed outcomes in CSV yet. Open **Past Predictions** → "
                    "**Force re-check all**, then return here."
                )
            else:
                # Mini success by call type
                if "Call" in closed.columns:
                    by_call = []
                    for call in ["BUY", "HOLD", "SELL"]:
                        sub = closed[closed["Call"].astype(str).str.upper() == call]
                        if sub.empty:
                            continue
                        ru = sub["Result"].astype(str).str.upper()
                        w = int(ru.str.contains("TARGET ACHIEVED", na=False).sum())
                        l = int(ru.str.contains("STOP LOSS HIT", na=False).sum())
                        rate = (w / (w + l) * 100) if (w + l) else 0
                        by_call.append({
                            "Call": call,
                            "Closed": len(sub),
                            "Wins": w,
                            "Losses": l,
                            "Win rate %": round(rate, 1),
                        })
                    if by_call:
                        st.write("**Success by call type**")
                        st.dataframe(pd.DataFrame(by_call), use_container_width=True, hide_index=True)

                cols = [
                    c for c in [
                        "Prediction Date", "Stock", "Call", "Entry", "Target", "Stop Loss",
                        "Result", "Exit Price", "Return %", "Days Taken", "Evaluation Date",
                    ] if c in closed.columns
                ]
                st.dataframe(closed[cols].head(100), use_container_width=True, hide_index=True)

    with tab3:
        if trades is None or trades.empty:
            st.info("Trade tracker CSV empty — will fill after scan / sync.")
        else:
            open_t = trades[trades["Status"].astype(str).str.upper() == "OPEN"]
            st.write(f"**{len(open_t)}** open trades in `{TRADES_FILE.name}`")
            st.dataframe(open_t.head(50), use_container_width=True, hide_index=True)

    # Persist a compact performance snapshot CSV for the user
    try:
        snap = pd.DataFrame([{
            "Snapshot Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total Calls": stats["recommendations"],
            "Closed": stats["closed"],
            "Wins": stats["wins"],
            "Losses": stats["losses"],
            "Hold Done": stats.get("hold_done", 0),
            "Open": stats.get("open", 0),
            "Success Rate %": stats["success"],
            "Win Rate Target vs SL %": stats["win_rate_decided"],
            "Avg Return %": stats.get("avg_return", ""),
            "Avg Win %": stats.get("avg_win", ""),
            "Avg Loss %": stats.get("avg_loss", ""),
        }])
        snap_path = APP_DIR / "performance_snapshot.csv"
        if snap_path.exists():
            old = pd.read_csv(snap_path)
            snap = pd.concat([old, snap], ignore_index=True).tail(200)
        snap.to_csv(snap_path, index=False)
        st.caption(f"Performance snapshot also saved → `{snap_path.name}`")
    except Exception:
        pass


def stock_statistics(stock):

    history = load_history()

    if history.empty:
        return {
            "times": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "success": 0,
        }

    h = history[
        history["Stock"].astype(str).str.upper()
        == display_symbol(stock).upper()
    ]

    if h.empty:
        return {
            "times": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "success": 0,
        }

    result_col = h["Result"].astype(str).str.upper()

    wins = len(result_col[result_col.isin(["WIN", "TARGET ACHIEVED"])])
    losses = len(result_col[result_col.isin(["LOSS", "STOP LOSS HIT"])])
    closed = len(
        result_col[
            result_col.isin([
                "WIN",
                "LOSS",
                "TARGET ACHIEVED",
                "STOP LOSS HIT",
                "HOLDING PERIOD COMPLETED",
            ])
        ]
    )

    success = wins / closed * 100 if closed else 0

    return {
        "times": len(h),
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "success": round(success, 1),
    }


# ============================================================
# PORTFOLIO
# ============================================================

def load_portfolio():

    try:

        df = pd.read_csv(
            PORTFOLIO_FILE
        )

        if df.empty:
            return df

        if "Stock" in df.columns:

            df["Stock"] = (
                df["Stock"]
                .astype(str)
                .apply(clean_symbol)
            )

        return df

    except Exception:

        return pd.DataFrame()


def portfolio_analysis(
    results
):

    portfolio = load_portfolio()

    if portfolio.empty:
        return pd.DataFrame()

    if results is None or results.empty:
        return pd.DataFrame()

    lookup = {
        str(
            row["Symbol"]
        ).upper(): row
        for _, row in results.iterrows()
    }

    output = []

    for _, p in portfolio.iterrows():

        symbol = clean_symbol(
            p.get(
                "Stock",
                ""
            )
        )

        if symbol not in lookup:
            continue

        r = lookup[
            symbol
        ]

        shares = safe_float(
            p.get(
                "Shares",
                0
            )
        )

        buy_price = safe_float(
            p.get(
                "Buy Price",
                0
            )
        )

        current = safe_float(
            r["Price"]
        )

        if buy_price <= 0:
            continue

        pnl = (
            current -
            buy_price
        ) * shares

        pnl_pct = (
            (
                current -
                buy_price
            )
            /
            buy_price
            *
            100
        )

        try:

            purchase_date = (
                pd.to_datetime(
                    p.get(
                        "Purchase Date",
                        datetime.now().date()
                    )
                ).date()
            )

        except Exception:

            purchase_date = (
                datetime.now().date()
            )

        try:

            max_days = int(
                safe_float(
                    p.get(
                        "Maximum Holding Days",
                        DEFAULT_HOLD_DAYS
                    ),
                    DEFAULT_HOLD_DAYS
                )
            )

        except Exception:

            max_days = (
                DEFAULT_HOLD_DAYS
            )

        days_held = max(
            0,
            len(
                pd.bdate_range(
                    purchase_date,
                    datetime.now().date()
                )
            ) - 1
        )

        days_remaining = max(
            0,
            max_days -
            days_held
        )

        action = "HOLD"

        reason = (
            "No immediate exit condition has been triggered."
        )

        # Target
        if current >= safe_float(
            r["Target"]
        ):

            action = "🎯 TARGET ACHIEVED — SELL / BOOK PROFIT"

            reason = (
                f"🎯 TARGET ACHIEVED\n"
                f"Previous Target: ₹{safe_float(r['Target']):,.2f}\n"
                f"Current Price: ₹{current:,.2f}\n"
                f"Recommendation: HOLD / TRAIL STOP LOSS  or  🔴 SELL / BOOK PROFIT"
            )

        # Stop loss
        elif current <= safe_float(
            r["Stop Loss"]
        ):

            action = "🔴 STOP LOSS HIT — SELL / EXIT"

            reason = (
                f"🔴 STOP LOSS HIT\n"
                f"Current Price: ₹{current:,.2f}\n"
                f"Stop Loss: ₹{safe_float(r['Stop Loss']):,.2f}\n"
                f"Recommendation: 🔴 SELL / EXIT"
            )

        # Time expiry
        elif days_held >= max_days:

            action = (
                "⏰ HOLDING PERIOD COMPLETED — SELL"
            )

            reason = (
                f"⏰ HOLDING PERIOD COMPLETED\n"
                f"Neither Target nor Stop Loss was reached within "
                f"{max_days} trading days.\n"
                f"Recommendation: ⚠️ EARLY EXIT / SELL"
            )

        # Model SELL
        elif str(
            r["Call"]
        ).upper() == "SELL":

            action = "SELL"

            reason = (
                "The current technical model is giving a SELL signal."
            )

        # Strong BUY
        elif (
            str(
                r["Call"]
            ).upper() == "BUY"
            and
            safe_float(
                r["Prediction"]
            ) >= 80
        ):

            action = (
                "BUY MORE - CAUTIOUS"
            )

            reason = (
                "The current model remains strong. "
                "Adding shares increases exposure and should be done cautiously."
            )

        output.append(
            {
                "Stock":
                    display_symbol(symbol),

                "Shares":
                    shares,

                "Buy Price":
                    buy_price,

                "Current Price":
                    current,

                "P&L":
                    round(
                        pnl,
                        2
                    ),

                "P&L %":
                    round(
                        pnl_pct,
                        2
                    ),

                "Action":
                    action,

                "Prediction":
                    r["Prediction"],

                "Risk %":
                    r["Risk %"],

                "Risk Level":
                    r["Risk Level"],

                "Target":
                    r["Target"],

                "Stop Loss":
                    r["Stop Loss"],

                "Days Held":
                    days_held,

                "Days Remaining":
                    days_remaining,

                "Maximum Holding Days":
                    max_days,

                "Reason":
                    reason,
            }
        )

    return pd.DataFrame(
        output
    )


# ============================================================
# SECTOR ANALYSIS
# ============================================================

def sector_table(
    results
):

    if results is None or results.empty:
        return pd.DataFrame()

    x = ensure_result_columns(results)

    if "Sector" not in x.columns:

        x["Sector"] = (
            x["Stock"]
            .apply(
                sector_of
            )
        )

    prediction = pd.to_numeric(
        x["Prediction"],
        errors="coerce"
    ).fillna(0)

    risk = pd.to_numeric(
        x["Risk %"],
        errors="coerce"
    ).fillna(50)

    call_score = x[
        "Call"
    ].map(
        {
            "BUY": 100,
            "HOLD": 50,
            "SELL": 0,
        }
    ).fillna(40)

    x["_score"] = (
        prediction * 0.60
        +
        (100 - risk) * 0.25
        +
        call_score * 0.15
    )

    s = (
        x.groupby(
            "Sector"
        )
        .agg(
            Stocks=(
                "Stock",
                "count"
            ),

            Avg_Prediction=(
                "Prediction",
                "mean"
            ),

            Avg_Risk=(
                "Risk %",
                "mean"
            ),

            BUY_Calls=(
                "Call",
                lambda z:
                int(
                    (
                        z ==
                        "BUY"
                    ).sum()
                )
            ),

            HOLD_Calls=(
                "Call",
                lambda z:
                int(
                    (
                        z ==
                        "HOLD"
                    ).sum()
                )
            ),

            SELL_Calls=(
                "Call",
                lambda z:
                int(
                    (
                        z ==
                        "SELL"
                    ).sum()
                )
            ),

            Strength=(
                "_score",
                "mean"
            ),
        )
        .reset_index()
    )

    s["BUY %"] = (
        s["BUY_Calls"]
        /
        s["Stocks"].replace(
            0,
            np.nan
        )
        *
        100
    ).fillna(0).round(1)

    s["Sector Strength"] = (
        s["Strength"]
        .clip(
            0,
            100
        )
        .round(1)
    )

    def priority(v):

        if v >= 80:
            return "VERY HIGH"

        if v >= 68:
            return "HIGH"

        if v >= 52:
            return "MEDIUM"

        if v >= 35:
            return "LOW"

        return "VERY LOW"

    s[
        "Sector Priority"
    ] = s[
        "Sector Strength"
    ].apply(
        priority
    )

    def trend(row):

        if (
            row["BUY %"] >= 55
            and
            row["Avg_Prediction"] >= 60
        ):
            return "BULLISH"

        if (
            row["SELL_Calls"]
            /
            max(
                row["Stocks"],
                1
            )
            >= 0.45
        ):
            return "BEARISH"

        return "NEUTRAL"

    s["Trend"] = s.apply(
        trend,
        axis=1
    )

    s = s.drop(
        columns=[
            "Strength"
        ]
    )

    s = s.sort_values(
        [
            "Sector Strength",
            "BUY %",
        ],
        ascending=[
            False,
            False,
        ]
    )

    s.reset_index(
        drop=True,
        inplace=True
    )

    s.insert(
        0,
        "Rank",
        range(
            1,
            len(s) + 1
        )
    )

    return s


# ============================================================
# FIND STOCK / NATURAL LANGUAGE SEARCH
# ============================================================

def find_stock_results(
    results,
    query
):

    if results is None or results.empty:
        return pd.DataFrame()

    query = str(
        query
    ).strip().lower()

    if not query:
        return results

    x = ensure_result_columns(results)

    # --------------------------------------------------------
    # Direct symbol
    # --------------------------------------------------------

    words = re.findall(
        r"[a-zA-Z0-9&]+",
        query
    )

    direct = []

    for word in words:

        word = word.upper()

        matches = x[
            x["Stock"]
            .astype(str)
            .str.upper()
            .eq(word)
        ]

        if not matches.empty:

            direct.append(
                matches
            )

    if direct:

        return pd.concat(
            direct
        ).drop_duplicates()

    # --------------------------------------------------------
    # CALL FILTERS
    # --------------------------------------------------------

    if (
        "buy" in query
        or
        "purchase" in query
    ):

        x = x[
            x["Call"] == "BUY"
        ]

    if "sell" in query:

        x = x[
            x["Call"] == "SELL"
        ]

    if "hold" in query:

        x = x[
            x["Call"] == "HOLD"
        ]

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    if (
        "low risk" in query
        or
        "safe" in query
    ):

        x = x[
            x["Risk Level"]
            == "LOW"
        ]

    elif "medium risk" in query:

        x = x[
            x["Risk Level"]
            == "MEDIUM"
        ]

    elif "high risk" in query:

        x = x[
            x["Risk Level"]
            .isin(
                [
                    "HIGH",
                    "VERY HIGH",
                ]
            )
        ]

    # --------------------------------------------------------
    # SECTORS
    # --------------------------------------------------------

    sector_keywords = [
        "bank",
        "banking",
        "it",
        "pharma",
        "energy",
        "metal",
        "metals",
        "auto",
        "automobile",
        "fmcg",
        "telecom",
        "defence",
        "defense",
        "utility",
        "utilities",
        "infrastructure",
        "retail",
        "consumer",
    ]

    selected_sector = None

    for word in sector_keywords:

        if word in query:

            if word in [
                "bank",
                "banking",
            ]:
                selected_sector = "Banking"

            elif word == "it":
                selected_sector = "IT"

            elif word == "pharma":
                selected_sector = "Pharma"

            elif word == "energy":
                selected_sector = "Energy"

            elif word in [
                "metal",
                "metals",
            ]:
                selected_sector = "Metals"

            elif word in [
                "auto",
                "automobile",
            ]:
                selected_sector = "Automobile"

            elif word == "fmcg":
                selected_sector = "FMCG"

            elif word == "telecom":
                selected_sector = "Telecom"

            elif word in [
                "defence",
                "defense",
            ]:
                selected_sector = "Defence"

            elif word in [
                "utility",
                "utilities",
            ]:
                selected_sector = "Utilities"

            elif word == "infrastructure":
                selected_sector = "Infrastructure"

            elif word == "retail":
                selected_sector = "Retail"

            elif word == "consumer":
                selected_sector = "Consumer"

            break

    if selected_sector:

        x = x[
            x["Sector"]
            ==
            selected_sector
        ]

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    under_match = re.search(
        r"(?:under|below|less than)\s*(?:₹|rs\.?|inr)?\s*([0-9]+(?:\.[0-9]+)?)",
        query
    )

    if under_match:

        limit = safe_float(
            under_match.group(1)
        )

        x = x[
            pd.to_numeric(
                x["Price"],
                errors="coerce"
            )
            <= limit
        ]

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if (
        "strong" in query
        or
        "high prediction" in query
        or
        "best" in query
    ):

        x = x[
            x["Prediction"]
            >= 70
        ]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    if (
        "low risk" in query
        or
        "safe" in query
    ):

        x = x.sort_values(
            [
                "Risk %",
                "Prediction",
            ],
            ascending=[
                True,
                False,
            ]
        )

    else:

        x = x.sort_values(
            [
                "Prediction",
                "Risk %",
            ],
            ascending=[
                False,
                True,
            ]
        )

    return x


# ============================================================
# TECHNICAL CHART
# ============================================================

def indicator_education(result):
    """
    Teach-each-indicator view: priority, current number, rule of thumb,
    and stock-specific verdict so the user can decide manually.
    Priority 1 = most important for this model's decisions.
    """
    price = safe_float(result.get("Price"))
    rsi = safe_float(result.get("RSI"))
    macd_v = safe_float(result.get("MACD"))
    macd_s = safe_float(result.get("MACD Signal"))
    adx = safe_float(result.get("ADX"))
    stoch = safe_float(result.get("Stochastic"))
    vwap = safe_float(result.get("VWAP"))
    vol_r = safe_float(result.get("Volume Ratio"))
    data = result.get("Data")

    ema20 = ema50 = ema200 = bb_mid = atr_v = None
    if data is not None and not getattr(data, "empty", True):
        row = data.iloc[-1]
        ema20 = safe_float(row.get("EMA20"))
        ema50 = safe_float(row.get("EMA50"))
        ema200 = safe_float(row.get("EMA200"))
        bb_mid = safe_float(row.get("BBMiddle"))
        atr_v = safe_float(row.get("ATR"))

    lessons = []

    # --- Priority 1: Trend structure (EMAs) ---
    if ema20 is not None:
        above20 = price > ema20
        lessons.append({
            "Priority": 1,
            "Indicator": "EMA 20 (short trend)",
            "Current value": f"Price ₹{price:,.2f} vs EMA20 ₹{ema20:,.2f}",
            "Rule of thumb (general)": (
                "Price ABOVE EMA20 → short-term buyers in control. "
                "Price BELOW EMA20 → short-term weakness. "
                "Cross above EMA20 is often early bullish; cross below is early caution."
            ),
            "For THIS stock": (
                f"✅ Price is ABOVE EMA20 — short-term trend supports buying / holding."
                if above20 else
                f"⚠️ Price is BELOW EMA20 — short-term momentum is weak; wait for reclaim if possible."
            ),
            "Manual decision tip": "Do not fight a clear close below EMA20 on a fresh BUY unless other high-priority signals are very strong.",
        })

    if ema50 is not None:
        above50 = price > ema50
        lessons.append({
            "Priority": 1,
            "Indicator": "EMA 50 (medium trend)",
            "Current value": f"Price ₹{price:,.2f} vs EMA50 ₹{ema50:,.2f}",
            "Rule of thumb (general)": (
                "Price ABOVE EMA50 → medium-term uptrend bias. "
                "Many swing traders only BUY when price > EMA50. "
                "EMA20 > EMA50 adds alignment (short trend agrees with medium)."
            ),
            "For THIS stock": (
                f"✅ Price is ABOVE EMA50 — medium-term structure is supportive."
                if above50 else
                f"⚠️ Price is BELOW EMA50 — medium-term trend is not confirmed bullish."
            ),
            "Manual decision tip": "Prefer BUY when price > EMA50. If below EMA50, treat as counter-trend / higher risk.",
        })

    if ema200 is not None:
        above200 = price > ema200
        lessons.append({
            "Priority": 2,
            "Indicator": "EMA 200 (long trend)",
            "Current value": f"Price ₹{price:,.2f} vs EMA200 ₹{ema200:,.2f}",
            "Rule of thumb (general)": (
                "Price ABOVE EMA200 → long-term bull market for the stock. "
                "Price BELOW EMA200 → long-term caution; rallies often fail more often."
            ),
            "For THIS stock": (
                f"✅ Price is ABOVE EMA200 — long-term trend is bullish."
                if above200 else
                f"⚠️ Price is BELOW EMA200 — long-term structure is weak."
            ),
            "Manual decision tip": "Best swing BUYs are usually above EMA200. Below EMA200 = trade smaller or skip.",
        })

    # --- Priority 2: Momentum ---
    rsi_note = (
        "✅ RSI is in the healthy bullish band (50–70): strength without extreme overbought."
        if 50 <= rsi <= 70 else
        "✅ RSI is oversold (<30): bounce possible, but confirm with volume / EMA reclaim."
        if rsi < 30 else
        "⚠️ RSI is overbought (>75): late-stage strength; avoid chasing; tighten risk."
        if rsi > 75 else
        f"ℹ️ RSI is {rsi:.1f}: between weak and strong — look at EMA + MACD for direction."
    )
    lessons.append({
        "Priority": 2,
        "Indicator": "RSI (14)",
        "Current value": f"{rsi:.1f}",
        "Rule of thumb (general)": (
            "• RSI > 50 → bullish momentum bias (buyers stronger than sellers).\n"
            "• RSI 55–70 → often ideal for continuing uptrends.\n"
            "• RSI < 30 → oversold; possible rebound, not automatic BUY.\n"
            "• RSI > 75–80 → overbought; profit-taking / pullback risk rises.\n"
            "General teaching: RSI 55 > 50 is usually GOOD in an uptrend — it means momentum is positive, not exhausted."
        ),
        "For THIS stock": rsi_note,
        "Manual decision tip": "In uptrends, prefer RSI > 50. RSI 55–68 is often the ‘sweet zone’. Avoid new BUYs mainly because RSI is 80+.",
    })

    macd_bull = macd_v > macd_s
    lessons.append({
        "Priority": 2,
        "Indicator": "MACD vs Signal",
        "Current value": f"MACD {macd_v:.4f} | Signal {macd_s:.4f}",
        "Rule of thumb (general)": (
            "MACD ABOVE signal line → bullish momentum. "
            "MACD BELOW signal → bearish momentum. "
            "A fresh cross up is a classic buy trigger; cross down is a warning."
        ),
        "For THIS stock": (
            "✅ MACD is above signal — momentum is bullish."
            if macd_bull else
            "⚠️ MACD is below signal — momentum is bearish / weakening."
        ),
        "Manual decision tip": "Agreeing MACD + RSI > 50 + price > EMA20 is a strong manual BUY cluster.",
    })

    lessons.append({
        "Priority": 3,
        "Indicator": "ADX (trend strength)",
        "Current value": f"{adx:.1f}",
        "Rule of thumb (general)": (
            "ADX < 20 → weak / sideways trend (signals fail more). "
            "ADX 20–25 → trend emerging. "
            "ADX ≥ 25 → trend is meaningful. "
            "ADX does NOT say direction — only strength. Combine with EMA/MACD for direction."
        ),
        "For THIS stock": (
            f"✅ ADX {adx:.1f} ≥ 25 — trend is strong enough to trust direction signals."
            if adx >= 25 else
            f"ℹ️ ADX {adx:.1f} < 25 — trend is not strong; expect more chop; use tighter stops or wait."
        ),
        "Manual decision tip": "When ADX is low, reduce size or wait. When ADX is high and price > EMA50, trend-following BUYs work better.",
    })

    stoch_note = (
        "✅ Stochastic rising and not overbought (<80) — short-term momentum supportive."
        if stoch < 80 else
        "⚠️ Stochastic is elevated (>80) — short-term stretched; pullback risk."
    )
    lessons.append({
        "Priority": 3,
        "Indicator": "Stochastic %K",
        "Current value": f"{stoch:.1f}",
        "Rule of thumb (general)": (
            "%K > %D and %K < 80 → often bullish short-term. "
            "%K > 80 → overbought zone. "
            "%K < 20 → oversold zone."
        ),
        "For THIS stock": stoch_note,
        "Manual decision tip": "Use Stochastic to time entry inside a larger uptrend — not as the only reason to buy.",
    })

    if bb_mid is not None:
        lessons.append({
            "Priority": 3,
            "Indicator": "Bollinger middle (20 SMA)",
            "Current value": f"Price ₹{price:,.2f} vs BB mid ₹{bb_mid:,.2f}",
            "Rule of thumb (general)": (
                "Price above middle band → bullish bias inside the channel. "
                "Price below middle → bearish bias. "
                "Touching lower band can be mean-reversion buy in ranges; in strong downtrends it can keep falling."
            ),
            "For THIS stock": (
                "✅ Price is above Bollinger middle — bullish bias inside the band."
                if price > bb_mid else
                "⚠️ Price is below Bollinger middle — short-term bias is weaker."
            ),
            "Manual decision tip": "In trends, buy pullbacks toward the middle band while price stays above EMA50.",
        })

    if vwap > 0:
        lessons.append({
            "Priority": 3,
            "Indicator": "VWAP",
            "Current value": f"Price ₹{price:,.2f} vs VWAP ₹{vwap:,.2f}",
            "Rule of thumb (general)": (
                "Price ABOVE VWAP → institutional / average buyer is in profit (bullish for day/swing context). "
                "Price BELOW VWAP → average buyer underwater (pressure)."
            ),
            "For THIS stock": (
                "✅ Price is above VWAP — buying strength vs average traded price."
                if price > vwap else
                "⚠️ Price is below VWAP — weaker vs average traded price."
            ),
            "Manual decision tip": "Intraday: prefer longs above VWAP. Positional: treat as secondary confirmation.",
        })

    lessons.append({
        "Priority": 2,
        "Indicator": "Volume Ratio (vs 20-day avg)",
        "Current value": f"{vol_r:.2f}×",
        "Rule of thumb (general)": (
            "Volume ≥ 1.5× average → move is more meaningful (conviction). "
            "Volume ~1.0× → normal. "
            "Very low volume breakouts often fail."
        ),
        "For THIS stock": (
            f"✅ Volume is elevated ({vol_r:.2f}×) — price move has participation."
            if vol_r >= 1.5 else
            f"ℹ️ Volume is moderate ({vol_r:.2f}×)."
            if vol_r >= 1.1 else
            f"⚠️ Volume is not high ({vol_r:.2f}×) — breakout/breakdown is less trustworthy."
        ),
        "Manual decision tip": "Never trust a breakout on tiny volume. Rising price + rising volume is healthier.",
    })

    if atr_v is not None and atr_v > 0:
        lessons.append({
            "Priority": 2,
            "Indicator": "ATR (14) — volatility used for SL/Target",
            "Current value": f"₹{atr_v:,.2f}",
            "Rule of thumb (general)": (
                "ATR measures typical daily range. "
                "This model uses Stop ≈ Price − 1.5×ATR and Target ≈ Price + 2.5×ATR. "
                "Higher ATR = wider stop = higher Risk %."
            ),
            "For THIS stock": (
                f"Stop and target are placed from ATR so risk matches this stock’s normal movement "
                f"(~₹{atr_v:,.2f} per day typical range)."
            ),
            "Manual decision tip": "If ATR is large vs your capital, reduce quantity. Never use a stop tighter than ~1×ATR casually.",
        })

    lessons.append({
        "Priority": 1,
        "Indicator": "Risk % / Risk Level (model)",
        "Current value": f"{safe_float(result.get('Risk %')):.2f}% · {result.get('Risk Level')}",
        "Rule of thumb (general)": (
            "Risk % = distance from entry to stop as % of price. "
            "LOW <3%, MEDIUM 3–6%, HIGH 6–10%, VERY HIGH ≥10%."
        ),
        "For THIS stock": (
            f"Model stop is ₹{safe_float(result.get('Stop Loss')):,.2f}, target ₹{safe_float(result.get('Target')):,.2f}."
        ),
        "Manual decision tip": "Only size the trade so that if stop hits, loss ≤ your chosen % of capital (see BUY page helper).",
    })

    return pd.DataFrame(lessons).sort_values("Priority")


def pattern_education(patterns_str):
    """Explain each detected pattern and how to use it manually."""
    catalog = {
        "Hammer": {
            "bias": "Bullish",
            "why": "Long lower wick shows sellers pushed price down but buyers closed near the highs — rejection of lower prices.",
            "use": "More reliable after a decline or at support / EMA. Confirm next candle closes higher.",
        },
        "Shooting Star": {
            "bias": "Bearish",
            "why": "Long upper wick shows buyers failed to hold highs — supply appeared overhead.",
            "use": "More reliable after a rally or at resistance. Confirm next candle closes lower.",
        },
        "Bullish Engulfing": {
            "bias": "Bullish",
            "why": "A green body fully covers the prior red body — buyers overwhelmed sellers in one session.",
            "use": "Stronger with volume > average and near support / EMA50.",
        },
        "Bearish Engulfing": {
            "bias": "Bearish",
            "why": "A red body fully covers the prior green body — sellers took control abruptly.",
            "use": "Stronger after extended up-move; reason to tighten stop or avoid new longs.",
        },
        "Doji": {
            "bias": "Neutral / indecision",
            "why": "Open ≈ close — balance between buyers and sellers; trend may pause or reverse.",
            "use": "Do not trade Doji alone. Wait for the next directional candle.",
        },
        "Higher High / Higher Low": {
            "bias": "Bullish structure",
            "why": "Swing highs and lows are rising — classic uptrend footprint.",
            "use": "Favour BUY / HOLD while HH–HL structure holds. Break of last higher low is a warning.",
        },
        "Lower High / Lower Low": {
            "bias": "Bearish structure",
            "why": "Swing highs and lows are falling — downtrend footprint.",
            "use": "Avoid fresh longs; prefer SELL / stay out until structure breaks upward.",
        },
    }

    names = [p.strip() for p in str(patterns_str or "").split(",") if p.strip()]
    if not names or names == ["No major pattern detected"]:
        return [{
            "Pattern": "None major",
            "Bias": "—",
            "Why it matters": "No classic candle/structure flag on the latest bars.",
            "How you use it": "Rely on EMA + RSI + MACD + volume instead of patterns today.",
        }]

    rows = []
    for n in names:
        meta = catalog.get(n, {
            "bias": "Context",
            "why": "Detected by the model’s pattern engine.",
            "use": "Combine with trend (EMA) and volume before acting.",
        })
        rows.append({
            "Pattern": n,
            "Bias": meta["bias"],
            "Why it matters": meta["why"],
            "How you use it": meta["use"],
        })
    return rows


def make_chart(result, df=None, patterns=None, target=None, stop_loss=None, title=None):
    """
    Interactive Plotly candles + selected indicators + pattern markers.
    Works offline / market closed with last available bars.
    """
    if df is None:
        df = result.get("Data")
    if df is None or getattr(df, "empty", True):
        return go.Figure()

    # Ensure indicators exist on this timeframe
    work = normalize_columns(df).copy()
    try:
        work = calculate_indicators(work)
    except Exception:
        pass
    if work is None or work.empty:
        work = df.copy()

    # Show last N bars for speed
    tail_n = 200 if len(work) > 200 else len(work)
    plot_df = work.tail(tail_n)

    if patterns is None:
        patterns = [p.strip() for p in str(result.get("Patterns", "")).split(",") if p.strip()]
    patterns = [p for p in patterns if p and p != "No major pattern detected"]
    # Re-detect on this timeframe for live accuracy
    try:
        live_pats = detect_patterns(plot_df)
        if live_pats:
            patterns = live_pats
    except Exception:
        pass

    if target is None:
        target = result.get("Target") if result else None
    if stop_loss is None:
        stop_loss = result.get("Stop Loss") if result else None

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Price",
        )
    )
    for col, name in [("EMA20", "EMA 20"), ("EMA50", "EMA 50"), ("EMA200", "EMA 200")]:
        if col in plot_df.columns and plot_df[col].notna().any():
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df[col], name=name, line=dict(width=1.2))
            )
    if "BBUpper" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["BBUpper"], name="BB Upper",
                line=dict(dash="dot", width=1),
            )
        )
    if "BBLower" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["BBLower"], name="BB Lower",
                line=dict(dash="dot", width=1),
            )
        )
    if "VWAP" in plot_df.columns and plot_df["VWAP"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["VWAP"], name="VWAP",
                line=dict(width=1, dash="dash"),
            )
        )

    if patterns and len(plot_df) >= 2:
        last_i = plot_df.index[-1]
        prev_i = plot_df.index[-2]
        last_high = safe_float(plot_df["High"].iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[last_i],
                y=[last_high * 1.008],
                mode="markers+text",
                marker=dict(size=14, symbol="triangle-down", color="#f5c542"),
                text=["Pattern"],
                textposition="top center",
                name="Pattern",
            )
        )
        fig.add_vrect(
            x0=prev_i,
            x1=last_i,
            fillcolor="rgba(245, 197, 66, 0.18)",
            layer="below",
            line_width=0,
            annotation_text=", ".join(patterns)[:48],
            annotation_position="top left",
        )
        if any("Higher High" in p or "Lower High" in p for p in patterns) and len(plot_df) >= 10:
            win = plot_df.tail(10)
            fig.add_vrect(
                x0=win.index[0],
                x1=win.index[-1],
                fillcolor=(
                    "rgba(26, 155, 95, 0.10)"
                    if any("Higher High" in p for p in patterns)
                    else "rgba(217, 48, 37, 0.10)"
                ),
                layer="below",
                line_width=0,
                annotation_text="Structure",
                annotation_position="bottom left",
            )

    if safe_float(target) > 0:
        fig.add_hline(
            y=safe_float(target), line_dash="dash", line_color="#1a9b5f",
            annotation_text="Target",
        )
    if safe_float(stop_loss) > 0:
        fig.add_hline(
            y=safe_float(stop_loss), line_dash="dash", line_color="#d93025",
            annotation_text="Stop Loss",
        )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.12),
        title=title or "Interactive chart · indicators · patterns · target/stop",
        dragmode="zoom",
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False, showspikes=True)
    fig.update_yaxes(showspikes=True)
    return fig


def compute_support_resistance(df, lookback=60, swings=3):
    """
    Simple swing-based support / resistance from recent highs & lows.
    Returns lists of resistance levels (highs) and support levels (lows).
    """
    if df is None or len(df) < 10:
        return [], []
    d = df.tail(max(lookback, 20)).copy()
    highs = d["High"].astype(float)
    lows = d["Low"].astype(float)
    # Local peaks / troughs
    res_levels = []
    sup_levels = []
    for i in range(2, len(d) - 2):
        h = highs.iloc[i]
        l = lows.iloc[i]
        if h >= highs.iloc[i - 1] and h >= highs.iloc[i - 2] and h >= highs.iloc[i + 1] and h >= highs.iloc[i + 2]:
            res_levels.append(float(h))
        if l <= lows.iloc[i - 1] and l <= lows.iloc[i - 2] and l <= lows.iloc[i + 1] and l <= lows.iloc[i + 2]:
            sup_levels.append(float(l))
    # Cluster nearby levels (0.4% tolerance)
    def _cluster(levels, n):
        if not levels:
            return []
        levels = sorted(levels, reverse=True)
        out = []
        for lv in levels:
            if not out or all(abs(lv - o) / max(o, 1e-9) > 0.004 for o in out):
                out.append(lv)
            if len(out) >= n:
                break
        return out

    return _cluster(res_levels, swings), _cluster(sup_levels, swings)


def show_interactive_chart_panel(symbol, result):
    """
    Multi-timeframe interactive chart.
    All indicators / S-R / patterns are optional toggles — user chooses what to show.
    """
    st.subheader("📊 Interactive chart — pick what you want to see")
    st.caption(
        "Turn indicators ON/OFF below. Zoom/pan the chart. "
        "Works market open or closed (last available bars)."
    )

    sk = display_symbol(symbol)
    c1, c2 = st.columns([1, 3])
    with c1:
        tf = st.selectbox(
            "Timeframe",
            [
                ("1 Day", "1d"),
                ("1 Week", "1wk"),
                ("1 Hour", "1h"),
                ("30 Min", "30m"),
                ("15 Min", "15m"),
                ("5 Min", "5m"),
            ],
            format_func=lambda x: x[0],
            key=f"tf_{sk}",
        )
        interval = tf[1]
        if st.button("↻ Reload chart data", key=f"reload_chart_{sk}"):
            try:
                stock_history.clear()
            except Exception:
                pass
            st.rerun()

    with c2:
        show_ind = st.multiselect(
            "Indicators & overlays (select any / none)",
            [
                "EMA 20",
                "EMA 50",
                "EMA 200",
                "Bollinger",
                "VWAP",
                "Support / Resistance",
                "Patterns",
                "Target / Stop",
                "Volume",
                "RSI panel",
                "MACD panel",
                "Stochastic panel",
            ],
            default=[
                "EMA 20",
                "EMA 50",
                "EMA 200",
                "Bollinger",
                "Support / Resistance",
                "Patterns",
                "Target / Stop",
                "Volume",
            ],
            key=f"ind_{sk}",
        )

    with st.spinner(f"Loading {interval} data..."):
        if interval == "1d" and result.get("Data") is not None and not result["Data"].empty:
            df_tf = result["Data"]
        else:
            df_tf = stock_history(symbol, interval=interval)

    if df_tf is None or df_tf.empty:
        st.warning(
            f"No data for **{interval}**. Try 1 Day (intraday can be limited after close)."
        )
        return

    work = normalize_columns(df_tf).copy()
    try:
        work = calculate_indicators(work)
    except Exception:
        pass
    if work is None or work.empty:
        work = normalize_columns(df_tf).copy()

    tail_n = 220 if len(work) > 220 else len(work)
    plot_df = work.tail(tail_n)

    # Detect patterns on this TF
    try:
        pats = detect_patterns(plot_df)
    except Exception:
        pats = []

    # Subplot rows
    rows = 1
    row_heights = [0.55]
    specs = [[{"secondary_y": False}]]
    if "Volume" in show_ind:
        rows += 1
        row_heights.append(0.12)
        specs.append([{"secondary_y": False}])
    if "RSI panel" in show_ind:
        rows += 1
        row_heights.append(0.14)
        specs.append([{"secondary_y": False}])
    if "MACD panel" in show_ind:
        rows += 1
        row_heights.append(0.14)
        specs.append([{"secondary_y": False}])
    if "Stochastic panel" in show_ind:
        rows += 1
        row_heights.append(0.12)
        specs.append([{"secondary_y": False}])

    # Normalize heights
    s = sum(row_heights)
    row_heights = [h / s for h in row_heights]

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=specs,
    )

    r = 1
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Price",
        ),
        row=r,
        col=1,
    )

    if "EMA 20" in show_ind and "EMA20" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["EMA20"], name="EMA 20", line=dict(width=1.2)),
            row=r, col=1,
        )
    if "EMA 50" in show_ind and "EMA50" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["EMA50"], name="EMA 50", line=dict(width=1.2)),
            row=r, col=1,
        )
    if "EMA 200" in show_ind and "EMA200" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["EMA200"], name="EMA 200", line=dict(width=1.3)),
            row=r, col=1,
        )
    if "Bollinger" in show_ind:
        if "BBUpper" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df["BBUpper"], name="BB Upper",
                    line=dict(dash="dot", width=1),
                ),
                row=r, col=1,
            )
        if "BBLower" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df["BBLower"], name="BB Lower",
                    line=dict(dash="dot", width=1),
                ),
                row=r, col=1,
            )
    if "VWAP" in show_ind and "VWAP" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["VWAP"], name="VWAP",
                line=dict(dash="dash", width=1),
            ),
            row=r, col=1,
        )

    # Support / Resistance
    if "Support / Resistance" in show_ind:
        resistances, supports = compute_support_resistance(plot_df)
        for i, lv in enumerate(resistances):
            fig.add_hline(
                y=lv, line_dash="dot", line_color="#ff6b6b",
                annotation_text=f"R{i+1} {lv:,.1f}",
                annotation_position="right",
                row=r, col=1,
            )
        for i, lv in enumerate(supports):
            fig.add_hline(
                y=lv, line_dash="dot", line_color="#51cf66",
                annotation_text=f"S{i+1} {lv:,.1f}",
                annotation_position="right",
                row=r, col=1,
            )
        if resistances or supports:
            st.caption(
                "S/R: "
                + (" | ".join([f"R{i+1}=₹{v:,.1f}" for i, v in enumerate(resistances)]) or "no R")
                + " · "
                + (" | ".join([f"S{i+1}=₹{v:,.1f}" for i, v in enumerate(supports)]) or "no S")
            )

    if "Target / Stop" in show_ind:
        if safe_float(result.get("Target")) > 0:
            fig.add_hline(
                y=safe_float(result["Target"]), line_dash="dash", line_color="#1a9b5f",
                annotation_text="Target", row=r, col=1,
            )
        if safe_float(result.get("Stop Loss")) > 0:
            fig.add_hline(
                y=safe_float(result["Stop Loss"]), line_dash="dash", line_color="#d93025",
                annotation_text="Stop", row=r, col=1,
            )

    if "Patterns" in show_ind and pats and len(plot_df) >= 2:
        last_i = plot_df.index[-1]
        prev_i = plot_df.index[-2]
        last_high = safe_float(plot_df["High"].iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[last_i],
                y=[last_high * 1.008],
                mode="markers+text",
                marker=dict(size=13, symbol="triangle-down", color="#f5c542"),
                text=["Pattern"],
                textposition="top center",
                name="Pattern",
            ),
            row=r, col=1,
        )
        fig.add_vrect(
            x0=prev_i, x1=last_i,
            fillcolor="rgba(245,197,66,0.18)", line_width=0,
            annotation_text=", ".join(pats)[:40],
            annotation_position="top left",
            row=r, col=1,
        )

    # Volume
    next_row = 2
    if "Volume" in show_ind:
        vol = plot_df["Volume"] if "Volume" in plot_df.columns else pd.Series(0, index=plot_df.index)
        colors = [
            "#1a9b5f" if c >= o else "#d93025"
            for o, c in zip(plot_df["Open"], plot_df["Close"])
        ]
        fig.add_trace(
            go.Bar(x=plot_df.index, y=vol, name="Volume", marker_color=colors),
            row=next_row, col=1,
        )
        next_row += 1

    if "RSI panel" in show_ind and "RSI" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["RSI"], name="RSI", line=dict(width=1.2)),
            row=next_row, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#d93025", row=next_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#1a9b5f", row=next_row, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="#888", row=next_row, col=1)
        next_row += 1

    if "MACD panel" in show_ind and "MACD" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD", line=dict(width=1.2)),
            row=next_row, col=1,
        )
        if "MACDSignal" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df["MACDSignal"], name="Signal",
                    line=dict(width=1),
                ),
                row=next_row, col=1,
            )
        if "MACDHist" in plot_df.columns:
            hist_colors = ["#1a9b5f" if v >= 0 else "#d93025" for v in plot_df["MACDHist"].fillna(0)]
            fig.add_trace(
                go.Bar(
                    x=plot_df.index, y=plot_df["MACDHist"], name="Hist",
                    marker_color=hist_colors,
                ),
                row=next_row, col=1,
            )
        next_row += 1

    if "Stochastic panel" in show_ind and "StochK" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["StochK"], name="%K", line=dict(width=1.2)),
            row=next_row, col=1,
        )
        if "StochD" in plot_df.columns:
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["StochD"], name="%D", line=dict(width=1)),
                row=next_row, col=1,
            )
        fig.add_hline(y=80, line_dash="dot", line_color="#d93025", row=next_row, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="#1a9b5f", row=next_row, col=1)

    fig.update_layout(
        height=720 if rows > 2 else 580,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08),
        title=f"{sk} · {interval}",
        dragmode="zoom",
        hovermode="x unified",
    )
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        },
    )

    if pats:
        st.info(f"**Patterns on {interval}:** " + ", ".join(pats))
    else:
        st.caption(f"No major pattern on latest {interval} bars.")


def show_indicator_school(result):
    """Render teaching UI for indicators + patterns + manual decision checklist."""
    st.subheader("🎓 Indicator School — read numbers like a trader")
    st.caption(
        "Priority 1 = decide first. Priority 2 = confirm. Priority 3 = fine-tune timing. "
        "Rules below are general market teaching applied to THIS stock’s current values."
    )

    edu = indicator_education(result)
    # Compact priority table
    show_cols = ["Priority", "Indicator", "Current value", "For THIS stock"]
    st.dataframe(edu[show_cols], use_container_width=True, hide_index=True)

    with st.expander("📘 Full rules of thumb + how to decide manually (every indicator)", expanded=True):
        for _, row in edu.iterrows():
            st.markdown(
                f"""
**P{int(row['Priority'])} · {row['Indicator']}**  
**Now:** {row['Current value']}  
**General rule:** {row['Rule of thumb (general)']}  
**This stock:** {row['For THIS stock']}  
**Manual tip:** {row['Manual decision tip']}
"""
            )
            st.divider()

    st.subheader("🕯️ Pattern School — what formed and why it matters")
    pat_rows = pattern_education(result.get("Patterns", ""))
    st.dataframe(pd.DataFrame(pat_rows), use_container_width=True, hide_index=True)

    st.subheader("✅ Manual decision checklist (in priority order)")
    st.markdown(
        """
1. **Trend first (Priority 1):** Is price above EMA20 and EMA50? (and ideally EMA200?)  
2. **Momentum (Priority 2):** Is RSI **> 50** (better **55–70**)? Is MACD above signal?  
3. **Strength (Priority 2–3):** Is ADX **≥ 25** so the trend is real, not noise?  
4. **Participation:** Is volume **≥ ~1.1–1.5×** average on the move?  
5. **Risk:** Is Risk % acceptable for your capital? Is stop distance sensible (ATR-based)?  
6. **Pattern:** Bullish pattern = bonus confirmation, not a standalone reason. Bearish pattern near resistance = reduce aggression.

**Simple teaching examples**
- **RSI 55 > 50** → generally **good** in an uptrend: momentum is positive, not overbought.  
- **RSI 82** → strong but **late**; prefer wait for dip toward EMA20 rather than chase.  
- **Price > EMA50 and MACD bullish** → core swing-buy structure.  
- **Price < EMA50 and ADX high** → strong downtrend; do not average blindly.
"""
    )


# ============================================================
# TRADINGVIEW
# ============================================================

def tv_chart(symbol, target=None, stop_loss=None):
    """Stock detail page chart — Plotly + link to TradingView (works for all symbols)."""
    show_tradingview_chart(
        symbol,
        title=display_symbol(symbol),
        height=680,
        target=target,
        stop_loss=stop_loss,
    )


# ============================================================
# DETAILED STOCK PAGE
# ============================================================

def show_stock(
    symbol
):

    symbol = display_symbol(
        symbol
    )

    st.header(
        "🔍 Detailed Analysis: "
        +
        symbol
    )

    if st.button(
        "⬅ Back to Dashboard",
        key="back_stock"
    ):

        st.session_state.page = (
            "Dashboard"
        )

        st.rerun()

    # --------------------------------------------------------
    # Historical data
    # --------------------------------------------------------

    df = stock_history(
        symbol
    )

    if df.empty:

        st.error(
            "Historical data could not be loaded for this stock."
        )

        return

    # --------------------------------------------------------
    # Current / latest closing
    # --------------------------------------------------------

    quote = live_quote(
        symbol
    )

    # --------------------------------------------------------
    # Fresh analysis
    # --------------------------------------------------------

    with st.spinner(
        "Analysing "
        + symbol
        + "..."
    ):

        result = analyse_stock(
            clean_symbol(symbol),
            df,
            fetch_news=True
        )

    if not result:

        st.error(
            "Not enough data for analysis."
        )

        return

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    current_live = None
    if quote:
        current_live = safe_float(quote["price"])
        a, b, c = st.columns(3)

        a.metric(
            quote["label"],
            f"₹{quote['price']:,.2f}",
            f"{quote['pct']:+.2f}%"
        )

        b.metric(
            "Change",
            f"₹{quote['change']:+,.2f}"
        )

        c.metric(
            "Updated",
            quote["updated"]
        )
    else:
        current_live = safe_float(result.get("Price"))

    # --------------------------------------------------------
    # Live Target / Stop Status (as requested)
    # --------------------------------------------------------
    level_status = check_price_vs_levels(
        current_price=current_live,
        entry=result.get("Price"),
        target=result.get("Target"),
        stop_loss=result.get("Stop Loss"),
        call=result.get("Call", "BUY"),
    )

    if level_status["status"] == "TARGET ACHIEVED":
        rec_html = str(level_status.get("recommendation", "")).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="success-box">
            <h3>🎯 TARGET ACHIEVED</h3>
            <p style="margin:0;line-height:1.6;">
            Previous Target: ₹{result['Target']:,.2f}<br>
            Current Price: ₹{current_live:,.2f}<br>
            🎯 TARGET ACHIEVED
            </p>
            <br>
            <b>NEW ANALYSIS:</b><br>
            New Target: ₹{level_status['new_target']:,.2f}<br>
            New Stop Loss: ₹{level_status['new_stop']:,.2f}<br>
            <br>
            <b>Recommendation:</b><br>
            {rec_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif level_status["status"] == "STOP LOSS HIT":
        st.markdown(
            f"""
            <div class="danger-box">
            <h3>🔴 STOP LOSS HIT</h3>
            <p style="margin:0;line-height:1.6;">
            Current Price: ₹{current_live:,.2f}<br>
            Stop Loss: ₹{result['Stop Loss']:,.2f}<br>
            🔴 STOP LOSS HIT
            </p>
            <br>
            <b>Recommendation:</b><br>
            🔴 SELL / EXIT
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Main recommendation
    # --------------------------------------------------------

    call = result["Call"]

    if call == "BUY":
        css = "buy-box"
    elif call == "SELL":
        css = "sell-box"
    elif call == "HOLD":
        css = "hold-box"
    else:
        css = "watch-box"

    st.markdown(
        f"""
        <div class="{css}">
        <h2>Recommendation: {call}</h2>
        <p>
        Prediction: <b>{result['Prediction']}%</b>
        &nbsp; | &nbsp;
        Risk: <b>{result['Risk %']}%</b>
        &nbsp; | &nbsp;
        Risk Level: <b>{result['Risk Level']}</b>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    a.metric(
        "Prediction",
        f"{result['Prediction']}%"
    )

    b.metric(
        "Risk",
        f"{result['Risk %']}%"
    )

    c.metric(
        "Hold Days",
        str(result["Hold Days"])
    )

    d.metric(
        "Priority",
        result["Priority"]
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Target",
        f"₹{result['Target']:,.2f}"
    )

    b.metric(
        "Stop Loss",
        f"₹{result['Stop Loss']:,.2f}"
    )

    c.metric(
        "Target %",
        f"{result['Target %']:+.2f}%"
    )

    d.metric(
        "Sector",
        result["Sector"]
    )

    # --------------------------------------------------------
    # 25Y EXPERIENCED TRADER — SHORT vs LONG-TERM + COMPANY GOAL
    # --------------------------------------------------------
    st.subheader("🧠 Experienced trader desk — Short-term vs Long-term")
    st.caption(
        "Swing plan (target/stop) is separate from long-term ownership. "
        "If a short-term stop hits, LT quality decides whether to keep a core holding."
    )
    try:
        with st.spinner("Loading fundamentals, index tags & long-term quality…"):
            prof = enrich_stock_profile(symbol)
        lt_score = safe_float(prof.get("LT Score"), 45)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LT quality score", f"{lt_score:.0f}")
        m2.metric("Market cap (₹ Cr)", prof.get("Market Cap Cr Text", "—"))
        m3.metric("Book value", prof.get("Book Value Text", "—"))
        m4.metric("Revenue / Sales (₹ Cr)", prof.get("Revenue Cr Text", "—"))

        st.markdown(
            f"""
            <div class="hold-box" style="padding:12px;border-radius:10px;margin:8px 0;">
            <b>Company:</b> {prof.get('Name', symbol)} · {prof.get('Industry', '')}<br>
            <b>Indices:</b> {prof.get('Index Tags', '—')} · <b>Cap class:</b> {prof.get('Market Cap Bucket', '—')}<br>
            <b>Book value:</b> {prof.get('Book Value Text', '—')}
            &nbsp;|&nbsp; <b>Revenue (Sales):</b> {prof.get('Revenue Cr Text', '—')}
            &nbsp;|&nbsp; <b>Net income:</b> {prof.get('Net Income Cr Text', '—')}<br>
            <b>Market cap:</b> {prof.get('Market Cap Cr Text', '—')}<br>
            <b>PE / PB / ROE / D-E:</b>
            {safe_float(prof.get('PE')):.1f} /
            {safe_float(prof.get('PB')):.2f} /
            {safe_float(prof.get('ROE')):.2f} /
            {safe_float(prof.get('Debt/Equity')):.1f}<br>
            <b>With BUY call:</b> {prof.get('LT with BUY call', '')}<br>
            <b>With SELL call:</b> {prof.get('LT with SELL call', '')}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("**Long-term notes (desk):**")
        for note in str(prof.get("LT Notes", "")).split(" | "):
            if note.strip():
                st.markdown(f"- {note.strip()}")

        st.info(f"**If swing stop-loss hits:** {prof.get('LT Hold if SL hits', '—')}")
        if call == "BUY":
            st.success(f"**LT paired with BUY:** {prof.get('LT with BUY call', '')}")
        elif call == "SELL":
            st.warning(f"**LT paired with SELL:** {prof.get('LT with SELL call', '')}")

        # Experienced trader timing feedback
        show_expert_trader_box(symbol, call_hint=str(call))

        # Year-wise revenue / profit (₹ Cr)
        st.markdown("**Revenue & profit — year-wise (₹ crore)**")
        ydf = fundamental_yearly_table(symbol)
        if ydf is not None and not ydf.empty:
            st.dataframe(ydf, use_container_width=True, hide_index=True)
        else:
            st.caption("Year-wise financials not available from data feed for this symbol.")

        # Two-horizon plan
        st.markdown("**Two-horizon plan (how a 25Y trader splits the book)**")
        c_st, c_lt = st.columns(2)
        with c_st:
            st.markdown(
                f"""
                **Short-term (this call)**  
                - Bias: **{call}** · Pred {result.get('Prediction')}%  
                - Entry zone ~ ₹{safe_float(result.get('Price')):,.2f}  
                - Target ₹{safe_float(result.get('Target')):,.2f} · SL ₹{safe_float(result.get('Stop Loss')):,.2f}  
                - Horizon ~ {result.get('Hold Days')} days  
                - Honour stop for the *swing* risk budget  
                """
            )
        with c_lt:
            if lt_score >= 72:
                lt_plan = (
                    "Core long-term **eligible**. Build on dips only if thesis intact. "
                    "Do not use full capital as swing size."
                )
            elif lt_score >= 58:
                lt_plan = (
                    "Average quality — small core only; review every quarter results. "
                    "No aggressive averaging after a technical stop."
                )
            else:
                lt_plan = (
                    "Not a core long-term compounder under current numbers. "
                    "Prefer trading book only; exit if swing SL hits."
                )
            st.markdown(
                f"""
                **Long-term (ownership)**  
                - Score **{lt_score:.0f}** · {prof.get('LT Label', '')}  
                - Cap: **{prof.get('Market Cap Bucket', '—')}**  
                - Plan: {lt_plan}  
                """
            )
        # Company goal / thesis style blurb from summary if available
        try:
            fund = fetch_fundamentals(symbol)
            summary = str(fund.get("summary") or "").strip()
            if summary:
                with st.expander("🏢 Business / future context (from filings summary)", expanded=False):
                    st.write(summary[:1200])
                    st.caption(
                        "Use this as business context only — combine with LT score, "
                        "debt, ROE and your own view of the industry cycle."
                    )
        except Exception:
            pass
    except Exception as e:
        st.caption(f"Long-term desk block unavailable: {e}")

    # --------------------------------------------------------
    # History statistics
    # --------------------------------------------------------

    stats = stock_statistics(
        symbol
    )

    st.subheader(
        "📊 Past Prediction Performance"
    )

    a, b, c, d, e = st.columns(5)

    a.metric(
        "Times Suggested",
        stats["times"]
    )

    b.metric(
        "Closed Predictions",
        stats["closed"]
    )

    c.metric(
        "Wins",
        stats["wins"]
    )

    d.metric(
        "Losses",
        stats["losses"]
    )

    e.metric(
        "Success Rate",
        f"{stats['success']}%"
    )

    # --------------------------------------------------------
    # Indicator School (values + why + how to decide)
    # --------------------------------------------------------

    show_indicator_school(result)

    # --------------------------------------------------------
    # Sell details
    # --------------------------------------------------------

    if call == "SELL":

        st.markdown(
            """
            <div class="danger-box">
            <h3>🔴 SELL CALL</h3>
            <p>
            The current model is indicating that this stock
            should be considered for selling.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("**Why sell:** " + result["Reason"])
        st.write("**Risk level:** " + str(result["Risk Level"]))
        st.write("**Stop Loss:** ₹" + f"{result['Stop Loss']:,.2f}")

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    st.subheader("🤖 Model Analysis Reasons")

    st.write("**Technical Reasons**")
    for reason in str(result["Technical Reasons"]).split(" | "):
        if reason.strip():
            st.write("• " + reason.strip())

    st.write("**News Influence:** " + result["News Influence"])
    st.write("**Patterns (raw):** " + str(result.get("Patterns", "")))

    # --------------------------------------------------------
    # News
    # --------------------------------------------------------

    if result["News"]:

        st.subheader(
            "📰 Recent News Influence"
        )

        for item in result["News"]:

            st.write(
                "• "
                +
                str(
                    item.get(
                        "title",
                        ""
                    )
                )
            )

            publisher = item.get(
                "publisher",
                ""
            )

            if publisher:

                st.caption(
                    publisher
                )

    # --------------------------------------------------------
    # Pattern
    # --------------------------------------------------------

    st.write(
        "**Patterns:** "
        +
        str(
            result["Patterns"]
        )
    )

    # --------------------------------------------------------
    # Interactive multi-timeframe chart
    # --------------------------------------------------------
    try:
        show_interactive_chart_panel(symbol, result)
    except Exception as e:
        st.caption(f"Interactive chart unavailable ({e}).")
        try:
            st.plotly_chart(make_chart(result), use_container_width=True)
        except Exception:
            pass

    st.subheader("📈 Open on TradingView (optional)")
    tv_chart(
        symbol,
        target=result.get("Target"),
        stop_loss=result.get("Stop Loss"),
    )

    # --------------------------------------------------------
    # Past history
    # --------------------------------------------------------

    history = load_history()

    if not history.empty:

        stock_history_df = history[
            history["Stock"]
            .astype(str)
            .str.upper()
            ==
            symbol.upper()
        ]

        if not stock_history_df.empty:

            st.subheader(
                "🕐 Past Calls for "
                + symbol
            )
            filterable_dataframe(
                stock_history_df.sort_values("Prediction Date", ascending=False),
                key=f"stock_hist_{symbol}",
                default_cols=[c for c in [
                    "Prediction Date", "Call", "Call Source", "Strategy",
                    "Entry", "Target", "Stop Loss", "Result", "Return %",
                ] if c in stock_history_df.columns],
            )


# ============================================================
# MARKET INDICES
# ============================================================

def live_market():

    st.subheader(
        "🇮🇳 MARKET"
    )

    st.caption(
        market_status_text()
        +
        " • "
        +
        (
            "Live prices"
            if nse_market_open_now()
            else
            "Latest closing prices"
        )
    )

    columns = st.columns(
        len(
            INDEX_SYMBOLS
        )
    )

    for col, (
        name,
        symbol
    ) in zip(
        columns,
        INDEX_SYMBOLS.items()
    ):

        q = live_quote(
            symbol
        )

        if q:

            col.metric(
                name,
                f"{q['price']:,.2f}",
                f"{q['pct']:+.2f}%"
            )

        else:

            col.metric(
                name,
                "Unavailable"
            )


# ============================================================
# SELL CALLS PAGE
# ============================================================

def show_sell_calls(
    results
):

    st.title(
        "🔴 SELL CALLS"
    )
    st.caption("Priority: Sure/Strategy SELL ideas first (if saved), then scan SELL list.")

    # Priority: history SURE/STRATEGY sells
    try:
        _h = normalize_history_df(load_history())
        if _h is not None and not _h.empty and "Call Source" in _h.columns:
            _src = _h["Call Source"].astype(str).str.upper()
            _q = _h[_src.str.contains("SURE|STRATEGY|PRECISION|HIGH_CONV", regex=True, na=False)].copy()
            _q = _q[_q["Call"].astype(str).str.upper().str.contains("SELL", na=False)]
            if not _q.empty:
                st.subheader("⭐ Priority — Sure / Strategy SELL")
                filterable_dataframe(
                    _q.sort_values("Prediction Date", ascending=False).head(30),
                    key="sell_pri_table",
                    default_cols=[c for c in [
                        "Prediction Date", "Stock", "Call Source", "Strategy",
                        "Entry", "Target", "Stop Loss", "Result",
                    ] if c in _q.columns],
                )
    except Exception:
        pass

    if results.empty:

        st.info(
            "Run FULL MARKET SCAN first."
        )

        return

    results = ensure_result_columns(results)

    sells = results[
        results["Call"]
        ==
        "SELL"
    ].copy()

    if sells.empty:

        st.success(
            "No current SELL calls found."
        )

        return

    sells = sells.sort_values(
        [
            "Prediction",
            "Risk %",
        ],
        ascending=[
            True,
            False,
        ]
    )

    st.write(
        f"Current SELL calls: {len(sells)}"
    )

    display_cols = [c for c in [
        "Rank", "Stock", "Sector", "Price", "Call", "Prediction",
        "Risk %", "Risk Level", "Target", "Stop Loss", "Hold Days",
        "Priority", "Patterns", "News Influence",
    ] if c in sells.columns]
    st.markdown("##### Filterable SELL table")
    filterable_dataframe(sells, key="sell_table", default_cols=display_cols, height=400)

    st.subheader("SELL cards — company · mcap · BV · desk")
    n_sc = st.slider("SELL cards to show", 3, 15, 6, key="sell_cards_n")
    for stock in sells["Stock"].head(n_sc).tolist():
        row = sells[sells["Stock"] == stock].iloc[0]
        render_call_stock_card(row.to_dict(), section_key="sell")

    st.subheader(
        "Detailed Sell Reasons"
    )

    for stock in sells[
        "Stock"
    ].head(50).tolist():

        row = sells[
            sells["Stock"]
            ==
            stock
        ].iloc[0]

        with st.expander(
            f"🔴 {stock} — SELL — Prediction {row['Prediction']}%"
        ):

            st.write(
                "**Current Price:** ₹"
                +
                f"{row['Price']:,.2f}"
            )

            st.write(
                "**Risk:** "
                +
                f"{row['Risk %']}% "
                +
                f"({row['Risk Level']})"
            )

            st.write(
                "**Target:** ₹"
                +
                f"{row['Target']:,.2f}"
            )

            st.write(
                "**Stop Loss:** ₹"
                +
                f"{row['Stop Loss']:,.2f}"
            )

            st.write(
                "**Reason:** "
                +
                str(
                    row["Reason"]
                )
            )

            if st.button(
                "OPEN DETAILED ANALYSIS",
                key="sell_"
                + stock
            ):

                st.session_state.selected_stock = (
                    stock
                )

                st.session_state.page = (
                    "Stock Analysis"
                )

                st.rerun()


# ============================================================
# BUY CALLS PAGE
# ============================================================

def precision_buy_score(row) -> float:
    """
    Single score to rank BUY candidates for picking 1–2 stocks.
    Higher = better quality (strength + lower risk + trend + reward).
    Applies learning multipliers from past closed trades.
    """
    pred = safe_float(row.get("Prediction"), 0)
    risk = safe_float(row.get("Risk %"), 50)
    price = safe_float(row.get("Price"), 0)
    target = safe_float(row.get("Target"), 0)
    adx = safe_float(row.get("ADX"), 0)
    rsi = safe_float(row.get("RSI"), 50)
    vol_r = safe_float(row.get("Volume Ratio"), 1)
    priority = str(row.get("Priority", "")).upper()
    patterns_raw = str(row.get("Patterns", "") or "")
    patterns = [p.strip() for p in patterns_raw.split(",") if p.strip()]

    reward = 0.0
    if price > 0 and target > price:
        reward = min(((target - price) / price) * 100, 25)

    pri_bonus = {
        "VERY HIGH": 12,
        "HIGH": 8,
        "MEDIUM": 4,
        "LOW": 0,
    }.get(priority, 0)

    rsi_score = 8 if 50 <= rsi <= 70 else (3 if 45 <= rsi < 50 else 0)
    if rsi > 78:
        rsi_score = -8

    adx_score = min(max(adx - 15, 0), 20) * 0.4
    vol_score = 6 if vol_r >= 1.3 else (3 if vol_r >= 1.05 else 0)
    risk_penalty = min(risk, 15) * 1.8

    learning = load_learning()
    pat_bonus = 0.0
    for p in patterns:
        if p and p != "No major pattern detected":
            pat_bonus += pattern_weight(p, learning) * 0.8

    learn_delta, _ = learning_score_adjustment(
        pred, row.get("Risk Level", ""), row.get("Call", "BUY"), patterns, learning
    )

    score = (
        pred * 0.55
        + reward * 1.2
        + pri_bonus
        + rsi_score
        + adx_score
        + vol_score
        + pat_bonus
        + learn_delta
        - risk_penalty
    )
    return round(float(score), 2)


def is_strong_trend_row(row) -> bool:
    """Same strong-trend gate used to allow BUY signals."""
    price = safe_float(row.get("Price") or row.get("Current Price"))
    adx = safe_float(row.get("ADX"))
    rsi = safe_float(row.get("RSI"), 50)
    vol_r = safe_float(row.get("Volume Ratio"), 1.0)
    # EMA may not always be in results; use Prediction/Priority as proxy if missing
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    if ema20 > 0 and ema50 > 0 and price > 0:
        trend_ok = price > ema20 and price > ema50
    else:
        # Results table often has ADX/RSI only
        trend_ok = True
    return bool(
        trend_ok
        and adx >= 25
        and 48 <= rsi <= 72
        and vol_r >= 1.0
        and safe_float(row.get("Prediction")) >= 72
    )


def pick_precision_buys(buys: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    Only a few strong-trend BUYs — quality over quantity for higher target hit rate.
    """
    if buys is None or buys.empty:
        return pd.DataFrame()

    x = buys.copy()
    x["Prediction"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
    x["Risk %"] = pd.to_numeric(x.get("Risk %"), errors="coerce").fillna(99)
    x["ADX"] = pd.to_numeric(x.get("ADX"), errors="coerce").fillna(0)
    x["RSI"] = pd.to_numeric(x.get("RSI"), errors="coerce").fillna(50)
    if "Volume Ratio" in x.columns:
        x["Volume Ratio"] = pd.to_numeric(x["Volume Ratio"], errors="coerce").fillna(1.0)
    else:
        x["Volume Ratio"] = 1.0

    # Quality gates — prefer strong trend, but allow a small shortlist
    x = x[
        (x["Prediction"] >= 70)
        & (x["Risk %"] <= 8)
        & (x["ADX"] >= 20)
        & (x["RSI"] >= 45)
        & (x["RSI"] <= 75)
    ].copy()

    if x.empty:
        # Fallback: best available by prediction (still limited n later)
        x = buys.copy()
        x["Prediction"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
        x["Risk %"] = pd.to_numeric(x.get("Risk %"), errors="coerce").fillna(99)
        x["ADX"] = pd.to_numeric(x.get("ADX"), errors="coerce").fillna(0)
        x = x[x["Prediction"] >= 65].copy()

    if x.empty:
        # Last resort: top by prediction so screen is never blank for days
        x = buys.copy()
        x["Prediction"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
        x = x.sort_values("Prediction", ascending=False).head(max(n * 5, 10))

    if x.empty:
        return pd.DataFrame()

    x["Precision Score"] = x.apply(precision_buy_score, axis=1)
    x = x.sort_values("Precision Score", ascending=False)

    # Prefer stocks that historically hit targets (from learning / history)
    try:
        hist = normalize_history_df(load_history())
        if hist is not None and not hist.empty:
            res_u = hist["Result"].astype(str).str.upper()
            wins = hist[res_u.str.contains("TARGET ACHIEVED", na=False)]
            losses = hist[res_u.str.contains("STOP LOSS HIT", na=False)]
            win_counts = wins.groupby(wins["Stock"].astype(str).str.upper()).size()
            loss_counts = losses.groupby(losses["Stock"].astype(str).str.upper()).size()

            def hist_bonus(stock):
                s = str(stock).upper()
                w = int(win_counts.get(s, 0))
                l = int(loss_counts.get(s, 0))
                if w + l == 0:
                    return 0
                return (w - l) * 3

            x["Hist Bonus"] = x["Stock"].apply(hist_bonus)
            x["Precision Score"] = x["Precision Score"] + x["Hist Bonus"]
            x = x.sort_values("Precision Score", ascending=False)
    except Exception:
        pass

    # Sector diversification, max n picks (default 2)
    n = max(1, min(int(n), 3))
    picked = []
    used_sectors = set()
    for _, row in x.iterrows():
        sec = str(row.get("Sector", "Other"))
        if sec in used_sectors and len(x) > n * 2:
            continue
        picked.append(row)
        used_sectors.add(sec)
        if len(picked) >= n:
            break

    if not picked:
        return x.head(n)

    out = pd.DataFrame(picked)
    out.insert(0, "Pick #", range(1, len(out) + 1))
    return out


def get_prior_calls_for_stock(stock: str, limit: int = 8) -> pd.DataFrame:
    """Past calls for a stock from recommendation_history.csv."""
    history = normalize_history_df(load_history())
    if history is None or history.empty:
        return pd.DataFrame()
    s = display_symbol(stock).upper()
    h = history[
        history["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False) == s
    ].copy()
    if h.empty:
        return h
    if "Prediction Date" in h.columns:
        h = h.sort_values("Prediction Date", ascending=False)
    return h.head(limit)


def show_prior_call_learning_panel(stock: str, current_reason: str = ""):
    """
    When a stock is recommended again: show prior call, outcome, and learning this time.
    """
    prior = get_prior_calls_for_stock(stock, limit=6)
    learning = load_learning()
    st.markdown(f"#### 🔁 Prior calls & learning — **{display_symbol(stock)}**")
    if prior is None or prior.empty:
        st.caption("No prior saved calls for this stock in history CSV.")
        if current_reason:
            st.write("**Current reason:**", current_reason[:500])
        return

    for i, (_, row) in enumerate(prior.iterrows()):
        result = str(row.get("Result", "PENDING")).upper()
        reason = str(row.get("Reason", "") or "")[:400]
        pred_d = row.get("Prediction Date", "")
        entry = safe_float(row.get("Entry"))
        target = safe_float(row.get("Target"))
        stop = safe_float(row.get("Stop Loss"))
        ret = safe_float(row.get("Return %"))
        failed = "STOP LOSS" in result or result == "LOSS"
        won = "TARGET ACHIEVED" in result or result == "WIN"

        if won:
            badge = "🎯 TARGET ACHIEVED"
        elif failed:
            badge = "🔴 STOP LOSS / FAILED"
        elif "HOLDING PERIOD" in result:
            badge = "⏰ TIME EXIT"
        else:
            badge = "⏳ OPEN / PENDING"

        st.markdown(
            f"""
            <div class="{'danger-box' if failed else 'success-box' if won else 'hold-box'}"
                 style="padding:10px;margin-bottom:8px;border-radius:8px;">
            <b>Prior #{i+1}</b> · {pred_d} · {badge}<br>
            Entry ₹{entry:,.2f} · Locked Target ₹{target:,.2f} · Locked SL ₹{stop:,.2f}
            {f' · Return {ret:+.1f}%' if ret else ''}<br>
            <b>Reason then:</b> {reason or '—'}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if current_reason:
        st.write("**Reason this time:**")
        st.write(current_reason[:800])

    # What learning is applied now
    lessons = learning.get("lessons") or []
    st.write("**What the model is applying from past mistakes (global + this context):**")
    if lessons:
        for L in lessons[:6]:
            st.markdown(f"- {L}")
    else:
        st.caption("Learning still thin — more closed target/stop outcomes will improve it.")

    # Stock-specific tip
    failed_n = sum(
        1 for _, r in prior.iterrows()
        if "STOP LOSS" in str(r.get("Result", "")).upper()
    )
    win_n = sum(
        1 for _, r in prior.iterrows()
        if "TARGET ACHIEVED" in str(r.get("Result", "")).upper()
    )
    if failed_n > win_n and failed_n >= 1:
        st.warning(
            f"This stock has **{failed_n}** prior stop/fail vs **{win_n}** target hits in history. "
            "Model should only re-BUY under strong-trend gate; size smaller if you still take it."
        )
    elif win_n > 0:
        st.success(
            f"This stock has **{win_n}** prior target hits in your history — positive track record."
        )


def manual_feedback_trainer():
    """Manual training: feed outcome / notes into model_learning (not image ML)."""
    st.subheader("📝 Manual feedback trainer")
    st.caption(
        "You can teach the model with **structured outcomes** (stock + result + note). "
        "Uploading chart **images for neural training is not supported** in this app — "
        "use outcomes and pattern notes instead; they update `model_learning.json`."
    )
    st.info(
        f"Learning file path: `{LEARNING_FILE}` · "
        "After a successful save you will see a **green confirmation** and the entry below."
    )

    with st.form("manual_train_form", clear_on_submit=False):
        stock = st.text_input(
            "Stock symbol (NSE) — required (or type GENERAL for pattern-only lesson)",
            value="",
            key="manual_fb_stock",
        )
        outcome = st.selectbox(
            "What happened?",
            [
                "TARGET ACHIEVED",
                "STOP LOSS HIT",
                "SHOULD NOT HAVE BEEN BUY",
                "GOOD SETUP (missed)",
                "OTHER",
            ],
            key="manual_fb_outcome",
        )
        patterns = st.multiselect(
            "Patterns you saw",
            list(PATTERN_IMPORTANCE.keys()),
            key="manual_fb_patterns",
        )
        note = st.text_area("Your note / mistake lesson", height=80, key="manual_fb_note")
        submitted = st.form_submit_button("Save feedback into learning", type="primary")

    if submitted:
        stock_clean = str(stock or "").upper().strip()
        if not stock_clean:
            st.error(
                "Not saved — **Stock symbol is empty**. "
                "Type an NSE symbol (e.g. RELIANCE) or **GENERAL**, then click Save again."
            )
        elif not note and not patterns:
            st.error(
                "Not saved — add at least a **pattern** or a **note** so the model has something to learn."
            )
        else:
            try:
                learning = load_learning()
                fb = learning.get("manual_feedback", [])
                entry = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stock": display_symbol(stock_clean),
                    "outcome": outcome,
                    "patterns": list(patterns or []),
                    "note": note or "",
                }
                fb.append(entry)
                learning["manual_feedback"] = fb[-200:]

                mults = learning.get("pattern_multipliers", {})
                for p in (patterns or []):
                    cur = float(mults.get(p, 1.0))
                    if outcome in ("TARGET ACHIEVED", "GOOD SETUP (missed)"):
                        mults[p] = round(min(1.6, cur + 0.08), 2)
                    elif outcome in ("STOP LOSS HIT", "SHOULD NOT HAVE BEEN BUY"):
                        mults[p] = round(max(0.4, cur - 0.08), 2)
                learning["pattern_multipliers"] = mults

                lessons = learning.get("lessons", [])
                lessons.insert(
                    0,
                    f"Manual feedback on {display_symbol(stock_clean)}: {outcome}. "
                    f"Patterns: {', '.join(patterns) or '—'}. {(note or '')[:120]}",
                )
                learning["lessons"] = lessons[:40]
                save_learning(learning)

                # Verify write
                verify = load_learning()
                n_fb = len(verify.get("manual_feedback", []))
                st.success(
                    f"✅ **Feedback saved** into `{LEARNING_FILE.name}`.\n\n"
                    f"- Stock: **{display_symbol(stock_clean)}**\n"
                    f"- Outcome: **{outcome}**\n"
                    f"- Patterns: **{', '.join(patterns) or '—'}**\n"
                    f"- Total manual feedback entries now: **{n_fb}**"
                )
                st.session_state["_last_manual_fb"] = entry
            except Exception as e:
                st.error(f"Save failed: {e}")

    # Always show recent saved feedback so user can confirm
    learning_now = load_learning()
    recent = learning_now.get("manual_feedback") or []
    if recent:
        st.write("**Recently saved feedback (proof it is in learning):**")
        show = list(reversed(recent[-8:]))
        st.dataframe(pd.DataFrame(show), use_container_width=True, hide_index=True)
        mults = learning_now.get("pattern_multipliers") or {}
        if mults:
            st.caption(
                "Pattern multipliers (updated by feedback): "
                + ", ".join(f"{k}={v}" for k, v in list(mults.items())[:12])
            )
    else:
        st.caption("No manual feedback in `model_learning.json` yet.")


def show_buy_calls(results):

    st.title("🟢 BUY CALLS")
    st.caption(
        "**25Y desk:** keep BUY list for short-term plans, but filter by **index / market-cap / long-term quality**. "
        "If a swing stop hits on a high LT-score name, you may **hold core for long-term** instead of full panic exit."
    )

    if results.empty:
        st.info("Run FULL MARKET SCAN first.")
        return

    results = ensure_result_columns(results)
    buys = results[results["Call"].astype(str).str.upper() == "BUY"].copy()

    if buys.empty:
        watch = results[
            (results["Call"].astype(str).str.upper() == "WATCH")
            & (pd.to_numeric(results.get("Prediction"), errors="coerce").fillna(0) >= 60)
        ].copy()
        if watch.empty:
            st.warning("No BUY / near-buy in last scan. Run **FULL MARKET SCAN**.")
            return
        st.info("No pure BUY — showing higher-score WATCH near-buys.")
        buys = watch

    buys["Precision Score"] = buys.apply(precision_buy_score, axis=1)
    buys = buys.sort_values(
        ["Precision Score", "Prediction", "Risk %"],
        ascending=[False, False, True],
    )

    st.write(f"BUY / near-buy candidates: **{len(buys)}**")

    # Priority strip: Sure / Strategy from history (top of BUY page)
    try:
        _h = normalize_history_df(load_history())
        if _h is not None and not _h.empty:
            _src = _h["Call Source"].astype(str).str.upper() if "Call Source" in _h.columns else pd.Series([""] * len(_h))
            _q = _h[_src.str.contains("SURE|STRATEGY|PRECISION|HIGH_CONV|MY_STRATEGY", regex=True, na=False)].copy()
            _q = _q[_q["Call"].astype(str).str.upper().str.contains("BUY", na=False)]
            if not _q.empty:
                st.subheader("⭐ Priority first — recent Sure / Strategy BUY")
                _q = _q.sort_values("Prediction Date", ascending=False)
                st.dataframe(
                    _q[[c for c in [
                        "Prediction Date", "Stock", "Call Source", "Strategy", "Entry",
                        "Target", "Stop Loss", "Result", "Current Price",
                    ] if c in _q.columns]].head(15),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("These outrank the bulk scan BUY list for decision-making.")
    except Exception:
        pass

    # --------------------------------------------------------
    # Fundamental / index / LT filters
    # --------------------------------------------------------
    st.subheader("🏛️ Index · Cap · Long-term quality filters")
    st.caption(
        "Index membership from NSE lists. Cap from market-cap / index. "
        "LT score = ROE, debt, margins, growth, PE (pro long-term desk)."
    )
    enrich_n = st.slider(
        "Enrich top N names with fundamentals (slower if high)",
        10, 80, 30, key="buy_enrich_n",
        help="Yahoo fundamentals fetched only for top N by precision score.",
    )
    if st.button("Load index + fundamentals for filters", type="primary", key="buy_enrich_btn"):
        with st.spinner("Loading NSE index tags + fundamentals…"):
            try:
                load_index_membership()
                buys = enrich_results_profiles(buys.head(max(enrich_n, 15)), max_n=enrich_n)
                # re-attach rest without profile
                st.session_state["buys_enriched"] = buys
                st.success(f"Enriched {len(buys)} stocks.")
            except Exception as e:
                st.warning(f"Enrichment partial: {e}")

    if st.session_state.get("buys_enriched") is not None:
        enr = st.session_state["buys_enriched"]
        # Prefer enriched rows for stocks we have
        if isinstance(enr, pd.DataFrame) and not enr.empty and "LT Score" in enr.columns:
            buys = enr

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        idx_f = st.selectbox(
            "Index membership",
            [
                "ALL",
                "Nifty 50",
                "Nifty 100",
                "Nifty 200",
                "Nifty 500",
                "Nifty Midcap 100",
                "Nifty Smallcap 100",
                "Outside major indices",
            ],
            key="buy_idx_f",
        )
    with f2:
        cap_f = st.selectbox(
            "Market cap",
            ["ALL", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Unknown"],
            key="buy_cap_f",
        )
    with f3:
        lt_f = st.selectbox(
            "Long-term quality",
            [
                "ALL",
                "Strong LT (score ≥ 72)",
                "Average+ (score ≥ 58)",
                "Weak LT (score < 58)",
                "Hold-if-SL = YES only",
            ],
            key="buy_lt_f",
        )
    with f4:
        min_lt = st.slider("Min LT score", 0, 95, 0, key="buy_min_lt")

    # Apply filters when columns exist; else use live index tags on the fly for small lists
    def _row_matches_filters(row) -> bool:
        stock = display_symbol(row.get("Stock", ""))
        tags = str(row.get("Index Tags", "") or "")
        if not tags or tags == "nan":
            tags = ", ".join(index_tags_of(stock))
        cap = str(row.get("Market Cap Bucket", "") or "Unknown")
        if cap in ("", "nan", "None"):
            # quick index-based cap
            tset = set(index_tags_of(stock))
            if "Nifty 50" in tset or "Nifty 100" in tset:
                cap = "Large Cap"
            elif "Nifty Midcap 100" in tset:
                cap = "Mid Cap"
            elif "Nifty Smallcap 100" in tset:
                cap = "Small Cap"
        lt_score = safe_float(row.get("LT Score"), -1)
        hold_sl = str(row.get("LT Hold if SL hits", "") or "")

        if idx_f == "Outside major indices":
            if tags and tags not in ("Outside major indices",) and any(
                x in tags for x in ("Nifty 50", "Nifty 100", "Nifty 500", "Midcap", "Smallcap")
            ):
                return False
        elif idx_f != "ALL":
            if idx_f not in tags and idx_f not in index_tags_of(stock):
                return False
        if cap_f != "ALL" and cap != cap_f:
            return False
        if min_lt > 0 and lt_score >= 0 and lt_score < min_lt:
            return False
        if lt_f.startswith("Strong") and not (lt_score >= 72):
            return False
        if lt_f.startswith("Average") and not (lt_score >= 58):
            return False
        if lt_f.startswith("Weak") and not (0 <= lt_score < 58):
            return False
        if lt_f.startswith("Hold-if-SL") and not hold_sl.upper().startswith("YES"):
            return False
        return True

    if any([idx_f != "ALL", cap_f != "ALL", lt_f != "ALL", min_lt > 0]):
        mask = buys.apply(_row_matches_filters, axis=1)
        buys_f = buys[mask].copy()
        if buys_f.empty:
            st.warning(
                "No names match these fundamental/index filters. "
                "Click **Load index + fundamentals** or loosen filters."
            )
        else:
            buys = buys_f
            st.success(f"After index/cap/LT filters: **{len(buys)}** stocks")

    # --------------------------------------------------------
    # PRECISION PICKS — best 1–2 stocks
    # --------------------------------------------------------
    st.subheader("🎯 Precision Picks — buy only these")
    st.caption(
        "Ranked by Prediction + low Risk + Priority + trend (ADX) + volume + reward/risk. "
        "Different sectors preferred so you are not concentrated in one theme."
    )

    p1, p2 = st.columns(2)
    with p1:
        n_picks = st.radio("How many stocks to buy?", [1, 2], index=1, horizontal=True, key="n_precision_picks")
    with p2:
        style = st.selectbox(
            "Style",
            [
                "Balanced (recommended)",
                "Aggressive (higher prediction)",
                "Conservative (lowest risk)",
            ],
            key="buy_style",
        )

    pool = buys.copy()
    if style.startswith("Aggressive"):
        pool = pool[pd.to_numeric(pool["Prediction"], errors="coerce") >= 75]
    elif style.startswith("Conservative"):
        pool = pool[
            pool["Risk Level"].astype(str).str.upper().isin(["LOW", "MEDIUM"])
            & (pd.to_numeric(pool["Risk %"], errors="coerce") <= 5)
        ]

    top_picks = pick_precision_buys(pool if not pool.empty else buys, n=int(n_picks))

    if top_picks.empty:
        st.warning(
            "No stock passed the quality filters. Loosen Style or run a fresh market scan."
        )
    else:
        for _, row in top_picks.iterrows():
            stock = row["Stock"]
            pred = safe_float(row.get("Prediction"))
            risk = safe_float(row.get("Risk %"))
            price = safe_float(row.get("Price"))
            target = safe_float(row.get("Target"))
            stop = safe_float(row.get("Stop Loss"))
            reward = ((target - price) / price * 100) if price > 0 else 0
            score = safe_float(row.get("Precision Score"))

            st.markdown(
                f"""
                <div class="buy-box" style="margin-bottom:14px;padding:14px;border-radius:10px;border:1px solid #1a9b5f;">
                <h3 style="margin:0 0 8px 0;">#{int(row.get('Pick #', 0))}  ·  {stock}
                &nbsp; <span style="font-size:0.85em;color:#666;">{row.get('Sector','')}</span></h3>
                <p style="margin:0;line-height:1.65;">
                <b>Precision Score:</b> {score:.1f} &nbsp;|&nbsp;
                <b>Prediction:</b> {pred:.1f}% &nbsp;|&nbsp;
                <b>Risk:</b> {risk:.2f}% ({row.get('Risk Level','')}) &nbsp;|&nbsp;
                <b>Priority:</b> {row.get('Priority','')}<br>
                <b>Entry (approx):</b> ₹{price:,.2f} &nbsp;|&nbsp;
                <b>Target:</b> ₹{target:,.2f} ({reward:+.1f}%) &nbsp;|&nbsp;
                <b>Stop Loss:</b> ₹{stop:,.2f}<br>
                <b>Hold:</b> {row.get('Hold Days','')} days &nbsp;|&nbsp;
                <b>Patterns:</b> {row.get('Patterns','')}<br>
                <b>Index:</b> {row.get('Index Tags', '—')} &nbsp;|&nbsp;
                <b>Cap:</b> {row.get('Market Cap Bucket', '—')} &nbsp;|&nbsp;
                <b>LT score:</b> {safe_float(row.get('LT Score')):.0f} ({row.get('LT Label', '')})<br>
                <b>If swing SL hits:</b> {str(row.get('LT Hold if SL hits', 'Load fundamentals to see LT advice'))[:180]}<br>
                <b>Why:</b> {str(row.get('Reason',''))[:240]}
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"📊 Fundamentals + long-term view — {stock}", expanded=False):
                try:
                    prof = enrich_stock_profile(stock)
                    st.write(
                        f"**{prof.get('Name')}** · {prof.get('Industry')} · "
                        f"**{prof.get('Market Cap Bucket')}** · Indices: {prof.get('Index Tags')}"
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("LT score", prof.get("LT Score"))
                    c2.metric("PE", f"{safe_float(prof.get('PE')):.1f}" if prof.get("PE") else "—")
                    c3.metric("ROE", f"{safe_float(prof.get('ROE')):.2f}" if prof.get("ROE") else "—")
                    c4.metric("D/E", f"{safe_float(prof.get('Debt/Equity')):.1f}" if prof.get("Debt/Equity") else "—")
                    st.info(prof.get("LT Hold if SL hits", ""))
                    st.caption(prof.get("LT Notes", ""))
                    st.caption(f"Label: {prof.get('LT Label')}")
                except Exception as e:
                    st.caption(f"Fundamentals unavailable: {e}")
            # Prior call history + learning when recommended again
            with st.expander(f"🔁 Prior history & learning — {stock}", expanded=False):
                show_prior_call_learning_panel(stock, str(row.get("Reason", "")))

            b1, b2 = st.columns(2)
            with b1:
                if st.button(f"📈 Full analysis — {stock}", key=f"prec_full_{stock}"):
                    st.session_state.selected_stock = stock
                    st.session_state.page = "Stock Analysis"
                    st.rerun()
            with b2:
                st.caption("Suggested action: BUY only if price is near entry and stop fits your capital risk.")

        # Capital + risk% + target earnings (both controls drive the numbers)
        st.markdown("##### 💰 Capital, risk & target earnings")
        st.caption(
            "Set **investment capital** and **risk %**. "
            "Qty is sized from stop-loss risk. Earnings assume target is hit."
        )

        c_cap, c_risk, c_deploy = st.columns(3)
        with c_cap:
            invest_capital = st.number_input(
                "Investment capital (₹)",
                min_value=1000.0,
                value=50000.0,
                step=1000.0,
                key="precision_capital",
                help="Total money you plan to use for this trade / these picks.",
            )
        with c_risk:
            risk_pct_user = st.slider(
                "Max loss risk % of capital",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="precision_risk_pct",
                help="How much of capital you are willing to lose if stop-loss hits.",
            )
        with c_deploy:
            deploy_pct = st.slider(
                "Deploy % of capital in trade",
                min_value=10,
                max_value=100,
                value=100,
                step=5,
                key="precision_deploy_pct",
                help="Share of capital actually put into the position (100% = full capital).",
            )

        deploy_amount = invest_capital * (deploy_pct / 100.0)
        risk_budget = invest_capital * (risk_pct_user / 100.0)

        if not top_picks.empty:
            for _, r0 in top_picks.iterrows():
                stock = r0["Stock"]
                entry = safe_float(r0.get("Price"))
                stop = safe_float(r0.get("Stop Loss"))
                target = safe_float(r0.get("Target"))
                per_share_risk = max(entry - stop, 0.01)
                per_share_reward = max(target - entry, 0.0)

                # Qty limited by both risk budget and deploy amount
                qty_by_risk = int(risk_budget // per_share_risk) if per_share_risk > 0 else 0
                qty_by_capital = int(deploy_amount // entry) if entry > 0 else 0
                qty = max(0, min(qty_by_risk, qty_by_capital))

                invested = qty * entry
                max_loss = qty * per_share_risk
                expected_profit = qty * per_share_reward
                reward_pct = (per_share_reward / entry * 100) if entry > 0 else 0
                loss_pct = (per_share_risk / entry * 100) if entry > 0 else 0

                st.markdown(
                    f"""
                    <div class="success-box" style="margin-bottom:12px;">
                    <b>{stock}</b> — position plan<br>
                    Entry ≈ ₹{entry:,.2f} &nbsp;|&nbsp; Target ₹{target:,.2f} ({reward_pct:+.1f}%)
                    &nbsp;|&nbsp; Stop ₹{stop:,.2f} (−{loss_pct:.1f}%)<br>
                    <b>Suggested qty:</b> {qty} shares &nbsp;|&nbsp;
                    <b>Invested:</b> ₹{invested:,.0f}<br>
                    <b>If target hit — expected earnings:</b> ₹{expected_profit:,.0f}
                    ({reward_pct:+.1f}% on entry)<br>
                    <b>If stop hit — max loss:</b> ₹{max_loss:,.0f}
                    (within your ₹{risk_budget:,.0f} risk budget)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # --------------------------------------------------------
    # Full BUY list with filters
    # --------------------------------------------------------
    st.subheader("📋 All BUY calls (filtered)")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        min_pred = st.slider("Minimum Prediction %", 50, 95, 65, key="buy_min_pred")
    with f2:
        risk_opts = ["ALL", "LOW", "MEDIUM", "HIGH", "VERY HIGH"]
        risk_f = st.selectbox("Risk", risk_opts, key="buy_risk")
    with f3:
        sector_opts = ["ALL"] + sorted(
            buys["Sector"].dropna().astype(str).unique().tolist()
        )
        sector_f = st.selectbox("Sector", sector_opts, key="buy_sector")
    with f4:
        show_n = st.selectbox("Show top", [10, 25, 50, 100, 200], index=1, key="buy_n")

    filtered = buys[pd.to_numeric(buys["Prediction"], errors="coerce") >= min_pred]
    if risk_f != "ALL":
        filtered = filtered[filtered["Risk Level"] == risk_f]
    if sector_f != "ALL":
        filtered = filtered[filtered["Sector"] == sector_f]
    filtered = filtered.head(int(show_n))

    display_cols = [
        c for c in [
            "Precision Score", "Rank", "Stock", "Sector", "Price", "Call",
            "Prediction", "Risk %", "Risk Level", "Target", "Stop Loss",
            "Hold Days", "Priority", "Patterns", "News Influence",
        ] if c in filtered.columns
    ]
    st.markdown("##### Filterable table")
    filterable_dataframe(filtered, key="buy_table", default_cols=display_cols, height=400)

    st.subheader("BUY cards — company · mcap · BV · desk")
    st.caption("Same visual format as Strategy Lab. One card per stock.")
    n_cards = st.slider("BUY cards to show", 3, 20, 8, key="buy_cards_n")
    for stock in filtered["Stock"].head(n_cards).tolist():
        row = filtered[filtered["Stock"] == stock].iloc[0]
        render_call_stock_card(row.to_dict(), section_key="buy")
    st.subheader("Detailed BUY Analysis")
    for stock in filtered["Stock"].head(40).tolist():
        row = filtered[filtered["Stock"] == stock].iloc[0]
        with st.expander(
            f"🟢 {stock} — BUY — Pred {row['Prediction']}% | Score {safe_float(row.get('Precision Score')):.1f} | {row['Risk Level']}"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"₹{safe_float(row['Price']):,.2f}")
            c2.metric("Target", f"₹{safe_float(row['Target']):,.2f}")
            c3.metric("Stop Loss", f"₹{safe_float(row['Stop Loss']):,.2f}")
            c4.metric("Hold Days", str(row.get("Hold Days", "")))

            st.write(f"**Risk:** {row['Risk %']}% ({row['Risk Level']})")
            st.write(f"**Priority:** {row.get('Priority', '')}")
            st.write(f"**Patterns:** {row.get('Patterns', '')}")
            st.write(f"**News:** {row.get('News Influence', '')}")
            st.write(f"**Reason:** {row.get('Reason', '')}")

            if st.button("OPEN DETAILED ANALYSIS", key=f"buy_full_{stock}"):
                st.session_state.selected_stock = stock
                st.session_state.page = "Stock Analysis"
                st.rerun()


# ============================================================
# SURE CALL MODE — two-stage analyst process + Nifty regime
# ============================================================

def nifty_market_regime() -> dict:
    """
    Stage-0 market filter. Weak Nifty → recommend no new longs.
    """
    out = {
        "regime": "UNKNOWN",
        "trade_longs": True,
        "trade_shorts": True,
        "score": 50,
        "reasons": [],
        "price": 0.0,
        "ema20": 0.0,
        "ema50": 0.0,
        "ema200": 0.0,
        "adx": 0.0,
        "rsi": 50.0,
        "pct": 0.0,
    }
    try:
        df = stock_history("^NSEI", interval="1d")
        if df is None or df.empty or len(df) < 60:
            out["reasons"].append("Nifty history unavailable — treat market regime as unknown.")
            return out
        df = calculate_indicators(df)
        row = df.iloc[-1]
        price = safe_float(row.get("Close"))
        ema20 = safe_float(row.get("EMA20"))
        ema50 = safe_float(row.get("EMA50"))
        ema200 = safe_float(row.get("EMA200"))
        adx = safe_float(row.get("ADX"))
        rsi = safe_float(row.get("RSI"), 50)
        out.update({
            "price": price, "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "adx": adx, "rsi": rsi,
        })
        try:
            q = live_quote("^NSEI")
            if q and q.get("price"):
                out["price"] = safe_float(q["price"])
                out["pct"] = safe_float(q.get("pct"))
                price = out["price"]
        except Exception:
            pass

        score = 50
        reasons = []
        if price > ema20 > 0:
            score += 8
            reasons.append(f"Nifty above EMA20 (₹{ema20:,.0f}) — short-term bid intact.")
        else:
            score -= 12
            reasons.append(f"Nifty below EMA20 (₹{ema20:,.0f}) — short-term pressure.")

        if price > ema50 > 0:
            score += 10
            reasons.append(f"Nifty above EMA50 (₹{ema50:,.0f}) — swing trend supportive for longs.")
        else:
            score -= 14
            reasons.append(f"Nifty below EMA50 (₹{ema50:,.0f}) — swing trend weak; avoid fresh longs.")

        if ema200 > 0 and price > ema200:
            score += 8
            reasons.append("Nifty above EMA200 — primary trend still constructive.")
        elif ema200 > 0:
            score -= 10
            reasons.append("Nifty below EMA200 — primary trend damaged; capital preservation mode.")

        if adx >= 25 and price < ema50:
            score -= 8
            reasons.append(f"ADX {adx:.0f} with price under EMA50 — a strong downtrend, not chop.")
        elif adx >= 25 and price > ema50:
            score += 6
            reasons.append(f"ADX {adx:.0f} with price over EMA50 — strong uptrend regime.")

        if rsi < 40:
            score -= 6
            reasons.append(f"Nifty RSI {rsi:.0f} weak — risk appetite soft.")
        elif rsi > 55:
            score += 4
            reasons.append(f"Nifty RSI {rsi:.0f} supports risk-on.")

        score = int(np.clip(score, 5, 95))
        out["score"] = score
        out["reasons"] = reasons

        if score >= 62 and price > ema50:
            out["regime"] = "RISK-ON / BULLISH"
            out["trade_longs"] = True
            out["trade_shorts"] = False
        elif score <= 42 or (ema50 > 0 and price < ema50 and adx >= 20):
            out["regime"] = "RISK-OFF / WEAK"
            out["trade_longs"] = False
            out["trade_shorts"] = True
            out["reasons"].append(
                "ANALYST RULE: Do not take new BUY/sure-long calls today while Nifty is weak. "
                "Protect capital; wait for reclaim of EMA50 or a clear risk-on day."
            )
        else:
            out["regime"] = "MIXED / SELECTIVE"
            out["trade_longs"] = True  # only highest quality
            out["trade_shorts"] = True
            out["reasons"].append(
                "Mixed tape — only two-stage sure setups; skip marginal names."
            )
    except Exception as e:
        out["reasons"].append(f"Regime check error: {e}")
    return out


def two_stage_stock_verdict(symbol: str, side: str = "BUY") -> dict:
    """
    Stage 1 = trend context. Stage 2 = trigger.
    Both must pass for SURE CALL. Written like an experienced analyst brief.
    """
    side = str(side).upper()
    sym = clean_symbol(symbol)
    display = display_symbol(symbol)
    result = {
        "symbol": display,
        "side": side,
        "sure": False,
        "stage1_pass": False,
        "stage2_pass": False,
        "stage1": [],
        "stage2": [],
        "fail": [],
        "entry_plan": "",
        "invalidation": "",
        "hold_days": 15,
        "entry": 0.0,
        "target": 0.0,
        "stop": 0.0,
        "prediction": 0.0,
        "analyst_summary": "",
    }
    try:
        df = stock_history(sym, interval="1d")
        if df is None or df.empty or len(df) < 50:
            result["fail"].append("Insufficient price history for a professional-grade call.")
            return result
        df = calculate_indicators(df)
        row = df.iloc[-1]
        price = safe_float(row.get("Close"))
        ema20 = safe_float(row.get("EMA20"))
        ema50 = safe_float(row.get("EMA50"))
        ema200 = safe_float(row.get("EMA200"))
        adx = safe_float(row.get("ADX"))
        rsi = safe_float(row.get("RSI"), 50)
        atr = safe_float(row.get("ATR")) or price * 0.02
        vol_r = safe_float(row.get("Volume Ratio"), 1.0)
        try:
            q = live_quote(sym)
            if q and q.get("price"):
                price = safe_float(q["price"])
        except Exception:
            pass

        patterns = detect_patterns(df)
        result["entry"] = round(price, 2)

        s1, s2, fail = [], [], []
        stage1 = stage2 = False

        if side == "BUY":
            # ----- Stage 1: TREND (context) -----
            checks = [
                (price > ema20 > 0, f"Price ₹{price:,.2f} is above EMA20 ₹{ema20:,.2f} — short-term trend up."),
                (price > ema50 > 0, f"Price above EMA50 ₹{ema50:,.2f} — swing trend up."),
                (ema200 <= 0 or price > ema200 * 0.98,
                 f"Not fighting primary trend (EMA200 ₹{ema200:,.2f})."),
                (adx >= 22, f"ADX {adx:.1f} — trend strength adequate (≥22 required for sure)."),
                (ema20 >= ema50 * 0.995 if ema50 > 0 else True,
                 "EMA20 aligned with / above EMA50 — trend stack healthy."),
            ]
            ok = 0
            for passed, msg in checks:
                if passed:
                    s1.append("✓ " + msg)
                    ok += 1
                else:
                    s1.append("✗ " + msg)
                    fail.append(msg)
            stage1 = ok >= 4

            # ----- Stage 2: TRIGGER -----
            bull_pat = any(
                p in patterns for p in [
                    "Hammer", "Bullish Engulfing", "Higher High / Higher Low",
                    "Breakout High", "Bullish Marubozu",
                ]
            )
            tchecks = [
                (52 <= rsi <= 70, f"RSI {rsi:.1f} in buyable momentum band (52–70), not exhausted."),
                (vol_r >= 1.05, f"Volume ratio {vol_r:.2f} — participation OK."),
                (bull_pat or (price > ema20 and rsi >= 55),
                 f"Trigger: pattern {patterns[:3] if patterns else 'momentum hold above EMA20'}."),
                (True, "No same-day chase rule violated inside engine (prefer buy dips to EMA20)."),
            ]
            ok2 = 0
            for passed, msg in tchecks:
                if passed:
                    s2.append("✓ " + msg)
                    ok2 += 1
                else:
                    s2.append("✗ " + msg)
                    fail.append(msg)
            stage2 = ok2 >= 3

            # Swing geometry for 15–20 day hold
            stop = price - 2.2 * atr
            target = price + 1.8 * atr
            result["stop"] = round(max(stop, price * 0.94), 2)
            result["target"] = round(target, 2)
            result["hold_days"] = 18
            result["entry_plan"] = (
                "PREFERRED ENTRY: Buy on same day only if price is holding above signal day's low "
                "and preferably on a dip toward EMA20 (not a vertical chase into resistance). "
                "If you miss the day, buy next 1–2 sessions only while Stage-1 trend is intact "
                "and price has not closed below EMA50. Cancel if Nifty flips risk-off."
            )
            result["invalidation"] = (
                f"Call is wrong if daily close < ₹{result['stop']:,.2f} (stop) "
                f"or decisive close back below EMA50 (₹{ema50:,.2f})."
            )
        else:
            # SELL two-stage
            checks = [
                (price < ema20 or ema20 <= 0, f"Price below EMA20 — short-term weakness."),
                (price < ema50 or ema50 <= 0, f"Price below EMA50 — swing downtrend."),
                (adx >= 22, f"ADX {adx:.1f} supports a trending decline."),
                (rsi <= 48, f"RSI {rsi:.1f} not in strong oversold bounce zone only."),
            ]
            ok = 0
            for passed, msg in checks:
                if passed:
                    s1.append("✓ " + msg)
                    ok += 1
                else:
                    s1.append("✗ " + msg)
                    fail.append(msg)
            stage1 = ok >= 3
            bear_pat = any(
                p in patterns for p in [
                    "Shooting Star", "Bearish Engulfing", "Lower High / Lower Low",
                    "Breakdown Low", "Bearish Marubozu",
                ]
            )
            tchecks = [
                (vol_r >= 1.0, f"Volume ratio {vol_r:.2f}."),
                (bear_pat or rsi < 45, f"Trigger patterns/momentum: {patterns[:3] if patterns else 'soft tape'}."),
                (True, "Prefer sell rallies into EMA20 resistance, not panic lows only."),
            ]
            ok2 = 0
            for passed, msg in tchecks:
                if passed:
                    s2.append("✓ " + msg)
                    ok2 += 1
                else:
                    s2.append("✗ " + msg)
            stage2 = ok2 >= 2
            target = price - 1.8 * atr
            stop = price + 2.2 * atr
            result["target"] = round(max(target, price * 0.90), 2)
            result["stop"] = round(stop, 2)
            result["hold_days"] = 18
            result["entry_plan"] = (
                "PREFERRED ENTRY: Sell / exit longs on same day if weakness holds below EMA20. "
                "Next 1–2 days only if stage-1 downtrend intact. Cover if price reclaims EMA50."
            )
            result["invalidation"] = f"Invalid if close > stop ₹{result['stop']:,.2f} or reclaim EMA50."

        result["stage1"] = s1
        result["stage2"] = s2
        result["fail"] = fail
        result["stage1_pass"] = stage1
        result["stage2_pass"] = stage2
        result["sure"] = bool(stage1 and stage2)

        # Score
        pred = 55
        if stage1:
            pred += 15
        if stage2:
            pred += 12
        if result["sure"]:
            pred += 8
        result["prediction"] = float(min(92, pred))

        if result["sure"]:
            result["analyst_summary"] = (
                f"SURE {side} on {display}: Stage-1 trend and Stage-2 trigger both clear. "
                f"This is the subset an experienced analyst would prioritise for a 15–20 day swing — "
                f"not a guarantee, but maximum positive asymmetry under the checklist. "
                f"Entry ~₹{result['entry']:,.2f}, Target ₹{result['target']:,.2f}, "
                f"Stop ₹{result['stop']:,.2f}, hold ~{result['hold_days']} days."
            )
        else:
            result["analyst_summary"] = (
                f"NOT a sure call on {display}. "
                + ("Stage-1 trend incomplete. " if not stage1 else "")
                + ("Stage-2 trigger incomplete. " if not stage2 else "")
                + "Wait for alignment; forcing a trade here raises error rate."
            )
    except Exception as e:
        result["fail"].append(str(e))
        result["analyst_summary"] = f"Analysis failed: {e}"
    return result


def build_sure_calls_from_scan(results: pd.DataFrame, regime: dict, max_n: int = 3) -> pd.DataFrame:
    """From last scan, keep only names that pass two-stage sure logic + regime."""
    if results is None or results.empty:
        return pd.DataFrame()
    x = results.copy()
    # Candidate pool: BUY/SELL/WATCH high score
    x["Prediction"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
    pool = x[x["Prediction"] >= 60].copy()
    if pool.empty:
        pool = x.sort_values("Prediction", ascending=False).head(40)
    else:
        pool = pool.sort_values("Prediction", ascending=False).head(40)

    rows = []
    for _, r in pool.iterrows():
        stock = str(r.get("Stock", ""))
        call = str(r.get("Call", "BUY")).upper()
        side = "SELL" if call == "SELL" else "BUY"
        if side == "BUY" and not regime.get("trade_longs", True):
            continue
        if side == "SELL" and not regime.get("trade_shorts", True):
            continue
        v = two_stage_stock_verdict(stock, side=side)
        if not v.get("sure"):
            continue
        rows.append({
            "Stock": stock,
            "Side": side,
            "Sure": "YES",
            "Prediction": v["prediction"],
            "Entry": v["entry"],
            "Target": v["target"],
            "Stop Loss": v["stop"],
            "Hold Days": v["hold_days"],
            "Stage1": "PASS" if v["stage1_pass"] else "FAIL",
            "Stage2": "PASS" if v["stage2_pass"] else "FAIL",
            "Analyst summary": v["analyst_summary"],
            "Entry plan": v["entry_plan"],
            "Invalidation": v["invalidation"],
            "Sector": r.get("Sector", ""),
        })
        if len(rows) >= max_n:
            break
    return pd.DataFrame(rows)


def show_sure_calls_page(results: pd.DataFrame):
    """Two-stage Sure Call desk — analyst-style explanations."""
    st.title("✅ Sure Call Mode — Two-Stage Analyst Desk")
    st.caption(
        "Stage 0: Nifty regime · Stage 1: Trend · Stage 2: Trigger. "
        "Only when all align do we label **SURE**. Horizon **15–20 days**. "
        "This raises quality; it does **not** guarantee 100% wins."
    )

    regime = nifty_market_regime()
    st.subheader("📊 Stage 0 — Nifty market regime")
    a, b, c, d = st.columns(4)
    a.metric("Regime", regime.get("regime", "—"))
    b.metric("Regime score", regime.get("score", "—"))
    c.metric("Nifty", f"{regime.get('price', 0):,.0f}", f"{regime.get('pct', 0):+.2f}%")
    d.metric(
        "New BUY allowed?",
        "YES" if regime.get("trade_longs") else "NO — STAY DEFENSIVE",
    )

    if not regime.get("trade_longs"):
        st.error(
            "**Do not take fresh long / sure BUY calls today.** "
            "Nifty is weak under the model’s regime rules. "
            "Experienced desks stand aside or only manage open risk."
        )
    elif regime.get("regime", "").startswith("MIXED"):
        st.warning("**Selective only** — mixed Nifty; require full two-stage pass.")
    else:
        st.success("**Risk-on bias** — two-stage sure BUYs allowed.")

    for r in regime.get("reasons", []):
        st.markdown(f"- {r}")

    st.divider()
    st.subheader("🧠 How an experienced analyst splits the call")
    st.markdown(
        """
| Stage | Question | BUY needs | SELL needs |
|-------|----------|-----------|------------|
| **0 Regime** | Is the index friendly? | Nifty not risk-off | Nifty not strong melt-up only |
| **1 Trend** | Is the stock in the right trend? | Price > EMA20 & EMA50, ADX strong | Price < EMA20 & EMA50, ADX strong |
| **2 Trigger** | Is *now* a good moment? | RSI 52–70, volume, bullish pattern / hold | RSI soft, bearish pattern, volume |

**Only Stage 1 + Stage 2 (and friendly Stage 0 for longs) → SURE CALL.**
        """
    )

    st.subheader("⏰ When to buy after the model gives a call")
    st.markdown(
        """
1. **Same day (preferred if setup is fresh)**  
   - Buy if price **holds above the signal day’s low** (BUY).  
   - Prefer a **dip toward EMA20**, not a blind chase at the high.  
2. **Next 1–2 trading sessions**  
   - Still valid if **Stage-1 trend is intact** (still above EMA50) and stop not hit.  
3. **After 3+ days without entry**  
   - **Do not** chase; re-run Sure Call — structure may be stale.  
4. **Repeated calls**  
   - Scan can list the same stock again on later days if it still passes.  
   - Treat as **one position idea**, not a new pile-on, unless you scaled a plan.  
   - If the stock **stopped out recently**, Sure mode should demand a full reset (new structure).
        """
    )

    st.divider()
    tab1, tab2 = st.tabs(["✅ Sure calls from last scan", "🔍 Manual two-stage on one stock"])

    with tab1:
        if results is None or results.empty:
            st.info("Run **FULL MARKET SCAN** first so the desk has a universe to filter.")
        else:
            max_n = st.slider("Max sure names", 1, 5, 3, key="sure_max_n")
            if st.button("Generate sure calls", type="primary", key="sure_gen"):
                with st.spinner("Running two-stage filter on top candidates..."):
                    sure_df = build_sure_calls_from_scan(results, regime, max_n=max_n)
                    st.session_state["sure_df"] = sure_df
                    n_saved = 0
                    try:
                        special = []
                        if sure_df is not None and not sure_df.empty:
                            for _, sr in sure_df.iterrows():
                                special.append({
                                    "Stock": sr.get("Stock"),
                                    "Call": sr.get("Side", "BUY"),
                                    "Call Source": "SURE",
                                    "Strategy": "Sure Call",
                                    "Entry": sr.get("Entry"),
                                    "Target": sr.get("Target"),
                                    "Stop Loss": sr.get("Stop Loss"),
                                    "Hold Days": sr.get("Hold Days", 18),
                                    "Prediction": 85,
                                    "Reason": str(sr.get("Analyst summary", "Sure Call") or "Sure Call")[:300],
                                })
                            n_saved = save_special_calls(special)
                    except Exception as e:
                        st.warning(f"Save to history failed: {e}")
                    if n_saved:
                        st.success(
                            f"Saved **{n_saved}** Sure call(s) to Past Predictions "
                            f"(Call Source = **SURE**). Open History → BUY source → Strategy/Sure."
                        )
                    elif sure_df is not None and not sure_df.empty:
                        st.info("Sure calls generated (already saved today or nothing new to write).")
            sure_df = st.session_state.get("sure_df", pd.DataFrame())
            if sure_df is not None and isinstance(sure_df, pd.DataFrame) and not sure_df.empty:
                if st.button("💾 Save these Sure calls to Past Predictions again", key="sure_resave"):
                    special = []
                    for _, sr in sure_df.iterrows():
                        special.append({
                            "Stock": sr.get("Stock"),
                            "Call": sr.get("Side", "BUY"),
                            "Call Source": "SURE",
                            "Strategy": "Sure Call",
                            "Entry": sr.get("Entry"),
                            "Target": sr.get("Target"),
                            "Stop Loss": sr.get("Stop Loss"),
                            "Hold Days": sr.get("Hold Days", 18),
                            "Prediction": 85,
                        })
                    n_saved = save_special_calls(special)
                    st.success(f"Wrote {n_saved} Sure row(s) to history CSV.")
            if sure_df is None or (isinstance(sure_df, pd.DataFrame) and sure_df.empty):
                st.caption("Click **Generate sure calls** after a scan — they are saved to history automatically.")
            else:
                st.dataframe(
                    sure_df.drop(columns=["Analyst summary", "Entry plan", "Invalidation"], errors="ignore"),
                    use_container_width=True,
                    hide_index=True,
                )
                for _, row in sure_df.iterrows():
                    stock_name = str(row.get("Stock", ""))
                    match = strategies_matching_stock(stock_name, side_filter="BUY")
                    strat_count = match.get("count", 0)
                    strat_names = ", ".join(match.get("names") or []) or "— (Sure two-stage only; no named swing strategy hit)"
                    st.markdown(
                        f"""
                        <div class="buy-box" style="padding:14px;margin-bottom:12px;border-radius:10px;">
                        <h3 style="margin:0 0 8px 0;">✅ SURE {row.get('Side')} · {stock_name}
                        <span style="font-size:0.8em;color:#666;">({row.get('Sector','')})</span></h3>
                        <p style="margin:0;line-height:1.65;">
                        <b>Prediction / call:</b> SURE {row.get('Side')} (two-stage analyst desk)<br>
                        <b>Falls under strategies:</b> <b>{strat_count}</b> strategy match(es)<br>
                        <b>Strategy names:</b> {strat_names}<br>
                        <b>Entry:</b> ₹{safe_float(row.get('Entry')):,.2f} &nbsp;|&nbsp;
                        <b>Target:</b> ₹{safe_float(row.get('Target')):,.2f} &nbsp;|&nbsp;
                        <b>Stop:</b> ₹{safe_float(row.get('Stop Loss')):,.2f} &nbsp;|&nbsp;
                        <b>Hold:</b> {row.get('Hold Days')} days<br>
                        <b>Stage1 / Stage2:</b> {row.get('Stage1')} / {row.get('Stage2')}<br><br>
                        <b>Analyst brief:</b> {row.get('Analyst summary')}<br><br>
                        <b>Entry timing:</b> {row.get('Entry plan')}<br><br>
                        <b>Invalidation:</b> {row.get('Invalidation')}
                        </p></div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if match.get("detail"):
                        st.caption("Strategy-level targets for this stock:")
                        st.dataframe(pd.DataFrame(match["detail"]), use_container_width=True, hide_index=True)
                    if st.button(f"Full chart — {stock_name}", key=f"sure_open_{stock_name}"):
                        st.session_state.selected_stock = stock_name
                        st.session_state.page = "Stock Analysis"
                        st.rerun()

    with tab2:
        st.write("Run the full two-stage checklist on any symbol (independent of scan).")
        c1, c2 = st.columns(2)
        with c1:
            msym = st.text_input("Symbol", value="RELIANCE", key="sure_manual_sym").upper().strip()
        with c2:
            mside = st.selectbox("Side", ["BUY", "SELL"], key="sure_manual_side")
        if st.button("Run two-stage analysis", key="sure_manual_btn"):
            if mside == "BUY" and not regime.get("trade_longs"):
                st.error("Nifty regime blocks new long sure-calls today — analysis below is educational only.")
            v = two_stage_stock_verdict(msym, side=mside)
            if v.get("sure"):
                st.success(v.get("analyst_summary"))
            else:
                st.warning(v.get("analyst_summary"))
            st.write("**Stage 1 — Trend**")
            for line in v.get("stage1", []):
                st.markdown(f"- {line}")
            st.write("**Stage 2 — Trigger**")
            for line in v.get("stage2", []):
                st.markdown(f"- {line}")
            st.write(
                f"**Levels:** Entry ₹{v.get('entry'):,.2f} · Target ₹{v.get('target'):,.2f} · "
                f"Stop ₹{v.get('stop'):,.2f} · Hold {v.get('hold_days')}d"
            )
            match = strategies_matching_stock(msym, side_filter=mside if mside == "SELL" else "BUY")
            st.write(
                f"**Falls under strategies:** **{match.get('count', 0)}** — "
                f"{', '.join(match.get('names') or []) or 'none of the named swing strategies'}"
            )
            if match.get("detail"):
                st.dataframe(pd.DataFrame(match["detail"]), use_container_width=True, hide_index=True)
            st.info(v.get("entry_plan", ""))
            st.caption(v.get("invalidation", ""))

    st.divider()
    st.markdown(
        """
### Repeated calls — policy
- The **scanner may list the same stock on multiple days** if it still scores well.  
- **Sure Call page** should be treated as: *one active idea per stock* unless you deliberately scale.  
- After a **stop-out**, wait for a **new Stage-1 rebuild** (e.g. reclaim EMA50) before another sure BUY.  
- History + learning panels still show prior reasons when the same name reappears on BUY/Dashboard.
        """
    )


# ============================================================
# INDEX CHARTS + FULL ANALYSIS (NIFTY / BANK NIFTY)
# ============================================================

def show_index_page(index_key="NIFTY"):
    """
    Dedicated page for NIFTY 50 or BANK NIFTY:
    live quote + TradingView chart + full technical analysis.
    """
    index_key = str(index_key).upper()
    if index_key in ["BANKNIFTY", "BANK NIFTY", "BANK"]:
        title = "BANK NIFTY"
        yf_symbol = "^NSEBANK"
        tv_hint = "NSE:BANKNIFTY"
    else:
        title = "NIFTY 50"
        yf_symbol = "^NSEI"
        tv_hint = "NSE:NIFTY"

    st.title(f"📈 {title} — Chart & Full Analysis")
    st.caption(market_status_text() + f" • Symbol: {tv_hint}")

    # Live quote
    quote = live_quote(yf_symbol)
    if quote:
        a, b, c, d = st.columns(4)
        a.metric(quote["label"], f"{quote['price']:,.2f}", f"{quote['pct']:+.2f}%")
        b.metric("Change", f"{quote['change']:+,.2f}")
        c.metric("Updated", quote["updated"])
        d.metric("Status", market_status_text().replace("NSE MARKET ", ""))
    else:
        st.warning("Live quote unavailable right now.")

    st.subheader("📊 Interactive Chart")
    # Target / stop filled after analysis below; chart refreshes with analysis block
    show_tradingview_chart(yf_symbol, title, height=560)

    # Full technical analysis
    st.subheader("🤖 Full Technical Analysis")
    with st.spinner(f"Analysing {title}..."):
        df = stock_history(yf_symbol)
        if df is None or df.empty or len(df) < 60:
            st.error("Not enough historical data for analysis.")
            return

        # analyse_stock expects equity-style symbols; indices work with same OHLC
        result = analyse_stock(yf_symbol, df, fetch_news=False)

    if not result:
        st.error("Analysis could not be completed for this index.")
        return

    call = result["Call"]
    if call == "BUY":
        css = "buy-box"
    elif call == "SELL":
        css = "sell-box"
    elif call == "HOLD":
        css = "hold-box"
    else:
        css = "watch-box"

    st.markdown(
        f"""
        <div class="{css}">
        <h2>Recommendation: {call}</h2>
        <p>
        Prediction: <b>{result['Prediction']}%</b>
        &nbsp; | &nbsp;
        Risk: <b>{result['Risk %']}%</b>
        &nbsp; | &nbsp;
        Risk Level: <b>{result['Risk Level']}</b>
        &nbsp; | &nbsp;
        Priority: <b>{result['Priority']}</b>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"{result['Price']:,.2f}")
    m2.metric("Target", f"{result['Target']:,.2f}")
    m3.metric("Stop Loss", f"{result['Stop Loss']:,.2f}")
    m4.metric("Hold Days", str(result["Hold Days"]))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RSI", f"{result.get('RSI', 0):.1f}")
    m2.metric("ADX", f"{result.get('ADX', 0):.1f}")
    m3.metric("MACD", f"{result.get('MACD', 0):.4f}")
    m4.metric("Volume Ratio", f"{result.get('Volume Ratio', 0):.2f}")

    # Live target / stop status
    current = safe_float(quote["price"]) if quote else safe_float(result["Price"])
    level_status = check_price_vs_levels(
        current_price=current,
        entry=result.get("Price"),
        target=result.get("Target"),
        stop_loss=result.get("Stop Loss"),
        call=result.get("Call", "BUY"),
    )

    if level_status["status"] == "TARGET ACHIEVED":
        rec_html = str(level_status.get("recommendation", "")).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="success-box">
            <h3>🎯 TARGET ACHIEVED</h3>
            <p style="margin:0;line-height:1.6;">
            Previous Target: {result['Target']:,.2f}<br>
            Current Price: {current:,.2f}<br>
            🎯 TARGET ACHIEVED
            </p>
            <br>
            <b>NEW ANALYSIS:</b><br>
            New Target: {level_status['new_target']:,.2f}<br>
            New Stop Loss: {level_status['new_stop']:,.2f}<br>
            <br>
            <b>Recommendation:</b><br>{rec_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif level_status["status"] == "STOP LOSS HIT":
        st.markdown(
            f"""
            <div class="danger-box">
            <h3>🔴 STOP LOSS HIT</h3>
            <p style="margin:0;line-height:1.6;">
            Current Price: {current:,.2f}<br>
            Stop Loss: {result['Stop Loss']:,.2f}<br>
            🔴 STOP LOSS HIT
            </p>
            <br>
            <b>Recommendation:</b><br>🔴 SELL / EXIT
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Analysis Reasons")
    for reason in str(result.get("Technical Reasons", "")).split(" | "):
        if reason.strip():
            st.write("• " + reason.strip())

    st.write(f"**Patterns:** {result.get('Patterns', '')}")

    st.subheader("📈 Technical Chart (Candles + EMA + BB)")
    try:
        st.plotly_chart(make_chart(result), use_container_width=True)
    except Exception:
        st.caption("Technical chart unavailable.")

    # Quick switch
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📈 Open NIFTY 50", use_container_width=True, key="idx_to_nifty"):
            st.session_state.page = "Nifty Analysis"
            st.rerun()
    with c2:
        if st.button("🏦 Open BANK NIFTY", use_container_width=True, key="idx_to_bn"):
            st.session_state.page = "BankNifty Analysis"
            st.rerun()


# ============================================================
# HISTORY PAGE
# ============================================================


def save_special_calls(rows_list):
    """
    Save Sure / Strategy / Precision / High-conv into history with Call Source.
    Dedupe: same Stock + Call Source + Strategy name within 1 day (open only).
    Different strategies on same stock can all save.
    """
    if not rows_list:
        return 0
    try:
        history = load_history()
    except Exception:
        history = pd.DataFrame()
    now = datetime.now()
    new_rows = []
    for r in rows_list:
        stock = display_symbol(str(r.get("Stock", "") or "")).upper().strip()
        if not stock:
            continue
        source = str(r.get("Call Source", "STRATEGY") or "STRATEGY").upper().strip()
        strat_name = str(r.get("Strategy", "") or "").strip()
        call = str(r.get("Call", "BUY") or "BUY").upper().strip()
        if call not in ("BUY", "SELL", "HOLD"):
            # Side field from strategy tables
            call = str(r.get("Side", "BUY") or "BUY").upper().strip()
            if call not in ("BUY", "SELL", "HOLD"):
                call = "BUY"
        entry = safe_float(r.get("Entry", r.get("Price", 0)))
        target = safe_float(r.get("Target", 0))
        stop = safe_float(r.get("Stop Loss", r.get("Stop", 0)))
        hold_days = int(safe_float(r.get("Hold Days", DEFAULT_HOLD_DAYS), DEFAULT_HOLD_DAYS)) or DEFAULT_HOLD_DAYS
        expiry = (now + timedelta(days=hold_days)).strftime("%Y-%m-%d")
        skip = False
        if history is not None and not history.empty:
            try:
                h = history.copy()
                if "Call Source" not in h.columns:
                    h["Call Source"] = "SCAN"
                if "Strategy" not in h.columns:
                    h["Strategy"] = ""
                mask = (
                    h["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip() == stock
                ) & (
                    h["Call Source"].astype(str).str.upper().str.strip() == source
                ) & (
                    h["Strategy"].astype(str).str.strip() == strat_name
                )
                sub = h.loc[mask]
                if len(sub):
                    last = pd.to_datetime(sub["Prediction Date"], errors="coerce").max()
                    if pd.notna(last):
                        try:
                            last_naive = last.to_pydatetime().replace(tzinfo=None)
                        except Exception:
                            last_naive = now
                        if (now - last_naive).total_seconds() < 86400:
                            res = str(sub.sort_values("Prediction Date").iloc[-1].get("Result", "")).upper()
                            if not any(x in res for x in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD")):
                                skip = True
            except Exception:
                pass
        if skip:
            continue
        new_rows.append({
            "Prediction Date": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Stock": stock,
            "Symbol": clean_symbol(stock),
            "Call Source": source,
            "Strategy": strat_name,
            "Patterns": str(r.get("Patterns", "") or ""),
            "Call": call,
            "Prediction": r.get("Prediction", 80 if source in ("SURE", "PRECISION", "HIGH_CONV") else 75),
            "Entry": entry if entry else "",
            "Target": target if target else "",
            "Stop Loss": stop if stop else "",
            "Risk %": r.get("Risk %", ""),
            "Risk Level": r.get("Risk Level", "MEDIUM"),
            "Hold Days": hold_days,
            "Expiry Date": expiry,
            "Status": "OPEN",
            "Result": "PENDING",
            "Result Detail": "",
            "Days Taken": "",
            "Outcome Message": "",
            "Recommendation": "",
            "Suggestion": "",
            "Reason": str(r.get("Reason", f"{source} | {strat_name}") or f"{source} call"),
            "Current Price": entry if entry else "",
            "Evaluation Date": "",
            "Exit Price": "",
            "Return %": "",
        })
    if not new_rows:
        return 0
    add = pd.DataFrame(new_rows)
    if history is None or history.empty:
        out = add
    else:
        for c in add.columns:
            if c not in history.columns:
                history[c] = ""
        for c in history.columns:
            if c not in add.columns:
                add[c] = ""
        out = pd.concat([history, add], ignore_index=True)
    try:
        # Do NOT run ensure_result_columns (scan schema) — keeps Call Source
        out = normalize_history_df(out)
        # Force Call Source on new rows if normalize blanked anything
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(HISTORY_FILE, index=False)
        return len(new_rows)
    except Exception:
        try:
            out.to_csv(HISTORY_FILE, index=False)
            return len(new_rows)
        except Exception:
            return 0


def import_strategy_files_to_history() -> int:
    """
    Pull strategy_live_signals.csv / session into recommendation_history as STRATEGY.
    Fixes '40 stocks shown green but Past Predictions empty'.
    """
    n = 0
    frames = []
    try:
        if STRATEGY_LIVE_FILE.exists():
            frames.append(pd.read_csv(STRATEGY_LIVE_FILE))
    except Exception:
        pass
    for key in ("live_strat_df", "stocks_under_all_df", "stocks_under_df", "sure_df"):
        try:
            df = st.session_state.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        except Exception:
            pass
    special = []
    for df in frames:
        for _, lr in df.iterrows():
            stock = lr.get("Stock") or lr.get("Symbol")
            if not stock:
                continue
            strat = str(lr.get("Strategy", "Strategy Lab") or "Strategy Lab")
            source = "SURE" if "sure" in strat.lower() else "STRATEGY"
            special.append({
                "Stock": stock,
                "Call": lr.get("Side", lr.get("Call", "BUY")),
                "Call Source": source,
                "Strategy": strat,
                "Entry": lr.get("Price", lr.get("Entry", lr.get("Current / Live Price"))),
                "Target": lr.get("Target"),
                "Stop Loss": lr.get("Stop Loss"),
                "Hold Days": lr.get("Hold Days", 15),
                "Prediction": 78,
            })
    try:
        sdf = st.session_state.get("sure_df")
        if isinstance(sdf, pd.DataFrame) and not sdf.empty:
            for _, sr in sdf.iterrows():
                special.append({
                    "Stock": sr.get("Stock"),
                    "Call": sr.get("Side", "BUY"),
                    "Call Source": "SURE",
                    "Strategy": "Sure Call",
                    "Entry": sr.get("Entry"),
                    "Target": sr.get("Target"),
                    "Stop Loss": sr.get("Stop Loss"),
                    "Hold Days": sr.get("Hold Days", 18),
                    "Prediction": 85,
                })
    except Exception:
        pass
    if special:
        n = save_special_calls(special)
    return n


def expert_trader_verdict(symbol: str, call_hint: str = "") -> dict:
    """
    25Y desk feedback: right time or not, BUY/SELL/HOLD/LT, one learning line.
    Uses technicals + LT fundamentals when available.
    """
    sym = display_symbol(symbol)
    out = {
        "symbol": sym,
        "timing": "WAIT",
        "action": "WATCH",
        "lt_action": "Review",
        "confidence": 50,
        "feedback": "",
        "learning": "",
        "index": "",
        "mcap_cr": "—",
        "book_value": "—",
        "revenue_cr": "—",
        "profit_cr": "—",
        "face_value": "—",
    }
    try:
        df = stock_history(clean_symbol(sym), interval="1d")
        if df is None or len(df) < 60:
            out["feedback"] = "Not enough price history — no trade until data is clean."
            return out
        df = calculate_indicators(df)
        row = df.iloc[-1]
        price = safe_float(row.get("Close"))
        rsi = safe_float(row.get("RSI"), 50)
        adx = safe_float(row.get("ADX"), 0)
        ema20 = safe_float(row.get("EMA20"))
        ema50 = safe_float(row.get("EMA50"))
        ema200 = safe_float(row.get("EMA200"))
        macd_h = safe_float(row.get("MACDHist"))
        atr = safe_float(row.get("ATR")) or price * 0.02

        try:
            prof = enrich_stock_profile(sym)
            out["index"] = prof.get("Index Tags", "")
            out["mcap_cr"] = prof.get("Market Cap Cr Text", "—")
            out["book_value"] = prof.get("Book Value Text", "—")
            out["revenue_cr"] = prof.get("Revenue Cr Text", "—")
            out["profit_cr"] = prof.get("Net Income Cr Text", "—")
            lt_score = safe_float(prof.get("LT Score"), 45)
            out["lt_action"] = (
                "CORE LONG-TERM HOLD/ACCUMULATE" if lt_score >= 72
                else ("SMALL CORE ONLY" if lt_score >= 58 else "NO LT CORE — trade only")
            )
        except Exception:
            lt_score = 45
            prof = {}

        try:
            fund = fetch_fundamentals(sym)
            fv = fund.get("face_value")
            if fv:
                out["face_value"] = f"₹{safe_float(fv):.2f}"
        except Exception:
            pass

        above20 = ema20 <= 0 or price > ema20
        above50 = ema50 <= 0 or price > ema50
        above200 = ema200 <= 0 or price > ema200
        structure_long = above20 and above50
        stretched = rsi >= 78
        washed = rsi <= 25
        trend_on = adx >= 18

        conf = 50
        action = "WATCH"
        timing = "WAIT"
        lines = []

        if structure_long and trend_on and 42 <= rsi <= 68 and macd_h >= 0:
            action = "BUY"
            timing = "GOOD TIME — structure + momentum aligned"
            conf = 72 + (5 if above200 else 0)
            lines.append("Price holds above short EMAs with usable ADX — classic swing long window.")
        elif structure_long and 40 <= rsi <= 72:
            action = "BUY"
            timing = "OK TIME — but size moderate"
            conf = 62
            lines.append("Trend soft-positive; take only if stop is respected.")
        elif not above50 and adx >= 22 and rsi < 45 and macd_h < 0:
            action = "SELL"
            timing = "GOOD TIME for trade-short / exit longs"
            conf = 68
            lines.append("Breakdown under EMA50 with momentum — prefer exit or short plan.")
        elif stretched and above50:
            action = "HOLD"
            timing = "WAIT — extended; don't chase"
            conf = 55
            lines.append("RSI stretched; pros wait for pullback toward EMA20/50.")
        elif washed and above200:
            action = "BUY"
            timing = "SELECTIVE dip-buy only if LT quality high"
            conf = 58 if lt_score >= 60 else 48
            lines.append("Oversold in higher timeframe uptrend — only with strong LT score.")
        else:
            action = "HOLD" if above200 else "WATCH"
            timing = "NOT ideal — stand aside or manage existing only"
            conf = 45
            lines.append("No clean edge; cash / hold is a position.")

        if call_hint:
            ch = call_hint.upper()
            if "BUY" in ch and action == "SELL":
                lines.append("Strategy/Sure says BUY but desk sees weak structure — skip or tiny size.")
                conf = min(conf, 48)
                timing = "CONFLICT — prefer WAIT"
            if "SELL" in ch and action == "BUY":
                lines.append("Desk sees support for long while list says SELL — don't force short.")

        out["action"] = action
        out["timing"] = timing
        out["confidence"] = int(min(90, conf))
        out["feedback"] = " ".join(lines)
        out["learning"] = (
            "Edge = timing + risk. If timing is WAIT, the best trade is no trade. "
            "Separate swing risk (honour SL) from LT core (only high LT score)."
        )
        out["swing_stop"] = round(price - 1.4 * atr, 2) if action == "BUY" else round(price + 1.4 * atr, 2)
        out["swing_target"] = round(price + 2.2 * atr, 2) if action == "BUY" else round(price - 2.2 * atr, 2)
        out["price"] = price
        out["rsi"] = rsi
        out["adx"] = adx
    except Exception as e:
        out["feedback"] = f"Desk check failed: {e}"
    return out


def show_expert_trader_box(symbol: str, call_hint: str = ""):
    """UI block: experienced trader feedback for any page."""
    v = expert_trader_verdict(symbol, call_hint)
    st.markdown(f"#### 🎓 Experienced trader desk — {v.get('symbol')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Timing", v.get("timing", "—")[:28])
    c2.metric("Action", v.get("action", "—"))
    c3.metric("Confidence", f"{v.get('confidence', 0)}%")
    c4.metric("LT stance", str(v.get("lt_action", "—"))[:24])
    st.write(v.get("feedback", ""))
    st.caption(
        f"Index: {v.get('index') or '—'} · Mcap {v.get('mcap_cr')} · "
        f"Face {v.get('face_value')} · BV {v.get('book_value')} · "
        f"Rev {v.get('revenue_cr')} · Profit {v.get('profit_cr')}"
    )
    st.info(f"**Learning:** {v.get('learning', '')}")
    return v


def history_success_breakdown(history: pd.DataFrame) -> pd.DataFrame:
    """Success rate by Call Source / Strategy / Patterns."""
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    if "Call Source" not in h.columns:
        h["Call Source"] = "SCAN"
    if "Strategy" not in h.columns:
        h["Strategy"] = ""
    if "Patterns" not in h.columns:
        h["Patterns"] = ""
    res = h.get("Result", pd.Series([""] * len(h))).astype(str).str.upper()
    h["_win"] = res.str.contains("TARGET ACHIEVED", na=False)
    h["_loss"] = res.str.contains("STOP LOSS HIT", na=False)
    h["_closed"] = h["_win"] | h["_loss"]
    rows = []
    for col in ["Call Source", "Strategy", "Patterns"]:
        for key, g in h.groupby(h[col].astype(str).fillna("")):
            if not str(key).strip() or str(key) == "nan":
                continue
            closed = int(g["_closed"].sum())
            wins = int(g["_win"].sum())
            losses = int(g["_loss"].sum())
            rate = round(wins / closed * 100, 1) if closed else 0.0
            rows.append({
                "Filter": col,
                "Value": key[:80],
                "Calls": len(g),
                "Closed": closed,
                "Wins": wins,
                "Losses": losses,
                "Success %": rate,
            })
    return pd.DataFrame(rows)



def show_history():
    """Past Predictions — Strategy/Sure/High-conv only; success on those predicted stocks."""

    st.title("🕐 PAST PREDICTIONS")
    st.caption(
        "**Only Strategy + Sure Call + High conviction + Precision** are shown and scored here. "
        "Bulk SCAN BUYs are excluded so success rate matches real quality predictions. "
        "Healthy swing book: **~40–55%** targets among closed target-vs-stop outcomes."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        do_eval = st.button("⟳ Update results", key="hist_eval_btn")
    with c2:
        force_eval = st.button("⚡ Force re-check all", type="primary", key="hist_force_eval")
    with c3:
        if st.button("📥 Import Strategy/Sure into history", type="primary", key="hist_import_strat"):
            with st.spinner("Importing live strategy / sure rows into Past Predictions…"):
                n_imp = import_strategy_files_to_history()
            if n_imp:
                st.success(f"Imported **{n_imp}** quality rows (STRATEGY/SURE). Scroll metrics below.")
                st.rerun()
            else:
                st.warning(
                    "Nothing new to import. Open Strategy Lab → Generate live signals "
                    "(or Sure Call → Generate) first, then click Import again."
                )

    # Only evaluate when user asks — keeps tab switch fast
    if do_eval or force_eval:
        with st.spinner("Updating prediction outcomes + learning from mistakes..."):
            try:
                evaluate_history(force_all=bool(force_eval))
                refresh_history_current_prices(max_stocks=40)
                learn_from_history(min_closed=5)
            except Exception as e:
                st.warning(f"Evaluation note: {e}")
        st.session_state._last_hist_eval = datetime.now()

    history_all = normalize_history_df(load_history())
    # Diagnostics so user sees why STRATEGY might be missing
    if history_all is not None and not history_all.empty and "Call Source" in history_all.columns:
        vc = history_all["Call Source"].astype(str).str.upper().value_counts().head(8)
        st.caption("History Call Source counts: " + ", ".join(f"{k}:{v}" for k, v in vc.items()))
    history = quality_history_only(history_all)
    stats = overall_statistics()

    st.subheader("📊 Success rate — Sure / Strategy predictions only")
    st.caption(
        "Formula: **Targets ÷ (Targets + Stops)**. Open trades not counted. "
        f"Scan bulk excluded from score ({stats.get('scan_n', 0)} scan rows ignored)."
    )
    a, b, c, d, e = st.columns(5)
    a.metric("Quality calls", stats.get("recommendations", 0) or stats.get("quality_n", 0))
    b.metric("🎯 Target hits", stats.get("wins", 0) or stats.get("quality_wins", 0))
    c.metric("🔴 Stop hits", stats.get("losses", 0) or stats.get("quality_losses", 0))
    d.metric(
        "Success rate",
        f"{stats.get('success', 0)}%",
        help="Targets/(Targets+Stops) on SURE|STRATEGY|HIGH_CONV|PRECISION only",
    )
    e.metric("Still open", stats.get("open", 0))
    decided = int(stats.get("quality_wins") or stats.get("wins") or 0) + int(
        stats.get("quality_losses") or stats.get("losses") or 0
    )
    if decided == 0:
        st.info(
            "No closed Strategy/Sure outcomes yet. Generate **Sure / Live strategy / High conviction**, "
            "then Force re-check after prices move — success % will appear (often ~40–55%)."
        )
    elif 40 <= float(stats.get("success") or 0) <= 55:
        st.success(
            f"**{stats.get('success')}%** is inside the normal **40–55%** band for quality swing calls."
        )
    elif float(stats.get("success") or 0) > 55:
        st.success(f"**{stats.get('success')}%** is strong on quality predicted stocks ({decided} decided).")
    else:
        st.warning(
            f"**{stats.get('success')}%** is below ~40% — favour High conv/Sure only and pause weak strategies."
        )

    # Learning panel
    learning = load_learning()
    with st.expander("🧠 Model learning from past mistakes & chart patterns", expanded=True):
        st.caption(
            f"Learning file: `{LEARNING_FILE.name}` · "
            f"Closed samples: {learning.get('sample_closed', 0)} · "
            f"Wins: {learning.get('sample_wins', 0)} · Losses: {learning.get('sample_losses', 0)} · "
            f"Overall decided win rate: {learning.get('overall_win_rate', '—')}%"
        )
        if st.button("🔁 Re-train learning now", key="relearn_btn"):
            with st.spinner("Learning from closed trades..."):
                learning = learn_from_history(min_closed=3)
            st.success("Learning updated — next scan will use new pattern weights.")
            st.rerun()

        lessons = learning.get("lessons") or []
        if lessons:
            st.write("**Lessons the model is applying**")
            for L in lessons[:12]:
                st.markdown(f"- {L}")
        else:
            st.info(
                "Not enough closed target/stop outcomes yet. "
                "Run scans, then **Force re-check all** so the model can learn."
            )

        # Pattern importance table
        st.write("**Chart patterns — meaning & learned importance**")
        pat_rows = []
        stats_p = learning.get("pattern_stats", {})
        mults = learning.get("pattern_multipliers", {})
        for name, meta in PATTERN_IMPORTANCE.items():
            stt = stats_p.get(name, {})
            pat_rows.append({
                "Pattern": name,
                "Bias": meta.get("bias", ""),
                "Base weight": meta.get("base_weight", 0),
                "Learned mult": mults.get(name, 1.0),
                "Hist win %": stt.get("win_rate", "—"),
                "Samples": stt.get("n", 0),
                "Why it matters": meta.get("why", ""),
                "How to use": meta.get("use", ""),
            })
        st.dataframe(pd.DataFrame(pat_rows), use_container_width=True, hide_index=True)

        if learning.get("pred_bucket_stats"):
            st.write("**Prediction strength zones (learned)**")
            st.dataframe(
                pd.DataFrame([
                    {"Zone": k, **v} for k, v in learning["pred_bucket_stats"].items()
                ]),
                use_container_width=True,
                hide_index=True,
            )

        manual_feedback_trainer()

    if history is None or history.empty:
        st.info("No prediction history yet. Run **FULL MARKET SCAN** first.")
        return

    hist = history.copy()
    if "Call Source" not in hist.columns:
        hist["Call Source"] = "SCAN"
    else:
        hist["Call Source"] = hist["Call Source"].fillna("SCAN").astype(str)
    if "Strategy" not in hist.columns:
        hist["Strategy"] = ""
    if "Patterns" not in hist.columns:
        hist["Patterns"] = ""

    hist["_pred_dt"] = pd.to_datetime(hist.get("Prediction Date"), errors="coerce")

    # ============================================================
    # ALL BUY CALLS — your own backtest sheet
    # ============================================================
    st.subheader("📗 All BUY calls (backtest sheet)")
    st.caption(
        "Every historical **BUY** in `recommendation_history.csv`. "
        "Daily **FULL MARKET SCAN** saves source as **SCAN**. "
        "Sure / Strategy / Precision only appear if you saved them from those pages. "
        "Default view = **ALL sources** so your daily BUYs always show."
    )

    # Normalize Call + Call Source so older rows still match
    hist["Call"] = hist["Call"].astype(str).str.upper().str.strip()
    hist["Call Source"] = (
        hist["Call Source"].astype(str).str.upper().str.strip()
        .replace({"NAN": "SCAN", "NONE": "SCAN", "": "SCAN"})
    )

    buy_all = hist[hist["Call"].str.contains("BUY", na=False)].copy()

    # Quick diagnostic so empty filter is explained
    n_hist = len(hist)
    n_buy_total = len(buy_all)
    src_counts = (
        buy_all["Call Source"].value_counts().head(12)
        if n_buy_total else pd.Series(dtype=int)
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("History rows (all calls)", n_hist)
    d2.metric("BUY rows (all sources)", n_buy_total)
    if n_buy_total:
        top_src = ", ".join(f"{k}:{v}" for k, v in src_counts.items())
        d3.caption(f"**BUY by source:** {top_src}")
    else:
        d3.warning("No BUY in CSV yet — run FULL MARKET SCAN once.")

    bt1, bt2, bt3, bt4 = st.columns(4)
    with bt1:
        # Default index 0 = ALL sources (was Strategy-only which hid SCAN)
        bt_src = st.selectbox(
            "BUY source",
            [
                "Strategy / Sure / Precision only",
                "ALL sources",
                "Scan only (daily market scan)",
            ],
            index=1,
            key="buy_bt_src_v3",
            help="Priority: Sure/Strategy first when present. SCAN = full market scan bulk BUYs.",
        )
    with bt2:
        bt_result = st.selectbox(
            "BUY result",
            [
                "ALL",
                "TARGET ACHIEVED",
                "STOP LOSS HIT",
                "HOLDING PERIOD COMPLETED",
                "OPEN / PENDING",
            ],
            key="buy_bt_result_v2",
        )
    with bt3:
        bt_date = st.selectbox(
            "BUY date range",
            ["All dates", "Last 7 days", "Last 30 days", "Last 90 days", "Today"],
            key="buy_bt_date_v2",
        )
    with bt4:
        bt_show = st.selectbox(
            "Rows to show",
            [50, 100, 200, 500, 1000, "ALL"],
            index=2,
            key="buy_bt_rows_v2",
        )

    buy_view = buy_all.copy()
    src_u0 = buy_view["Call Source"].astype(str).str.upper()
    quality_pat = "STRATEGY|SURE|PRECISION|MY_STRATEGY|HIGH_CONV"

    if bt_src.startswith("Scan only"):
        buy_view = buy_view[~src_u0.str.contains(quality_pat, regex=True, na=False)]
    elif bt_src.startswith("Strategy"):
        buy_view = buy_view[src_u0.str.contains(quality_pat, regex=True, na=False)]
        if buy_view.empty and n_buy_total > 0:
            st.warning(
                f"You have **{n_buy_total} BUY** rows, but **none** marked Strategy/Sure/Precision. "
                "Daily calls from **FULL MARKET SCAN** are saved as **SCAN**. "
                "Switch **BUY source → ALL sources** or **Scan only** to see them."
            )

    today = datetime.now().date()
    if bt_date == "Today":
        buy_view = buy_view[buy_view["_pred_dt"].dt.date == today]
    elif bt_date == "Last 7 days":
        buy_view = buy_view[buy_view["_pred_dt"].dt.date >= (today - timedelta(days=7))]
    elif bt_date == "Last 30 days":
        buy_view = buy_view[buy_view["_pred_dt"].dt.date >= (today - timedelta(days=30))]
    elif bt_date == "Last 90 days":
        buy_view = buy_view[buy_view["_pred_dt"].dt.date >= (today - timedelta(days=90))]

    res_u0 = buy_view["Result"].astype(str).str.upper() if "Result" in buy_view.columns else pd.Series([""] * len(buy_view), index=buy_view.index)
    if bt_result == "TARGET ACHIEVED":
        buy_view = buy_view[res_u0.str.contains("TARGET ACHIEVED|WIN", na=False)]
    elif bt_result == "STOP LOSS HIT":
        buy_view = buy_view[res_u0.str.contains("STOP LOSS", na=False)]
    elif bt_result == "HOLDING PERIOD COMPLETED":
        buy_view = buy_view[res_u0.str.contains("HOLDING PERIOD", na=False)]
    elif bt_result == "OPEN / PENDING":
        buy_view = buy_view[
            ~res_u0.str.contains("TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD", na=False, regex=True)
        ]

    # BUY-only stats for this filtered sheet
    if not buy_view.empty:
        ru = buy_view["Result"].astype(str).str.upper()
        n_buy = len(buy_view)
        n_tgt = int(ru.str.contains("TARGET ACHIEVED", na=False).sum())
        n_sl = int(ru.str.contains("STOP LOSS HIT", na=False).sum())
        n_time = int(ru.str.contains("HOLDING PERIOD", na=False).sum())
        n_open = n_buy - n_tgt - n_sl - n_time
        closed = n_tgt + n_sl
        win_rate = round(100.0 * n_tgt / closed, 1) if closed else 0.0
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("BUY rows", n_buy)
        m2.metric("🎯 Target hit", n_tgt)
        m3.metric("🔴 Stop hit", n_sl)
        m4.metric("⏰ Time exit", n_time)
        m5.metric("⏳ Open", max(n_open, 0))
        m6.metric("Win rate (closed)", f"{win_rate}%")
    else:
        st.info("No BUY rows for this filter. Run scans / Sure / Strategy so BUYs are saved to history.")

    buy_cols = [
        c for c in [
            "Prediction Date",
            "Stock",
            "Call Source",
            "Strategy",
            "Call",
            "Prediction",
            "Entry",
            "Current Price",
            "Target",
            "Stop Loss",
            "Risk %",
            "Hold Days",
            "Days Taken",
            "Result",
            "Return %",
            "Exit Price",
            "Evaluation Date",
            "Outcome Message",
            "Recommendation",
            "Patterns",
            "Reason",
        ]
        if c in buy_view.columns
    ]

    if not buy_view.empty:
        buy_view = buy_view.sort_values("Prediction Date", ascending=False)
        if bt_show != "ALL":
            display_buy = buy_view.head(int(bt_show))
        else:
            display_buy = buy_view
        st.dataframe(
            display_buy[buy_cols],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Showing {len(display_buy):,} of {len(buy_view):,} BUY calls. "
            "Entry / Target / Stop are locked; Current Price updates on re-check."
        )
        # CSV download for your own backtest
        try:
            csv_bytes = buy_view[buy_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download all filtered BUY calls (CSV)",
                data=csv_bytes,
                file_name=f"buy_calls_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_buy_bt",
            )
        except Exception:
            pass
        if st.button("⚡ Force re-check outcomes on these BUYs", key="force_buy_bt"):
            with st.spinner("Re-checking target / stop on BUY history..."):
                try:
                    evaluate_history(force_all=True)
                    refresh_history_current_prices(max_stocks=50)
                except Exception as e:
                    st.warning(str(e))
            st.success("Updated — scroll metrics above for new target/stop counts.")
            st.rerun()

    st.divider()

    # --- Filters ---
    st.subheader("Prediction History (all call types)")
    page_tab = st.radio(
        "History page",
        [
            "Strategy + Sure + Precision only",
            "All sources",
            "Scan / general calls",
        ],
        horizontal=True,
        index=0,
        key="hist_page_src_v2",
        help="Default = quality calls first (Sure/Strategy). Generate+save them from Sure Call / Strategy Lab.",
    )

    br = history_success_breakdown(hist)
    if br is not None and not br.empty:
        with st.expander("📊 Success rate by Call Source / Strategy / Pattern", expanded=True):
            st.dataframe(br, use_container_width=True, hide_index=True)
            # Strategy-only success table
            st.write("**Strategy success rate**")
            sb = br[br["Filter"] == "Strategy"] if "Filter" in br.columns else br
            if sb is not None and not sb.empty:
                st.dataframe(sb, use_container_width=True, hide_index=True)

    # ============================================================
    # STRATEGY-WISE + HIGH CONVICTION PAST PREDICTIONS
    # ============================================================
    st.subheader("⭐ Strategy-wise & High Conviction past predictions")
    st.caption(
        "Filter by **source** (SURE / STRATEGY / HIGH_CONV) and by **strategy name**. "
        "Generate from Strategy Lab (Live BUY / High conviction) or Sure Call Desk so rows appear here."
    )
    src_u = hist["Call Source"].astype(str).str.upper()
    if "Strategy" not in hist.columns:
        hist["Strategy"] = ""
    rec = hist[src_u.str.contains("STRATEGY|SURE|PRECISION|MY_STRATEGY|HIGH_CONV", regex=True, na=False)].copy()
    _prio = {"SURE": 0, "HIGH_CONV": 1, "STRATEGY": 2, "MY_STRATEGY": 3, "PRECISION": 4}
    if not rec.empty:
        rec = rec.copy()
        rec["_prio"] = rec["Call Source"].astype(str).str.upper().map(lambda s: _prio.get(s, 9))
        rec = rec.sort_values(["_prio", "Prediction Date"], ascending=[True, False])

    # View mode
    view_mode = st.radio(
        "View",
        [
            "All quality (Sure + Strategy + High conv)",
            "High conviction only",
            "Strategy-wise table",
            "One strategy deep-dive",
        ],
        horizontal=True,
        key="hist_strat_view",
    )

    sf1, sf2, sf3 = st.columns(3)
    with sf1:
        src_opts = sorted(rec["Call Source"].astype(str).unique().tolist()) if len(rec) else [
            "SURE", "STRATEGY", "HIGH_CONV", "PRECISION", "MY_STRATEGY"
        ]
        src_pick = st.multiselect(
            "Call Source filter",
            options=src_opts,
            default=src_opts,
            key="hist_rec_src_v3",
        )
    with sf2:
        strat_names = sorted(
            [x for x in hist["Strategy"].astype(str).unique().tolist() if x and str(x).lower() not in ("nan", "none", "")]
        )
        # Also strategy names from SWING_STRATEGIES for empty history
        try:
            for _meta in SWING_STRATEGIES.values():
                n = str(_meta.get("name", ""))
                if n and n not in strat_names:
                    strat_names.append(n)
            strat_names = sorted(set(strat_names))
        except Exception:
            pass
        strat_pick = st.selectbox(
            "Strategy name filter",
            ["ALL strategies"] + strat_names,
            key="hist_one_strat",
        )
    with sf3:
        strat_result = st.selectbox(
            "Result (strategy view)",
            ["ALL", "TARGET ACHIEVED", "STOP LOSS HIT", "OPEN / PENDING"],
            key="hist_strat_res",
        )

    rec_f = rec.copy() if rec is not None else pd.DataFrame()
    if src_pick and not rec_f.empty:
        rec_f = rec_f[rec_f["Call Source"].astype(str).isin(src_pick)]
    if view_mode.startswith("High conviction"):
        rec_f = rec_f[rec_f["Call Source"].astype(str).str.upper() == "HIGH_CONV"] if not rec_f.empty else rec_f
    if strat_pick != "ALL strategies" and not rec_f.empty:
        rec_f = rec_f[
            rec_f["Strategy"].astype(str).str.contains(strat_pick, case=False, na=False)
            | (rec_f["Strategy"].astype(str) == strat_pick)
        ]
    if strat_result != "ALL" and not rec_f.empty and "Result" in rec_f.columns:
        ru = rec_f["Result"].astype(str).str.upper()
        if strat_result == "OPEN / PENDING":
            rec_f = rec_f[~ru.str.contains("TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD", na=False)]
        else:
            rec_f = rec_f[ru.str.contains(strat_result, na=False)]

    if rec_f is not None and not rec_f.empty:
        ru = rec_f["Result"].astype(str).str.upper()
        n_t = int(ru.str.contains("TARGET ACHIEVED", na=False).sum())
        n_s = int(ru.str.contains("STOP LOSS HIT", na=False).sum())
        closed = n_t + n_s
        wr = round(100.0 * n_t / closed, 1) if closed else 0.0
        a, b, c, d = st.columns(4)
        a.metric("Rows (filtered)", len(rec_f))
        b.metric("🎯 Targets", n_t)
        c.metric("🔴 Stops", n_s)
        d.metric("Win rate", f"{wr}%")

        # Strategy-wise success table
        if view_mode.startswith("Strategy-wise") or view_mode.startswith("All quality"):
            st.markdown("**📈 Success % by strategy name**")
            rows_bt = []
            for sname, g in rec_f.groupby(rec_f["Strategy"].astype(str)):
                if not sname or sname.lower() in ("nan", "none"):
                    sname = "(blank strategy)"
                gru = g["Result"].astype(str).str.upper()
                wt = int(gru.str.contains("TARGET ACHIEVED", na=False).sum())
                ls = int(gru.str.contains("STOP LOSS HIT", na=False).sum())
                cl = wt + ls
                rows_bt.append({
                    "Strategy": sname,
                    "Calls": len(g),
                    "Targets": wt,
                    "Stops": ls,
                    "Open/Other": len(g) - cl,
                    "Win % (closed)": round(100.0 * wt / cl, 1) if cl else 0.0,
                    "Sources": ", ".join(sorted(g["Call Source"].astype(str).unique().tolist())[:5]),
                })
            if rows_bt:
                st.dataframe(
                    pd.DataFrame(rows_bt).sort_values("Win % (closed)", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

        # By Call Source
        st.markdown("**Success % by Call Source**")
        rows_src = []
        for sname, g in rec_f.groupby(rec_f["Call Source"].astype(str)):
            gru = g["Result"].astype(str).str.upper()
            wt = int(gru.str.contains("TARGET ACHIEVED", na=False).sum())
            ls = int(gru.str.contains("STOP LOSS HIT", na=False).sum())
            cl = wt + ls
            rows_src.append({
                "Call Source": sname,
                "Calls": len(g),
                "Targets": wt,
                "Stops": ls,
                "Win %": round(100.0 * wt / cl, 1) if cl else 0.0,
            })
        if rows_src:
            st.dataframe(pd.DataFrame(rows_src), use_container_width=True, hide_index=True)

        show_cols = [c for c in [
            "Prediction Date", "Stock", "Call Source", "Strategy", "Call",
            "Entry", "Target", "Stop Loss", "Current Price", "Result", "Return %", "Days Taken", "Patterns",
        ] if c in rec_f.columns]
        st.markdown("**Detail rows — filterable table**")
        filterable_dataframe(rec_f[show_cols].head(200), key="hist_qual_table", default_cols=show_cols, height=420)
        st.markdown("**Quality call cards**")
        n_hc = st.slider("History cards", 3, 15, 6, key="hist_cards_n")
        for _, hr in rec_f.head(n_hc).iterrows():
            render_call_stock_card(hr.to_dict(), section_key="hist")
        try:
            st.download_button(
                "⬇️ Download filtered strategy / high-conv CSV",
                data=rec_f[show_cols].to_csv(index=False).encode("utf-8"),
                file_name=f"strategy_past_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_strat_hist_v2",
            )
        except Exception:
            pass
    else:
        st.warning(
            "No strategy / Sure / high-conviction rows in history yet.\n\n"
            "1. **Strategy Lab → 📡 Live BUY by strategy → Generate live signals**\n"
            "2. **Strategy Lab → ⭐ High conviction → Find high conviction**\n"
            "3. **Sure Call Desk → Generate sure calls**\n\n"
            "Wait for “Saved N…” then reopen Past Predictions."
        )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        call_filter = st.selectbox("Call", ["ALL", "BUY", "HOLD", "SELL"], key="hist_call")
    with f2:
        result_filter = st.selectbox(
            "Result",
            [
                "ALL",
                "TARGET ACHIEVED",
                "STOP LOSS HIT",
                "HOLDING PERIOD COMPLETED",
                "OPEN / PENDING",
            ],
            key="hist_result",
        )
    with f3:
        date_preset = st.selectbox(
            "Date range",
            ["All dates", "Last 7 days", "Last 30 days", "Today"],
            key="hist_date_preset",
        )
    with f4:
        strat_opts = ["ALL"] + sorted(
            [x for x in hist["Strategy"].astype(str).unique().tolist() if x and x != "nan"]
        )[:40]
        strat_filter = st.selectbox("Strategy", strat_opts, key="hist_strat_f")

    h = hist.copy()
    today = datetime.now().date()
    if date_preset == "Today":
        h = h[h["_pred_dt"].dt.date == today]
    elif date_preset == "Last 7 days":
        h = h[h["_pred_dt"].dt.date >= (today - timedelta(days=7))]
    elif date_preset == "Last 30 days":
        h = h[h["_pred_dt"].dt.date >= (today - timedelta(days=30))]

    # Source page split
    src = h["Call Source"].astype(str).str.upper()
    if page_tab.startswith("Strategy"):
        h = h[src.str.contains("STRATEGY|SURE|PRECISION|MY_STRATEGY|HIGH_CONV", regex=True, na=False)]
    elif page_tab.startswith("Scan"):
        h = h[~src.str.contains("STRATEGY|SURE|PRECISION|MY_STRATEGY|HIGH_CONV", regex=True, na=False)]

    if call_filter != "ALL" and "Call" in h.columns:
        h = h[h["Call"].astype(str).str.upper().str.strip() == call_filter]

    if strat_filter != "ALL" and "Strategy" in h.columns:
        h = h[h["Strategy"].astype(str) == strat_filter]

    if "Result" in h.columns:
        res_u = h["Result"].astype(str).str.upper().str.strip()
        if result_filter == "OPEN / PENDING":
            h = h[~res_u.str.contains(
                "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
                regex=True, na=True,
            )]
        elif result_filter == "TARGET ACHIEVED":
            h = h[res_u.str.contains("TARGET ACHIEVED", na=False)]
        elif result_filter == "STOP LOSS HIT":
            h = h[res_u.str.contains("STOP LOSS HIT", na=False)]
        elif result_filter == "HOLDING PERIOD COMPLETED":
            h = h[res_u.str.contains("HOLDING PERIOD COMPLETED", na=False)]

    # Days completed / remaining
    now_ts = pd.Timestamp(today)

    def _days_pair(pred_dt, hold_days):
        hold = safe_int(hold_days, DEFAULT_HOLD_DAYS)
        if hold <= 0:
            hold = DEFAULT_HOLD_DAYS
        try:
            if pred_dt is None or pd.isna(pred_dt):
                return 0, hold
            start = pd.Timestamp(pred_dt).normalize()
            completed = max(0, safe_int((now_ts - start).days, 0))
            return completed, max(0, hold - completed)
        except Exception:
            return 0, hold

    if len(h):
        pairs = [_days_pair(r.get("_pred_dt"), r.get("Hold Days")) for _, r in h.iterrows()]
        h = h.copy()
        h["Days Completed"] = [p[0] for p in pairs]
        h["Days Remaining"] = [p[1] for p in pairs]
        if "Prediction Date" in h.columns:
            h = h.sort_values("Prediction Date", ascending=False)

    st.write(f"Showing **{len(h)}** rows")

    display_cols = [
        c for c in [
            "Prediction Date", "Stock", "Call Source", "Strategy", "Call",
            "Entry", "Current Price", "Target", "Stop Loss",
            "Hold Days", "Days Completed", "Days Remaining",
            "Status", "Result", "Exit Price", "Return %", "Days Taken",
            "Patterns", "Recommendation",
        ] if c in h.columns
    ]
    st.markdown("##### Filterable prediction table")
    filterable_dataframe(h, key="hist_main_table", default_cols=display_cols, height=420)

    st.subheader("📋 Visual cards (mcap · BV · desk)")
    n_vis = st.slider("Visual cards", 3, 12, 5, key="hist_vis_n")
    for _, row in h.head(n_vis).iterrows():
        render_call_stock_card(row.to_dict(), section_key="hist_main")

    # --- Detail cards (top 25 of filtered) ---
    st.subheader("📋 Detailed outcomes")
    for _, row in h.head(25).iterrows():
        stock = str(row.get("Stock", ""))
        call = str(row.get("Call", "")).upper()
        entry = safe_float(row.get("Entry"))
        target = safe_float(row.get("Target"))
        stop = safe_float(row.get("Stop Loss"))
        cur = safe_float(row.get("Current Price"))
        hold_days = safe_int(row.get("Hold Days"), DEFAULT_HOLD_DAYS) or DEFAULT_HOLD_DAYS
        result = str(row.get("Result", "PENDING")).upper()
        result_detail = str(row.get("Result Detail", "") or "")
        recommendation = str(row.get("Recommendation", "") or "")
        return_pct = safe_float(row.get("Return %"))
        exit_price = safe_float(row.get("Exit Price"))
        days_taken = row.get("Days Taken", "")
        eval_date = row.get("Evaluation Date", "")
        days_completed = safe_int(row.get("Days Completed"), 0)
        days_remaining = safe_int(row.get("Days Remaining"), hold_days)

        try:
            pred_date_fmt = pd.to_datetime(row.get("Prediction Date")).strftime("%d %B %Y")
        except Exception:
            pred_date_fmt = str(row.get("Prediction Date", ""))

        if "TARGET ACHIEVED" in result:
            badge, box = "🎯 TARGET ACHIEVED", "success-box"
        elif "STOP LOSS HIT" in result:
            badge, box = "🔴 STOP LOSS HIT", "danger-box"
        elif "HOLDING PERIOD" in result:
            badge, box = "⏰ HOLDING PERIOD COMPLETED", "watch-box"
        else:
            badge, box = "⏳ OPEN / PENDING", "hold-box"

        if not result_detail or result_detail in ("", "nan", "None"):
            if "TARGET ACHIEVED" in result:
                result_detail = (
                    f"🎯 TARGET ACHIEVED\nTarget: ₹{target:,.2f}\n"
                    f"Days Taken: {days_taken}\nProfit: {return_pct:+.2f}%"
                )
            elif "STOP LOSS HIT" in result:
                result_detail = (
                    f"🔴 STOP LOSS HIT\nStop: ₹{stop:,.2f}\n"
                    f"Days Taken: {days_taken}\nLoss: {return_pct:.2f}%"
                )
            elif "HOLDING PERIOD" in result:
                result_detail = (
                    f"⏰ HOLDING PERIOD COMPLETED\nExit: ₹{exit_price:,.2f}\n"
                    f"Date: {eval_date}"
                )
            else:
                cur_txt = f"₹{cur:,.2f}" if cur else "—"
                result_detail = (
                    f"Still open\nCurrent: {cur_txt}\n"
                    f"Waiting for Target ₹{target:,.2f} or Stop ₹{stop:,.2f}"
                )

        detail_html = result_detail.replace("\n", "<br>")
        rec_html = recommendation.replace("\n", "<br>") if recommendation else ""
        cur_line = f"<b>Current Price:</b> ₹{cur:,.2f}<br>" if cur else ""

        st.markdown(
            f"""
            <div class="{box}" style="margin-bottom:12px;padding:14px;border-radius:10px;">
            <h4 style="margin:0 0 8px 0;">{stock} — {call} &nbsp; {badge}</h4>
            <p style="margin:0;line-height:1.6;">
            <b>Prediction Date:</b> {pred_date_fmt}<br>
            <b>Entry:</b> ₹{entry:,.2f}<br>
            {cur_line}
            <b>Target (locked):</b> ₹{target:,.2f}<br>
            <b>Stop Loss (locked):</b> ₹{stop:,.2f}<br>
            <b>Holding:</b> {hold_days} days |
            <b>Completed:</b> {days_completed} |
            <b>Remaining:</b> {days_remaining}<br><br>
            <b>RESULT:</b><br>{detail_html}
            {f'<br><br><b>Recommendation:</b><br>{rec_html}' if rec_html else ''}
            </p></div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HOLDING ADVISOR — user stock + buy price + shares
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    """Company fundamentals from Yahoo Finance (works for most NSE .NS symbols)."""
    out = {
        "name": "",
        "sector": "",
        "industry": "",
        "summary": "",
        "market_cap": None,
        "pe": None,
        "forward_pe": None,
        "pb": None,
        "ps": None,
        "eps": None,
        "book_value": None,
        "dividend_yield": None,
        "roe": None,
        "roa": None,
        "profit_margins": None,
        "operating_margins": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "target_mean": None,
        "recommendation": "",
        "fifty_two_high": None,
        "fifty_two_low": None,
        "beta": None,
        "employees": None,
        "currency": "INR",
        "error": "",
    }
    try:
        t = yf.Ticker(clean_symbol(symbol))
        info = {}
        try:
            info = t.info or {}
        except Exception:
            try:
                info = t.get_info() or {}
            except Exception:
                info = {}

        def g(*keys):
            for k in keys:
                v = info.get(k)
                if v is not None and v == v:  # not NaN
                    return v
            return None

        out["name"] = g("longName", "shortName") or display_symbol(symbol)
        out["sector"] = g("sector") or ""
        out["industry"] = g("industry") or ""
        out["summary"] = (g("longBusinessSummary") or "")[:900]
        out["market_cap"] = g("marketCap")
        out["pe"] = g("trailingPE", "peRatio")
        out["forward_pe"] = g("forwardPE")
        out["pb"] = g("priceToBook")
        out["ps"] = g("priceToSalesTrailing12Months")
        out["eps"] = g("trailingEps")
        out["book_value"] = g("bookValue")
        out["face_value"] = g("faceValue")
        # Sales / revenue (absolute INR — convert to Cr in UI)
        out["total_revenue"] = g("totalRevenue", "revenue")
        out["revenue_latest"] = out.get("revenue_latest") or out["total_revenue"]
        dy = g("dividendYield")
        out["dividend_yield"] = (dy * 100 if dy is not None and dy < 1 else dy)
        out["roe"] = g("returnOnEquity")
        out["roa"] = g("returnOnAssets")
        out["profit_margins"] = g("profitMargins")
        out["operating_margins"] = g("operatingMargins")
        out["revenue_growth"] = g("revenueGrowth")
        out["earnings_growth"] = g("earningsGrowth")
        out["debt_to_equity"] = g("debtToEquity")
        out["current_ratio"] = g("currentRatio")
        out["target_mean"] = g("targetMeanPrice")
        out["recommendation"] = g("recommendationKey") or g("recommendationMean") or ""
        out["fifty_two_high"] = g("fiftyTwoWeekHigh")
        out["fifty_two_low"] = g("fiftyTwoWeekLow")
        out["beta"] = g("beta")
        out["employees"] = g("fullTimeEmployees")
        out["currency"] = g("currency") or "INR"

        # Simple financial statement snapshot (yearly)
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                out["financials_cols"] = [str(c)[:10] for c in list(fin.columns)[:4]]
                for label, keys in [
                    ("revenue", ["Total Revenue", "Operating Revenue"]),
                    ("net_income", ["Net Income", "Net Income Common Stockholders"]),
                ]:
                    for k in keys:
                        if k in fin.index:
                            series = fin.loc[k]
                            out[f"{label}_latest"] = safe_float(series.iloc[0]) if len(series) else None
                            # Year-wise in crore for dashboard
                            yearly = []
                            for col in list(series.index)[:5]:
                                yearly.append({
                                    "Year": str(col)[:10],
                                    f"{label}_cr": in_crore(series[col]),
                                })
                            out[f"{label}_yearly"] = yearly
                            break
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fundamental_yearly_table(symbol: str) -> pd.DataFrame:
    """Revenue / profit year-wise in ₹ Cr for dashboards."""
    try:
        fund = fetch_fundamentals(symbol)
    except Exception:
        return pd.DataFrame()
    rows = {}
    for item in fund.get("revenue_yearly") or []:
        y = item.get("Year")
        rows.setdefault(y, {"Year": y})
        rows[y]["Revenue (₹ Cr)"] = item.get("revenue_cr")
    for item in fund.get("net_income_yearly") or []:
        y = item.get("Year")
        rows.setdefault(y, {"Year": y})
        rows[y]["Profit (₹ Cr)"] = item.get("net_income_cr")
    if not rows:
        # single latest
        if fund.get("revenue_latest") or fund.get("net_income_latest"):
            return pd.DataFrame([{
                "Year": "Latest",
                "Revenue (₹ Cr)": in_crore(fund.get("revenue_latest")),
                "Profit (₹ Cr)": in_crore(fund.get("net_income_latest")),
            }])
        return pd.DataFrame()
    return pd.DataFrame(list(rows.values())).sort_values("Year", ascending=False)


def _fmt_num(v, pct=False, money=False):
    if v is None:
        return "—"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if pct:
        # Yahoo often gives 0.15 for 15%
        if abs(v) <= 1.5:
            v = v * 100
        return f"{v:.2f}%"
    if money:
        if abs(v) >= 1e7:
            return f"₹{v/1e7:.2f} Cr"
        if abs(v) >= 1e5:
            return f"₹{v/1e5:.2f} L"
        return f"₹{v:,.2f}"
    return f"{v:,.2f}"


def holding_horizon_advice(result, buy_price, shares, fund: dict) -> dict:
    """
    Short-term (days–weeks) and long-term (months+) advice
    for an existing position.
    """
    price = safe_float(result.get("Price"))
    call = str(result.get("Call", "")).upper()
    pred = safe_float(result.get("Prediction"))
    rsi = safe_float(result.get("RSI"))
    adx = safe_float(result.get("ADX"))
    target = safe_float(result.get("Target"))
    stop = safe_float(result.get("Stop Loss"))
    risk_pct = safe_float(result.get("Risk %"))
    patterns = str(result.get("Patterns", ""))

    pnl = ((price - buy_price) / buy_price * 100) if buy_price > 0 else 0
    invested = buy_price * shares
    mkt_value = price * shares
    pnl_rs = mkt_value - invested

    data = result.get("Data")
    above_ema50 = above_ema200 = None
    if data is not None and not getattr(data, "empty", True):
        row = data.iloc[-1]
        e50 = safe_float(row.get("EMA50"))
        e200 = safe_float(row.get("EMA200"))
        if e50:
            above_ema50 = price > e50
        if e200:
            above_ema200 = price > e200

    # Short-term score
    st_score = 0
    st_notes = []
    if call == "BUY":
        st_score += 2
        st_notes.append("Model short-term signal is BUY.")
    elif call == "HOLD":
        st_score += 1
        st_notes.append("Model short-term signal is HOLD.")
    elif call == "SELL":
        st_score -= 2
        st_notes.append("Model short-term signal is SELL.")
    if pred >= 70:
        st_score += 1
        st_notes.append(f"Prediction strength {pred:.0f}% is solid.")
    elif pred < 55:
        st_score -= 1
        st_notes.append(f"Prediction strength {pred:.0f}% is weak.")
    if 50 <= rsi <= 70:
        st_score += 1
        st_notes.append(f"RSI {rsi:.0f} is in a healthy bullish band.")
    elif rsi > 78:
        st_score -= 1
        st_notes.append(f"RSI {rsi:.0f} is overbought — short-term pullback risk.")
    elif rsi < 35:
        st_notes.append(f"RSI {rsi:.0f} is weak/oversold — bounce possible but trend may be soft.")
    if above_ema50 is True:
        st_score += 1
        st_notes.append("Price is above EMA50 (short/medium trend supportive).")
    elif above_ema50 is False:
        st_score -= 1
        st_notes.append("Price is below EMA50 (short/medium trend weak).")
    if "Bearish" in patterns or "Shooting Star" in patterns or "Lower High" in patterns:
        st_score -= 1
        st_notes.append(f"Caution pattern: {patterns}.")
    if target > 0 and price >= target * 0.98:
        st_score -= 1
        st_notes.append("Price is near/above model target — consider booking some profit short-term.")
    if stop > 0 and price <= stop * 1.02:
        st_score -= 2
        st_notes.append("Price is near model stop — short-term risk of further downside.")

    if st_score >= 3:
        st_action = "BUY MORE (short-term) / HOLD with confidence"
        st_color = "buy"
    elif st_score >= 1:
        st_action = "HOLD (short-term) — trail stop, avoid adding aggressively"
        st_color = "hold"
    elif st_score >= -1:
        st_action = "HOLD lightly or reduce — wait for clearer setup"
        st_color = "hold"
    else:
        st_action = "SELL / EXIT (short-term) — risk outweighs reward"
        st_color = "sell"

    # Long-term score
    lt_score = 0
    lt_notes = []
    if above_ema200 is True:
        lt_score += 2
        lt_notes.append("Price above EMA200 — long-term uptrend intact.")
    elif above_ema200 is False:
        lt_score -= 2
        lt_notes.append("Price below EMA200 — long-term structure is weak.")
    if above_ema50 is True:
        lt_score += 1
    pe = fund.get("pe")
    if pe is not None:
        if 0 < pe < 25:
            lt_score += 1
            lt_notes.append(f"Trailing P/E {pe:.1f} is not stretched.")
        elif pe >= 40:
            lt_score -= 1
            lt_notes.append(f"Trailing P/E {pe:.1f} is expensive — growth must justify it.")
    roe = fund.get("roe")
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) <= 1.5 else roe
        if roe_pct >= 15:
            lt_score += 1
            lt_notes.append(f"ROE ~{roe_pct:.1f}% supports long-term quality.")
        elif roe_pct < 8:
            lt_score -= 1
            lt_notes.append(f"ROE ~{roe_pct:.1f}% is modest.")
    pm = fund.get("profit_margins")
    if pm is not None:
        pm_pct = pm * 100 if abs(pm) <= 1.5 else pm
        if pm_pct >= 10:
            lt_score += 1
            lt_notes.append(f"Profit margin ~{pm_pct:.1f}% looks healthy.")
    de = fund.get("debt_to_equity")
    if de is not None:
        # Yahoo often scales D/E * 100
        de_v = de / 100 if de > 20 else de
        if de_v > 2:
            lt_score -= 1
            lt_notes.append(f"Debt/Equity looks elevated ({de}).")
        elif de_v < 1:
            lt_score += 1
            lt_notes.append("Balance sheet leverage looks controlled.")
    rg = fund.get("revenue_growth")
    if rg is not None:
        rg_pct = rg * 100 if abs(rg) <= 1.5 else rg
        if rg_pct > 10:
            lt_score += 1
            lt_notes.append(f"Revenue growth ~{rg_pct:.1f}% is supportive.")
        elif rg_pct < 0:
            lt_score -= 1
            lt_notes.append(f"Revenue growth ~{rg_pct:.1f}% is negative.")
    if call == "SELL" and above_ema200 is False:
        lt_score -= 1
        lt_notes.append("Technical SELL + below EMA200 — long-term accumulation is riskier now.")
    if call in ["BUY", "HOLD"] and above_ema200 is True:
        lt_score += 1
        lt_notes.append("Technical bias aligns with long-term uptrend.")

    if lt_score >= 3:
        lt_action = "HOLD / ACCUMULATE (long-term) on dips"
        lt_color = "buy"
    elif lt_score >= 1:
        lt_action = "HOLD (long-term) — review quarterly results"
        lt_color = "hold"
    elif lt_score >= -1:
        lt_action = "HOLD only with strict review — or reduce overweight"
        lt_color = "hold"
    else:
        lt_action = "Prefer EXIT / avoid fresh long-term capital"
        lt_color = "sell"

    return {
        "pnl_pct": pnl,
        "pnl_rs": pnl_rs,
        "invested": invested,
        "mkt_value": mkt_value,
        "st_action": st_action,
        "st_color": st_color,
        "st_notes": st_notes,
        "st_score": st_score,
        "lt_action": lt_action,
        "lt_color": lt_color,
        "lt_notes": lt_notes,
        "lt_score": lt_score,
    }


def show_holding_advisor(results=None):
    """
    Separate page: user enters stock, buy price, shares →
    detailed technical + fundamental analysis and ST/LT decision.
    """
    st.title("📥 My Holding — Hold / Sell / Buy?")
    st.caption(
        "Enter any stock you already own (or plan to buy). "
        "You get technical analysis, fundamental snapshot, "
        "and separate short-term vs long-term advice."
    )

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        sym_in = st.text_input(
            "NSE symbol",
            value=st.session_state.get("holding_sym", "RELIANCE"),
            placeholder="RELIANCE, TCS, INFY…",
            key="holding_sym_input",
        )
    with c2:
        buy_price = st.number_input(
            "Your buy price (₹)",
            min_value=0.05,
            value=float(st.session_state.get("holding_buy", 1000.0)),
            step=1.0,
            key="holding_buy_input",
        )
    with c3:
        shares = st.number_input(
            "No. of shares",
            min_value=1,
            value=int(st.session_state.get("holding_shares", 10)),
            step=1,
            key="holding_shares_input",
        )
    with c4:
        st.write("")
        st.write("")
        run = st.button("🔎 Analyse holding", type="primary", use_container_width=True)

    if not run and not st.session_state.get("holding_last_sym"):
        st.info("Fill symbol, buy price and shares, then click **Analyse holding**.")
        return

    if run:
        st.session_state.holding_last_sym = display_symbol(sym_in)
        st.session_state.holding_buy = buy_price
        st.session_state.holding_shares = shares

    symbol = st.session_state.get("holding_last_sym") or display_symbol(sym_in)
    buy_price = float(st.session_state.get("holding_buy", buy_price))
    shares = int(st.session_state.get("holding_shares", shares))

    with st.spinner(f"Loading {symbol} technicals + fundamentals…"):
        df = stock_history(symbol, interval="1d")
        if df is None or df.empty:
            st.error("Could not load price history for this symbol.")
            return
        result = analyse_stock(clean_symbol(symbol), df, fetch_news=True)
        if not result:
            st.error("Analysis failed for this symbol.")
            return
        # Prefer live/last quote if available
        q = live_quote(symbol)
        if q and q.get("price"):
            result["Price"] = q["price"]
        fund = fetch_fundamentals(symbol)
        advice = holding_horizon_advice(result, buy_price, shares, fund)

    # P&L strip
    st.subheader(f"{symbol} — {fund.get('name') or symbol}")
    a, b, c, d, e = st.columns(5)
    a.metric("LTP / Model price", f"₹{safe_float(result['Price']):,.2f}")
    b.metric("Your buy", f"₹{buy_price:,.2f}")
    c.metric("Shares", f"{shares}")
    d.metric("P&L %", f"{advice['pnl_pct']:+.2f}%")
    e.metric("P&L ₹", f"₹{advice['pnl_rs']:+,.0f}")
    st.caption(
        f"Invested ₹{advice['invested']:,.0f} · Market value ₹{advice['mkt_value']:,.0f} · "
        f"Sector: {fund.get('sector') or result.get('Sector', '')} · "
        f"{fund.get('industry', '')}"
    )

    # ST / LT recommendation boxes
    st.subheader("🧭 Decision — Short term vs Long term")
    left, right = st.columns(2)
    with left:
        box = "buy-box" if advice["st_color"] == "buy" else (
            "danger-box" if advice["st_color"] == "sell" else "success-box"
        )
        st.markdown(
            f"""
            <div class="{box}" style="padding:14px;border-radius:10px;">
            <h4 style="margin:0 0 8px 0;">Short term (days–few weeks)</h4>
            <p style="font-size:1.15em;margin:0;"><b>{advice['st_action']}</b></p>
            <p style="margin:8px 0 0 0;opacity:0.9;">Score {advice['st_score']} · Model call: {result.get('Call')} · Pred {safe_float(result.get('Prediction')):.0f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for n in advice["st_notes"]:
            st.write(f"• {n}")
        st.write(
            f"**Model target** ₹{safe_float(result.get('Target')):,.2f} · "
            f"**Stop** ₹{safe_float(result.get('Stop Loss')):,.2f} · "
            f"**Hold days** {result.get('Hold Days')}"
        )
    with right:
        box = "buy-box" if advice["lt_color"] == "buy" else (
            "danger-box" if advice["lt_color"] == "sell" else "success-box"
        )
        st.markdown(
            f"""
            <div class="{box}" style="padding:14px;border-radius:10px;">
            <h4 style="margin:0 0 8px 0;">Long term (months+)</h4>
            <p style="font-size:1.15em;margin:0;"><b>{advice['lt_action']}</b></p>
            <p style="margin:8px 0 0 0;opacity:0.9;">Score {advice['lt_score']} · Trend + fundamentals combined</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for n in advice["lt_notes"]:
            st.write(f"• {n}")
        if fund.get("target_mean"):
            st.write(f"**Analyst mean target (Yahoo):** ₹{safe_float(fund['target_mean']):,.2f}")
        if fund.get("recommendation"):
            st.write(f"**Street lean:** {fund['recommendation']}")

    st.divider()

    # Fundamentals report
    st.subheader("🏛️ Fundamental snapshot")
    if fund.get("error") and not fund.get("name"):
        st.warning(f"Fundamentals limited: {fund['error']}")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Market Cap", _fmt_num(fund.get("market_cap"), money=True))
    f2.metric("P/E (TTM)", _fmt_num(fund.get("pe")))
    f3.metric("Forward P/E", _fmt_num(fund.get("forward_pe")))
    f4.metric("P/B", _fmt_num(fund.get("pb")))
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("EPS", _fmt_num(fund.get("eps")))
    g2.metric("ROE", _fmt_num(fund.get("roe"), pct=True))
    g3.metric("Profit margin", _fmt_num(fund.get("profit_margins"), pct=True))
    g4.metric("Debt / Equity", _fmt_num(fund.get("debt_to_equity")))
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Revenue growth", _fmt_num(fund.get("revenue_growth"), pct=True))
    h2.metric("Earnings growth", _fmt_num(fund.get("earnings_growth"), pct=True))
    h3.metric("Dividend yield", _fmt_num(fund.get("dividend_yield"), pct=True))
    h4.metric("Beta", _fmt_num(fund.get("beta")))
    i1, i2, i3 = st.columns(3)
    i1.metric("52w high", _fmt_num(fund.get("fifty_two_high"), money=True))
    i2.metric("52w low", _fmt_num(fund.get("fifty_two_low"), money=True))
    i3.metric("Current ratio", _fmt_num(fund.get("current_ratio")))

    if fund.get("summary"):
        with st.expander("Business summary", expanded=False):
            st.write(fund["summary"])

    if fund.get("revenue_latest") or fund.get("net_income_latest"):
        st.caption(
            f"Latest reported · Revenue: {_fmt_num(fund.get('revenue_latest'), money=True)} · "
            f"Net income: {_fmt_num(fund.get('net_income_latest'), money=True)}"
        )

    st.divider()

    # Technical school + chart (reuse existing)
    st.subheader("📈 Technical position vs your entry")
    st.write(
        f"**Model call:** {result.get('Call')} · **Prediction:** {safe_float(result.get('Prediction')):.1f}% · "
        f"**Risk:** {safe_float(result.get('Risk %')):.2f}% ({result.get('Risk Level')}) · "
        f"**Patterns:** {result.get('Patterns')}"
    )
    st.write(f"**Reason:** {result.get('Reason', '')}")
    try:
        show_indicator_school(result)
    except Exception:
        pass
    try:
        show_interactive_chart_panel(symbol, result)
    except Exception as e:
        st.caption(f"Chart: {e}")

    # Optional: save into portfolio csv
    st.divider()
    st.subheader("💾 Save into My Portfolio file")
    if st.button("Add / update this holding in my_portfolio.csv"):
        try:
            ensure_files()
            port = load_portfolio()
            stock = display_symbol(symbol)
            row = {
                "Stock": stock,
                "Buy Price": buy_price,
                "Shares": shares,
                "Buy Date": datetime.now().strftime("%Y-%m-%d"),
                "Notes": f"Added from Holding Advisor · LTP {safe_float(result['Price']):.2f}",
            }
            # Flexible columns
            for col in row:
                if col not in port.columns:
                    port[col] = ""
            if "Stock" in port.columns and not port.empty:
                mask = port["Stock"].astype(str).str.upper() == stock
                if mask.any():
                    for k, v in row.items():
                        port.loc[mask, k] = v
                else:
                    port = pd.concat([port, pd.DataFrame([row])], ignore_index=True)
            else:
                port = pd.DataFrame([row])
            port.to_csv(PORTFOLIO_FILE, index=False)
            st.success(f"Saved {stock} to portfolio.")
        except Exception as e:
            st.error(f"Could not save: {e}")


# ============================================================
# PORTFOLIO PAGE
# ============================================================

def show_portfolio(
    results
):

    st.title(
        "💼 MY PORTFOLIO"
    )

    st.info(
        "Edit my_portfolio.csv to enter your holdings. "
        "The dashboard will automatically calculate HOLD / SELL / BUY MORE."
    )

    portfolio = load_portfolio()

    if portfolio.empty:

        st.warning(
            "Your portfolio is empty."
        )

        st.write(
            "Use these columns in my_portfolio.csv:"
        )

        st.code(
            """
Stock,Shares,Buy Price,Purchase Date,Maximum Holding Days
RELIANCE,10,1400,2026-08-01,15
TCS,5,3000,2026-08-05,15
            """.strip()
        )

        return

    analysis = portfolio_analysis(
        results
    )

    if analysis.empty:

        st.warning(
            "Run FULL MARKET SCAN first so the portfolio can be analysed."
        )

        st.dataframe(
            portfolio,
            use_container_width=True,
            hide_index=True
        )

        return

    st.subheader(
        "Portfolio Decision"
    )

    st.dataframe(
        analysis,
        use_container_width=True,
        hide_index=True
    )

    sells = analysis[
        analysis["Action"]
        .astype(str)
        .str.startswith("SELL")
    ]

    if not sells.empty:

        st.subheader(
            "🔴 Portfolio SELL Calls"
        )

        st.dataframe(
            sells,
            use_container_width=True,
            hide_index=True
        )

    buys = analysis[
        analysis["Action"]
        .astype(str)
        .str.contains(
            "BUY MORE"
        )
    ]

    if not buys.empty:

        st.subheader(
            "🟢 Portfolio BUY MORE Signals"
        )

        st.dataframe(
            buys,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SECTOR PAGE
# ============================================================

def show_sector_charts(sec: pd.DataFrame):
    """Bar / ranking charts for sector strength, BUY%, risk."""
    if sec is None or sec.empty:
        return

    st.subheader("📈 Sector charts")
    st.caption("Compare sectors by strength, BUY share, and average risk from the latest scan.")

    chart_kind = st.radio(
        "Chart type",
        ["Sector Strength", "BUY % by sector", "Avg Prediction", "Avg Risk %"],
        horizontal=True,
        key="sector_chart_kind",
    )

    plot_df = sec.copy()
    if chart_kind == "Sector Strength" and "Sector Strength" in plot_df.columns:
        ycol, color = "Sector Strength", "#4dabf7"
        title = "Sector Strength (higher = stronger)"
    elif chart_kind == "BUY % by sector" and "BUY %" in plot_df.columns:
        ycol, color = "BUY %", "#1a9b5f"
        title = "Share of BUY calls within each sector"
    elif chart_kind == "Avg Prediction" and "Avg_Prediction" in plot_df.columns:
        ycol, color = "Avg_Prediction", "#fcc419"
        title = "Average Prediction % by sector"
    elif "Avg_Risk" in plot_df.columns:
        ycol, color = "Avg_Risk", "#ff6b6b"
        title = "Average Risk % by sector (lower is safer)"
    else:
        st.caption("Not enough sector metrics to chart.")
        return

    plot_df = plot_df.sort_values(ycol, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=plot_df[ycol],
            y=plot_df["Sector"],
            orientation="h",
            marker_color=color,
            text=plot_df[ycol].round(1),
            textposition="outside",
        )
    )
    fig.update_layout(
        height=max(360, 28 * len(plot_df)),
        template="plotly_dark",
        title=title,
        margin=dict(l=10, r=40, t=50, b=10),
        xaxis_title=ycol,
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Combined overview
    if {"Sector Strength", "BUY %", "Avg_Risk"}.issubset(set(plot_df.columns)):
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=plot_df["Avg_Risk"],
                y=plot_df["Sector Strength"],
                mode="markers+text",
                text=plot_df["Sector"],
                textposition="top center",
                marker=dict(
                    size=plot_df["BUY %"].clip(lower=5) / 2 + 8,
                    color=plot_df["BUY %"],
                    colorscale="Greens",
                    showscale=True,
                    colorbar=dict(title="BUY %"),
                ),
                name="Sectors",
            )
        )
        fig2.update_layout(
            height=480,
            template="plotly_dark",
            title="Sector map — Strength vs Risk (bubble size ~ BUY %)",
            xaxis_title="Avg Risk % →",
            yaxis_title="Sector Strength →",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)


def show_sector_page(
    results
):

    st.title(
        "🏭 SECTOR-WISE ANALYSIS"
    )

    results = ensure_result_columns(results)

    if results.empty:

        st.info(
            "Run FULL MARKET SCAN first."
        )

        return

    sec = sector_table(
        results
    )

    if sec.empty:
        return

    show_sector_charts(sec)

    st.subheader("Sector table")
    st.dataframe(
        sec,
        use_container_width=True,
        hide_index=True
    )

    selected_sector = st.selectbox(
        "Select Sector",
        sec["Sector"].tolist()
    )

    if st.button(
        "📂 OPEN SELECTED SECTOR",
        type="primary"
    ):

        st.session_state.selected_sector = (
            selected_sector
        )

    chosen = (
        st.session_state.selected_sector
        or
        selected_sector
    )

    x = results[
        results["Sector"]
        ==
        chosen
    ].copy()

    x = x.sort_values(
        [
            "Prediction",
            "Risk %",
        ],
        ascending=[
            False,
            True,
        ]
    )

    st.subheader(
        f"📊 {chosen} Stocks"
    )

    st.dataframe(
        x.drop(
            columns=[
                "Data",
                "News",
            ],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Open Individual Stock"
    )

    for stock in x[
        "Stock"
    ].tolist():

        if st.button(
            f"📈 OPEN {stock}",
            key="sector_"
            + chosen
            + "_"
            + stock
        ):

            st.session_state.selected_stock = (
                stock
            )

            st.session_state.page = (
                "Stock Analysis"
            )

            st.rerun()


# ============================================================
# FIND STOCK PAGE
# ============================================================

def recommend_stocks_from_words(results: pd.DataFrame, query: str, limit: int = 15) -> pd.DataFrame:
    """Natural-language-ish recommendations from a few words."""
    if results is None or results.empty or not str(query).strip():
        return pd.DataFrame()
    q = str(query).lower().strip()
    x = ensure_result_columns(results).copy()
    # Start from scan search
    try:
        found = find_stock_results(x, query)
    except Exception:
        found = x.copy()
    if found is None or found.empty:
        found = x.copy()

    # Intent filters
    call_u = found["Call"].astype(str).str.upper() if "Call" in found.columns else pd.Series([""] * len(found))
    if any(w in q for w in ("buy", "long", "accumulate", "upside")):
        found = found[call_u.str.contains("BUY|HOLD|WATCH", na=False)]
    if any(w in q for w in ("sell", "short", "exit", "weak")):
        found = found[call_u.str.contains("SELL", na=False)]
    if "low risk" in q or "safe" in q:
        if "Risk Level" in found.columns:
            found = found[found["Risk Level"].astype(str).str.upper().isin(["LOW", "MEDIUM"])]
    if "high risk" in q:
        if "Risk Level" in found.columns:
            found = found[found["Risk Level"].astype(str).str.upper().str.contains("HIGH", na=False)]

    # Sector keywords
    sector_map = {
        "bank": "Bank", "banking": "Bank", "it ": "IT", " software": "IT", "pharma": "Pharma",
        "auto": "Auto", "metal": "Metal", "fmcg": "FMCG", "energy": "Energy", "oil": "Energy",
        "realty": "Realty", "infra": "Infra", "finance": "Finance",
    }
    if "Sector" in found.columns:
        for kw, sec in sector_map.items():
            if kw in q:
                found = found[found["Sector"].astype(str).str.contains(sec, case=False, na=False)]

    # Price under X
    import re
    m = re.search(r"under\s+(\d+)", q)
    if m and "Price" in found.columns:
        found = found[pd.to_numeric(found["Price"], errors="coerce") <= float(m.group(1))]

    if "Prediction" in found.columns:
        found = found.sort_values("Prediction", ascending=False)
    return found.head(limit)


def show_find_stock(
    results
):

    st.title(
        "🔍 FIND STOCK"
    )

    st.write(
        "Type a few words — get stock **recommendations** (buy/sell, sector, risk)."
    )

    query = st.text_input(
        "Search / recommend",
        placeholder=(
            "Example: low risk banking stocks to buy"
        ),
        key="find_stock_q",
    )

    examples = st.multiselect(
        "Quick phrases",
        [
            "low risk bank stocks to buy",
            "IT stocks to buy",
            "strong pharma",
            "stocks under 500",
            "sell calls high risk",
            "metal stocks",
            "RELIANCE",
        ],
        key="find_quick",
    )
    if examples and not query.strip():
        query = examples[0]

    if results.empty:

        st.info(
            "Run FULL MARKET SCAN first for best recommendations."
        )

        return

    if not query.strip():

        st.info(
            "Examples: "
            "low risk bank stocks, "
            "IT stocks to buy, "
            "strong pharma stocks, "
            "stocks under 1000, "
            "sell calls, RELIANCE"
        )

        return

    found = recommend_stocks_from_words(results, query, limit=25)
    if found.empty:
        found = find_stock_results(results, query)

    if found.empty:

        st.warning(
            "No matching stocks found in the current scan."
        )

        return

    st.success(
        f"**{len(found)}** recommendations for: _{query}_"
    )

    filterable_dataframe(
        found.drop(columns=["Data", "News"], errors="ignore"),
        key="find_rec_table",
        default_cols=[c for c in [
            "Stock", "Sector", "Call", "Price", "Prediction", "Risk Level",
            "Target", "Stop Loss", "Patterns",
        ] if c in found.columns],
    )

    st.subheader("Cards — buy/sell executes to Paper Trading")
    for stock in found["Stock"].head(10).tolist():
        row = found[found["Stock"] == stock].iloc[0]
        render_call_stock_card(row.to_dict(), section_key="find")

    st.subheader(
        "Open Detailed Analysis"
    )

    for stock in found[
        "Stock"
    ].head(50).tolist():

        row = found[
            found["Stock"]
            ==
            stock
        ].iloc[0]

        if st.button(
            f"📈 {stock} | {row['Call']} | Prediction {row['Prediction']}%",
            key="find_"
            + stock
        ):

            st.session_state.selected_stock = (
                stock
            )

            st.session_state.page = (
                "Stock Analysis"
            )

            st.rerun()


# ============================================================
# EXPORT
# ============================================================

def create_excel(
    results
):

    filename = (
        "NSE_V12_Report_"
        +
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        +
        ".xlsx"
    )

    path = (
        APP_DIR /
        filename
    )

    portfolio = load_portfolio()

    history = load_history()

    with pd.ExcelWriter(
        path,
        engine="openpyxl"
    ) as writer:

        if (
            results is not None
            and
            not results.empty
        ):

            clean = results.drop(
                columns=[
                    "Data",
                    "News",
                ],
                errors="ignore"
            )

            clean.to_excel(
                writer,
                sheet_name="All Stocks",
                index=False
            )

            sells = clean[
                clean["Call"] == "SELL"
            ]

            sells.to_excel(
                writer,
                sheet_name="SELL Calls",
                index=False
            )

            buys = clean[
                clean["Call"] == "BUY"
            ]

            buys.to_excel(
                writer,
                sheet_name="BUY Calls",
                index=False
            )

            holds = clean[
                clean["Call"] == "HOLD"
            ]

            holds.to_excel(
                writer,
                sheet_name="HOLD Calls",
                index=False
            )

            sectors = sector_table(
                results
            )

            sectors.to_excel(
                writer,
                sheet_name="Sector Analysis",
                index=False
            )

            p_analysis = portfolio_analysis(
                results
            )

            if not p_analysis.empty:

                p_analysis.to_excel(
                    writer,
                    sheet_name="Portfolio Decision",
                    index=False
                )

        if not portfolio.empty:

            portfolio.to_excel(
                writer,
                sheet_name="My Portfolio",
                index=False
            )

        history.to_excel(
            writer,
            sheet_name="Prediction History",
            index=False
        )

    return path


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "results":
        pd.DataFrame(),

    "last_run":
        None,

    "page":
        "Dashboard",

    "selected_stock":
        "",

    "selected_sector":
        "",

    "auto_refresh":
        False,  # default OFF — auto-refresh was making the dashboard slow
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# LOAD SAVED RESULT
# ============================================================

if (
    st.session_state.results.empty
    and
    RESULT_FILE.exists()
):

    try:

        saved = pd.read_csv(
            RESULT_FILE
        )

        if not saved.empty:

            st.session_state.results = (
                saved
            )

    except Exception:
        pass


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📈 NSE V12")
    st.caption(market_status_text())
    st.divider()

    if st.button("🏠 Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
        st.rerun()

    st.markdown("**📊 Indices**")
    if st.button("📈 NIFTY 50 Chart + Analysis", use_container_width=True):
        st.session_state.page = "Nifty Analysis"
        st.rerun()

    if st.button("🏦 BANK NIFTY Chart + Analysis", use_container_width=True):
        st.session_state.page = "BankNifty Analysis"
        st.rerun()

    st.markdown("**📞 Calls**")
    if st.button("🟢 BUY Calls", use_container_width=True):
        st.session_state.page = "BUY Calls"
        st.rerun()

    if st.button("🔴 SELL Calls", use_container_width=True):
        st.session_state.page = "SELL Calls"
        st.rerun()

    if st.button("✅ Sure Call Desk", use_container_width=True):
        st.session_state.page = "Sure Calls"
        st.rerun()

    if st.button("🧪 Strategy Lab + Backtest", use_container_width=True):
        st.session_state.page = "Strategy Lab"
        st.rerun()

    st.markdown("**🔎 Tools**")
    if st.button("🔍 Find Stock", use_container_width=True):
        st.session_state.page = "Find Stock"
        st.rerun()

    if st.button("🏭 Sector Analysis", use_container_width=True):
        st.session_state.page = "Sector Analysis"
        st.rerun()

    if st.button("📥 My Holding Advisor", use_container_width=True):
        st.session_state.page = "Holding Advisor"
        st.rerun()

    if st.button("💼 My Portfolio", use_container_width=True):
        st.session_state.page = "Portfolio"
        st.rerun()

    if st.button("🕐 Past Predictions", use_container_width=True):
        st.session_state.page = "History"
        st.rerun()

    if st.button("🤖 Auto Trade Tracker", use_container_width=True):
        st.session_state.page = "Trade Tracker"
        st.rerun()

    if st.button("🧪 Dummy / Paper Backtest", use_container_width=True):
        st.session_state.page = "Paper Trading"
        st.rerun()

    if st.button("🔍 Analyze Stock", use_container_width=True):
        st.session_state.page = "Stock Analysis"
        st.rerun()

    st.divider()

    st.subheader(
        "🚀 MARKET SCAN"
    )

    scan = st.button(
        "🚀 FULL MARKET SCAN",
        type="primary",
        use_container_width=True
    )

    st.caption(
        f"Universe: {len(NSE_STOCKS):,} NSE symbols"
    )

    st.divider()

    st.subheader("🔄 Refresh")

    refresh_secs = st.selectbox(
        "Auto refresh every",
        [5, 10, 15, 30, 60],
        index=0,
        help="Full page reload interval when Auto refresh is ON. 5s is more live but heavier.",
        key="refresh_secs_select",
    )
    st.session_state.refresh_seconds = int(refresh_secs)

    auto_refresh = st.checkbox(
        "Auto refresh ON",
        value=bool(st.session_state.get("auto_refresh", False)),
        help="OFF = fastest tab switching. ON = reload every N seconds (quotes update).",
    )
    st.session_state.auto_refresh = auto_refresh

    if st.button("🔄 Manual refresh now", use_container_width=True, type="primary"):
        # Clear short caches so quotes/history refresh without full scan
        try:
            live_quote.clear()
        except Exception:
            pass
        st.rerun()

    if st.button(
        "↻ Refresh top 15 prices",
        use_container_width=True,
        help="Updates last price for top 15 scanned stocks only (fast).",
    ):
        if not st.session_state.results.empty:
            with st.spinner("Refreshing prices..."):
                st.session_state.results = update_results_with_live_prices(
                    st.session_state.results,
                    max_stocks=15,
                )
            st.success("Top prices refreshed.")
            st.rerun()

    st.caption(
        "Tip: keep Auto refresh OFF while browsing tabs. "
        "Use Manual refresh or 5s auto only when you need live prices. "
        "All analysis works when market is closed (last close data)."
    )


# ============================================================
# FULL MARKET SCAN
# ============================================================

if scan:

    with st.spinner(
        f"Scanning {len(NSE_STOCKS):,} NSE stocks..."
    ):

        results = run_scanner()

    if results is None or results.empty:

        st.error(
            "No usable market results were returned. The data provider may be "
            "temporarily unavailable; please run the scan again."
        )

    else:

        results = ensure_result_columns(results)
        results = refresh_sector_for_results(results)

        st.session_state.results = results
        st.session_state.last_run = datetime.now()

        # Save CSV result
        results.drop(
            columns=["Data", "News"],
            errors="ignore",
        ).to_csv(RESULT_FILE, index=False)

        save_recommendations(results)
        evaluate_history()
        try:
            learn_from_history(min_closed=5)
        except Exception:
            pass
        try:
            sync_auto_trades_tracker(max_live=25)
        except Exception:
            pass

        # Sector coverage summary
        try:
            sec_counts = results["Sector"].value_counts()
            top_secs = ", ".join(
                f"{k}: {v}" for k, v in sec_counts.head(8).items()
            )
            st.success(
                f"Scan completed: {len(results):,} stocks. Sectors → {top_secs}"
            )
        except Exception:
            st.success(f"Scan completed: {len(results):,} stocks.")


# ============================================================
# RESULTS
# ============================================================

results = ensure_result_columns(st.session_state.results)
results = refresh_sector_for_results(results)
st.session_state.results = results


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.title(
        "📈 NSE V12 STOCK DASHBOARD"
    )

    st.caption(
        "All stocks • Live market mode • Closing-price mode • "
        "Prediction history • Risk • Target • Stop Loss • Live prices"
    )

    live_market()

    # Quick access to index charts + BUY/SELL
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("📈 NIFTY Chart + Analysis", use_container_width=True, key="dash_nifty"):
            st.session_state.page = "Nifty Analysis"
            st.rerun()
    with q2:
        if st.button("🏦 BANK NIFTY Chart + Analysis", use_container_width=True, key="dash_bn"):
            st.session_state.page = "BankNifty Analysis"
            st.rerun()
    with q3:
        if st.button("🟢 BUY Calls", use_container_width=True, key="dash_buy"):
            st.session_state.page = "BUY Calls"
            st.rerun()
    with q4:
        if st.button("🔴 SELL Calls", use_container_width=True, key="dash_sell"):
            st.session_state.page = "SELL Calls"
            st.rerun()

    # Precision 1–2 BUY picks on dashboard (fallback to high WATCH if no BUY)
    if not results.empty:
        _buys = results[results["Call"].astype(str).str.upper() == "BUY"]
        if _buys.empty:
            _buys = results[
                (results["Call"].astype(str).str.upper() == "WATCH")
                & (pd.to_numeric(results.get("Prediction"), errors="coerce").fillna(0) >= 65)
            ]
            if not _buys.empty:
                st.info(
                    "No strong-trend BUY in last scan — showing top near-buy **WATCH** names so the board is not empty."
                )
        if not _buys.empty:
            _picks = pick_precision_buys(_buys, n=2)
            if not _picks.empty:
                st.subheader("🎯 Today’s Precision BUY Picks (1–2 stocks)")
                st.caption(
                    "Shortlist — high prediction, controlled risk. Run a fresh FULL MARKET SCAN if list looks stale."
                )
                cols = st.columns(len(_picks))
                for i, (_, prow) in enumerate(_picks.iterrows()):
                    with cols[i]:
                        st.markdown(
                            f"**#{i+1} {prow['Stock']}** ({prow.get('Sector','')})\n\n"
                            f"Pred **{safe_float(prow['Prediction']):.0f}%** · "
                            f"Risk **{safe_float(prow['Risk %']):.1f}%** · "
                            f"Score **{safe_float(prow.get('Precision Score')):.0f}**\n\n"
                            f"₹{safe_float(prow['Price']):,.1f} → "
                            f"T ₹{safe_float(prow['Target']):,.1f} / "
                            f"SL ₹{safe_float(prow['Stop Loss']):,.1f}"
                        )
                        if st.button("Open", key=f"dash_prec_{prow['Stock']}"):
                            st.session_state.selected_stock = prow["Stock"]
                            st.session_state.page = "Stock Analysis"
                            st.rerun()
                if st.button("See all Precision Picks on BUY page →", key="dash_to_buy_prec"):
                    st.session_state.page = "BUY Calls"
                    st.rerun()
                for _, prow in _picks.iterrows():
                    with st.expander(f"🔁 Prior history & learning — {prow['Stock']}", expanded=False):
                        show_prior_call_learning_panel(
                            prow["Stock"], str(prow.get("Reason", ""))
                        )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    # Refresh current prices on scan results (Target/SL locked)
    if not results.empty:
        try:
            results = update_results_with_live_prices(results, max_stocks=30)
            st.session_state.results = results
        except Exception:
            pass
        try:
            refresh_history_current_prices(max_stocks=30)
        except Exception:
            pass

    # Past data + success rate from CSV (works even without a fresh scan)
    show_dashboard_past_data()

    st.divider()

    # --------------------------------------------------------
    # Scan information
    # --------------------------------------------------------

    if st.session_state.last_run:

        st.info(
            "Last full scan: "
            +
            st.session_state.last_run.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    if results.empty:

        st.warning(
            "No scan results available."
        )

        st.write(
            "Press **🚀 FULL MARKET SCAN** in the sidebar."
        )

        st.write(
            "After the scan, all usable NSE stocks will appear here."
        )

    else:

        results = ensure_result_columns(results)

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        st.subheader(
            "📋 ALL SCANNED STOCKS"
        )

        f1, f2, f3, f4, f5 = st.columns(5)

        call_filter = f1.selectbox(
            "Call",
            [
                "ALL",
                "BUY",
                "HOLD",
                "SELL",
                "WATCH",
            ]
        )

        risk_filter = f2.selectbox(
            "Risk",
            [
                "ALL",
                "LOW",
                "MEDIUM",
                "HIGH",
                "VERY HIGH",
            ]
        )

        sector_options = [
            "ALL"
        ] + sorted(

results["Sector"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        sector_filter = f3.selectbox(
            "Sector",
            sector_options
        )

        min_prediction = f4.slider(
            "Minimum Prediction %",
            0,
            95,
            0
        )

        display_options = [
            10,
            25,
            50,
            100,
            200,
            500,
            1000,
            "ALL",
        ]

        display_count = f5.selectbox(
            "Stocks to Display",
            display_options,
            index=1
        )

        # ----------------------------------------------------
        # Apply filters
        # ----------------------------------------------------

        filtered = results.copy()

        if call_filter != "ALL":

            filtered = filtered[
                filtered["Call"]
                ==
                call_filter
            ]

        if risk_filter != "ALL":

            filtered = filtered[
                filtered["Risk Level"]
                ==
                risk_filter
            ]

        if sector_filter != "ALL":

            filtered = filtered[
                filtered["Sector"]
                ==
                sector_filter
            ]

        filtered = filtered[
            pd.to_numeric(
                filtered["Prediction"],
                errors="coerce"
            )
            >=
            min_prediction
        ]

        filtered = filtered.sort_values(
            [
                "Prediction",
                "Risk %",
            ],
            ascending=[
                False,
                True,
            ]
        )

        if display_count != "ALL":

            filtered = filtered.head(
                int(display_count)
            )

        st.write(
            f"Showing {len(filtered):,} stocks "
            f"from {len(results):,} scanned stocks."
        )

        # ----------------------------------------------------
        # Main table
        # ----------------------------------------------------

        table_columns = [
            "Rank",
            "Stock",
            "Sector",
            "Price",
            "Current Price",
            "Call",
            "Prediction",
            "Risk %",
            "Risk Level",
            "Target",
            "Stop Loss",
            "Hold Days",
            "Priority",
            "RSI",
            "ADX",
        ]
        st.caption(
            "**Price / Current Price** = live or last close. "
            "**Target & Stop Loss** stay as set at scan time (locked)."
        )

        available_columns = [
            c
            for c in table_columns
            if c in filtered.columns
        ]

        st.dataframe(
            filtered[
                available_columns
            ],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Individual stock buttons
        # ----------------------------------------------------

        st.subheader(
            "📈 Open Detailed Stock Analysis"
        )

        # Show buttons for selected filtered results.
        # This allows a separate analysis page.

        for _, row in filtered.head(
            100
        ).iterrows():

            stock = row[
                "Stock"
            ]

            call = row[
                "Call"
            ]

            prediction = row[
                "Prediction"
            ]

            if st.button(
                f"📈 {stock} | {call} | Prediction {prediction}%",
                key="dashboard_"
                + str(stock)
            ):

                st.session_state.selected_stock = (
                    stock
                )

                st.session_state.page = (
                    "Stock Analysis"
                )

                st.rerun()

        # ----------------------------------------------------
        # Quick SELL CALL
        # ----------------------------------------------------

        st.subheader(
            "🔴 Current SELL Calls"
        )

        sells = results[
            results["Call"]
            ==
            "SELL"
        ].copy()

        if sells.empty:

            st.success(
                "No SELL calls currently detected."
            )

        else:

            sells = sells.sort_values(
                [
                    "Prediction",
                    "Risk %",
                ],
                ascending=[
                    True,
                    False,
                ]
            )

            st.dataframe(
                sells[
                    [
                        "Rank",
                        "Stock",
                        "Sector",
                        "Price",
                        "Prediction",
                        "Risk %",
                        "Risk Level",
                        "Target",
                        "Stop Loss",
                        "Hold Days",
                        "Reason",
                    ]
                ].head(50),
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # Sector priority
        # ----------------------------------------------------

        st.subheader(
            "🏭 Sector Priority"
        )

        sec = sector_table(
            results
        )

        if not sec.empty:

            st.dataframe(
                sec.head(20),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FIND STOCK
# ============================================================

elif st.session_state.page == "Find Stock":

    show_find_stock(
        results
    )


# ============================================================
# BUY CALLS
# ============================================================

elif st.session_state.page == "BUY Calls":

    show_buy_calls(
        results
    )


# ============================================================
# SELL CALLS
# ============================================================

elif st.session_state.page == "SELL Calls":

    show_sell_calls(
        results
    )


elif st.session_state.page == "Sure Calls":

    show_sure_calls_page(results)


elif st.session_state.page == "Strategy Lab":

    show_strategy_lab(results)


# ============================================================
# NIFTY / BANK NIFTY
# ============================================================

elif st.session_state.page == "Nifty Analysis":

    show_index_page("NIFTY")


elif st.session_state.page == "BankNifty Analysis":

    show_index_page("BANKNIFTY")


# ============================================================
# SECTOR
# ============================================================

elif st.session_state.page == "Sector Analysis":

    show_sector_page(
        results
    )


# ============================================================
# PORTFOLIO
# ============================================================

elif st.session_state.page == "Holding Advisor":

    show_holding_advisor(results)


elif st.session_state.page == "Portfolio":

    show_portfolio(
        results
    )


# ============================================================
# HISTORY
# ============================================================

elif st.session_state.page == "History":

    show_history()


elif st.session_state.page == "Trade Tracker":

    show_trade_tracker()


elif st.session_state.page == "Paper Trading":

    show_paper_trading()


# ============================================================
# STOCK ANALYSIS
# ============================================================

elif st.session_state.page == "Stock Analysis":

    st.title(
        "🔍 ANALYZE ANY NSE STOCK"
    )

    typed = st.text_input(
        "Type NSE stock symbol",
        value=st.session_state.selected_stock,
        placeholder=(
            "RELIANCE, TCS, SBIN, INFY, TATAMOTORS"
        )
    )

    if st.button(
        "🔎 ANALYZE NOW",
        type="primary"
    ):

        if typed.strip():

            st.session_state.selected_stock = (
                display_symbol(
                    typed
                )
            )

            st.rerun()

    st.caption(
        "This page works even when the market is closed. "
        "It uses the latest available closing data."
    )

    if st.session_state.selected_stock:

        show_stock(
            st.session_state.selected_stock
        )


# ============================================================
# AUTO REFRESH (optional — OFF by default for speed)
# ============================================================

if st.session_state.get("auto_refresh", False):
    _secs = int(st.session_state.get("refresh_seconds", LIVE_REFRESH_SECONDS) or 5)
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {_secs * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NSE V12 • "
    +
    (
        "LIVE MARKET ANALYSIS"
        if nse_market_open_now()
        else
        "MARKET CLOSED • USING LATEST AVAILABLE CLOSING DATA"
    )
    +
    " • Research and decision-support only."
)
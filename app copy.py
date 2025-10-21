# filename: app.py
# -*- coding: utf-8 -*-

"""
AI Financial Assistant — Deterministic + Dynamic Q&A (updated)
- Deterministic handlers (transactions, totals, thresholds, windows)
- Offers catalog is fully data-driven from data/offers_catalog_updated.txt
- Offer list/details/lookup-by-merchant wired to catalog (no hardcoding)
- Deterministic suggestions (fully specified follow-ups) to avoid "I don't know"
- Top-N fix: honors 'top_k' for top-account queries
"""

from __future__ import annotations
import os
import re
import difflib
from datetime import date, datetime, timedelta
from calendar import monthrange
from typing import Optional, Tuple, Literal, List, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

# Optional LangChain + Azure OpenAI (only if configured)
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    try:
        from langchain.prompts import ChatPromptTemplate  # older LC compatibility
    except Exception:
        ChatPromptTemplate = None
try:
    from langchain_openai import AzureChatOpenAI
    from langchain_experimental.agents import create_pandas_dataframe_agent
except Exception:
    AzureChatOpenAI = None
    create_pandas_dataframe_agent = None

# ───────────────────────────────────────────────────────────────────────────────
# 0) Configuration
# ───────────────────────────────────────────────────────────────────────────────
load_dotenv(override=True)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
ALLOW_CODE = True
HEAD_ROWS = 10
MAX_ITER = 10
TODAY = date.today()
CURRENT_YEAR = TODAY.year
DEBUG_ROUTING = False  # set True to see router decisions

st.set_page_config(page_title="AI Assistant", page_icon="💳")
st.title("💳 AI Spending Insights Assistant")

# ───────────────────────────────────────────────────────────────────────────────
# 1) Session State (light memory)
# ───────────────────────────────────────────────────────────────────────────────
if "cum_context" not in st.session_state:
    st.session_state["cum_context"] = {"window": None, "metric": None, "threshold": None, "merchant": None}
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "type": "text", "content": "How can I help you ?"}
    ]

def add_assistant_text(text: str) -> None:
    st.session_state["messages"].append({"role": "assistant", "type": "text", "content": text})

def add_assistant_df(df: pd.DataFrame, caption: Optional[str] = None) -> None:
    df.index = range(1, len(df) + 1)
    st.session_state["messages"].append({
        "role": "assistant", "type": "dataframe",
        "columns": list(df.columns), "data": df.to_dict("records"),
        "caption": caption
    })

def _queue_user_text(text: str):
    st.session_state["queued_user"] = text

# ───────────────────────────────────────────────────────────────────────────────
# 2) Data Loading
# ───────────────────────────────────────────────────────────────────────────────
# 2) Data Loading (Universal & Simple — accepts any CSV/Excel schema)
# 2) Data Loading — simplified, catalog-style (no uploader; always from data/)
st.markdown("### Data")
import glob, csv

# --- Canonical schema we want everywhere ---
CANONICAL_COLS = ["account_number", "transaction_date", "mcc_code", "merchant_name", "transaction_amount"]

# --- Synonyms for flexible column names (kept small but practical) ---
SYNONYMS = {
    "account_number": {"account_number", "account_no", "acct_no", "account", "acct", "customer_id", "cust_id", "card_account"},
    "transaction_date": {"transaction_date", "txn_date", "date", "purchase_date", "posted_date", "trans_date", "datetime"},
    "mcc_code": {"mcc_code", "merchant_category_code", "mcc", "category_code"},
    "merchant_name": {"merchant_name", "merchant", "store", "shop", "vendor", "brand"},
    "transaction_amount": {"transaction_amount", "amount", "amt", "value", "price", "spend"},
    "_debit": {"debit", "debit_amount", "withdrawal", "dr"},
    "_credit": {"credit", "credit_amount", "deposit", "cr", "refund_amount"},
    "card_name": {"card_name", "card", "product", "product_name"},
}

def _normalize_header(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

def _canonical_for(col: str) -> str | None:
    c = _normalize_header(col)
    for canonical, names in SYNONYMS.items():
        for n in names:
            if c == _normalize_header(n):
                return canonical
    for canonical, names in SYNONYMS.items():
        for n in names:
            if _normalize_header(n) in c or c in _normalize_header(n):
                return canonical
    return None

def _clean_merchant_token(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

def _detect_csv_delimiter(path_or_buffer) -> str:
    try:
        if hasattr(path_or_buffer, "read"):
            pos = path_or_buffer.tell()
            sample = path_or_buffer.read(2048).decode("utf-8", errors="ignore")
            path_or_buffer.seek(pos)
        else:
            with open(path_or_buffer, "rb") as fh:
                sample = fh.read(2048).decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "\n"])
        return dialect.delimiter
    except Exception:
        return ","

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        canon = _canonical_for(col)
        if canon and canon not in rename_map.values():
            rename_map[col] = canon
    return df.rename(columns=rename_map)

@st.cache_data(ttl="2h", show_spinner="Loading data…")
def load_transactions_any(path_or_buffer) -> pd.DataFrame:
    name = getattr(path_or_buffer, "name", str(path_or_buffer)).lower()
    if name.endswith(".csv"):
        if hasattr(path_or_buffer, "read"):
            delim = _detect_csv_delimiter(path_or_buffer)
            df = pd.read_csv(path_or_buffer, sep=delim)
        else:
            df = pd.read_csv(path_or_buffer)
    elif name.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
        df = pd.read_excel(path_or_buffer)
    else:
        raise ValueError(f"Unsupported file format: {name}")
    df = _auto_map_columns(df)
    if "transaction_amount" not in df.columns:
        debit_col = next((c for c in df.columns if _canonical_for(c) == "_debit"), None)
        credit_col = next((c for c in df.columns if _canonical_for(c) == "_credit"), None)
        deb = pd.to_numeric(df.get(debit_col), errors="coerce") if debit_col in df.columns else 0.0
        cre = pd.to_numeric(df.get(credit_col), errors="coerce") if credit_col in df.columns else 0.0
        df["transaction_amount"] = pd.Series(deb).fillna(0.0) - pd.Series(cre).fillna(0.0)
    if "mcc_code" not in df.columns:
        df["mcc_code"] = pd.NA
    missing = set(CANONICAL_COLS) - set(df.columns)
    if missing:
        raise ValueError("Missing required columns after normalization: " + ", ".join(sorted(missing)))
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    # For DD/MM/YYYY style: uncomment next line
    # df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", dayfirst=True)
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce").fillna(0.0)
    try:
        df["mcc_code"] = pd.to_numeric(df["mcc_code"], errors="coerce").astype("Int32")
    except Exception:
        pass
    df["merchant_name_norm"] = df["merchant_name"].apply(_clean_merchant_token)
    df = df.sort_values(
        ["account_number", "transaction_date", "merchant_name", "transaction_amount"],
        kind="mergesort"
    ).reset_index(drop=True)
    return df

def _discover_transactions_path() -> Optional[str]:
    env_path = os.getenv("TXN_DATA_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path
    default_path = "data/citi_5years_transactions.csv"
    if os.path.exists(default_path):
        return default_path
    candidates = []
    for pat in ("data/*.csv", "data/*.xls", "data/*.xlsx", "data/*.xlsm", "data/*.xlsb"):
        candidates.extend(glob.glob(pat))
    return candidates[0] if candidates else None

@st.cache_data(ttl="2h", show_spinner="Loading transactions…")
def load_transactions_catalog(path: Optional[str] = None) -> pd.DataFrame:
    use_path = path or _discover_transactions_path()
    if not use_path:
        raise FileNotFoundError(
            "No transactions file found. Put one at `data/citi_5years_transactions.csv` or set TXN_DATA_PATH."
        )
    return load_transactions_any(use_path)

df: Optional[pd.DataFrame] = None
try:
    df = load_transactions_catalog()
    st.info(f"Using local transactions: `{os.path.basename(_discover_transactions_path() or '')}`")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(25), use_container_width=True)
except Exception as e:
    st.error(f"Could not load transactions: {e}")
    df = None

# ───────────────────────────────────────────────────────────────────────────────
# NEW: Load offers catalog from data folder (catalog-driven UI)
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl="2h", show_spinner=False)
def load_offers_catalog(path: str = "data/offers_catalog_updated.txt") -> Optional[pd.DataFrame]:
    """
    Reads your offers catalog from data/offers_catalog_updated.txt and normalizes a few fields.
    """
    try:
        cat = pd.read_csv(path)
        for c in ["merchants", "valid_days", "mechanic", "description", "title"]:
            if c in cat.columns:
                cat[c] = cat[c].astype(str)
        cat["merchants_norm"] = cat["merchants"].str.strip().str.lower()
        cat["id_norm"] = cat["id"].astype(str).str.strip().str.lower()  # o1, o2, ...
        return cat
    except Exception as e:
        st.warning(f"Offers catalog not loaded: {e}")
        return None

offers_cat = load_offers_catalog()

# ───────────────────────────────────────────────────────────────────────────────
# 3) Deterministic Helpers (windows, merchants, cumulative, totals)
# ───────────────────────────────────────────────────────────────────────────────
# --- window helpers
def month_begin(d: date) -> date: return d.replace(day=1)
def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    return date(y, m, 1)
def prev_month_window(today: date = TODAY) -> Tuple[date, date]:
    first_this = month_begin(today)
    last_prev = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    return first_prev, last_prev
def last_3_complete_months(today: date = TODAY) -> Tuple[date, date]:
    first_this = month_begin(today)
    start = add_months(first_this, -3)
    end = first_this - timedelta(days=1)
    return start, end
def last_year_window(today: date = TODAY) -> Tuple[date, date]:
    y = today.year - 1
    return date(y, 1, 1), date(y, 12, 31)
def this_year_window(today: date = TODAY) -> Tuple[date, date]:
    return date(today.year, 1, 1), today
def this_month_window(today: date = TODAY) -> Tuple[date, date]:
    return month_begin(today), today
def today_window(today: date = TODAY) -> Tuple[date, date]:
    return today, today

# --- WEEK / DAY helpers
def week_begin(d: date, week_start: int = 0) -> date:
    return d - timedelta(days=(d.weekday() - week_start) % 7)
def this_week_window(today: date = TODAY, week_start: int = 0) -> Tuple[date, date]:
    start = week_begin(today, week_start)
    return start, today
def last_week_window(today: date = TODAY, week_start: int = 0) -> Tuple[date, date]:
    this_start = week_begin(today, week_start)
    prev_start = this_start - timedelta(days=7)
    prev_end = this_start - timedelta(days=1)
    return prev_start, prev_end
def last_n_weeks_complete(n: int, today: date = TODAY, week_start: int = 0) -> Tuple[date, date]:
    n = max(1, int(n))
    this_start = week_begin(today, week_start)
    start = this_start - timedelta(days=7 * n)
    end = this_start - timedelta(days=1)
    return start, end
def last_n_days_inclusive(n: int, today: date = TODAY) -> Tuple[date, date]:
    n = max(1, int(n))
    return today - timedelta(days=n - 1), today
def previous_n_days(n: int, today: date = TODAY) -> Tuple[date, date]:
    n = max(1, int(n))
    end = today - timedelta(days=1)
    start = end - timedelta(days=n - 1)
    return start, end
def yesterday_window(today: date = TODAY) -> Tuple[date, date]:
    y = today - timedelta(days=1)
    return y, y
def day_before_yesterday_window(today: date = TODAY) -> Tuple[date, date]:
    dby = today - timedelta(days=2)
    return dby, dby
def last_n_complete_months(n: int, today: date = TODAY) -> Tuple[date, date]:
    n = max(1, int(n))
    first_this = month_begin(today)
    start = add_months(first_this, -n)
    end = first_this - timedelta(days=1)
    return start, end

def filter_window(frame: pd.DataFrame, window: Tuple[date, date]) -> pd.DataFrame:
    start, end = window
    mask = (frame["transaction_date"] >= pd.to_datetime(start)) & (frame["transaction_date"] <= pd.to_datetime(end))
    return frame.loc[mask].copy()

def sort_for_cumsum(frame: pd.DataFrame, by_cols=("account_number",), date_col="transaction_date") -> pd.DataFrame:
    return frame.sort_values(list(by_cols) + [date_col], kind="mergesort")

def cumulative_by(frame: pd.DataFrame, by=("account_number",), window: Optional[Tuple[date, date]] = None) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    if f.empty: return f
    f = sort_for_cumsum(f, by_cols=by)
    f["cumulative_amount"] = f.groupby(list(by), dropna=False)["transaction_amount"].cumsum()
    return f

# Running-cumulative — original 'over' helpers
def count_accounts_cum_over_anytime(frame: pd.DataFrame, threshold: float, window=None, by=("account_number",)) -> int:
    f = frame if window is None else filter_window(frame, window)
    if f.empty: return 0
    f = sort_for_cumsum(f, by_cols=by)
    f["cumulative_amount"] = f.groupby(list(by), dropna=False)["transaction_amount"].cumsum()
    max_cum = f.groupby(list(by), dropna=False)["cumulative_amount"].max()
    return int((max_cum > threshold).sum())

def accounts_cum_over_anytime(frame: pd.DataFrame, threshold: float, window=None, by=("account_number",)) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    if f.empty:
        return pd.DataFrame(columns=[*by, "max_cumulative_amount", "first_cross_date"])
    f = sort_for_cumsum(f, by_cols=by)
    f["cumulative_amount"] = f.groupby(list(by), dropna=False)["transaction_amount"].cumsum()
    g = f.groupby(list(by), dropna=False)
    max_cum = g["cumulative_amount"].max()
    over = max_cum[max_cum > threshold]
    if over.empty:
        return pd.DataFrame(columns=[*by, "max_cumulative_amount", "first_cross_date"])
    crossed = f[f["cumulative_amount"] > threshold]
    first_dates = crossed.groupby(list(by), dropna=False)["transaction_date"].min()
    out = (
        pd.DataFrame({"max_cumulative_amount": over})
        .merge(first_dates.rename("first_cross_date"), left_index=True, right_index=True)
        .reset_index()
        .sort_values("max_cumulative_amount", ascending=False)
    )
    return out

# NEW: operator-aware cumulative helpers
def _symbol_for_op(op: str) -> str:
    return {"gt": ">", "ge": "≥", "lt": "<", "le": "≤"}.get(op, ">")

def count_accounts_cum_compare_anytime(
    frame: pd.DataFrame,
    threshold: float,
    op: Literal["gt", "ge", "lt", "le"] = "gt",
    window=None,
    by=("account_number",),
) -> int:
    f = frame if window is None else filter_window(frame, window)
    if f.empty: return 0
    f = sort_for_cumsum(f, by_cols=by)
    f["cumulative_amount"] = f.groupby(list(by), dropna=False)["transaction_amount"].cumsum()
    max_cum = f.groupby(list(by), dropna=False)["cumulative_amount"].max()
    if op == "gt":   return int((max_cum >  threshold).sum())
    if op == "ge":   return int((max_cum >= threshold).sum())
    if op == "lt":   return int((max_cum <  threshold).sum())
    return int((max_cum <= threshold).sum())

def accounts_cum_compare_anytime(
    frame: pd.DataFrame,
    threshold: float,
    op: Literal["gt", " " "ge", "lt", "le"] = "gt",
    window=None,
    by=("account_number",),
) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    if f.empty:
        return pd.DataFrame(columns=[*by, "max_cumulative_amount", "first_cross_date"])
    f = sort_for_cumsum(f, by_cols=by)
    f["cumulative_amount"] = f.groupby(list(by), dropna=False)["transaction_amount"].cumsum()
    g = f.groupby(list(by), dropna=False)
    max_cum = g["cumulative_amount"].max()
    if op == "gt":
        sel = max_cum[max_cum >  threshold]; cond = f["cumulative_amount"] >  threshold
    elif op == "ge":
        sel = max_cum[max_cum >= threshold]; cond = f["cumulative_amount"] >= threshold
    elif op == "lt":
        sel = max_cum[max_cum <  threshold]; cond = None
    else:  # "le"
        sel = max_cum[max_cum <= threshold]; cond = None
    if sel.empty:
        return pd.DataFrame(columns=[*by, "max_cumulative_amount", "first_cross_date"])
    if cond is not None:
        crossed = f[cond]
        first_dates = crossed.groupby(list(by), dropna=False)["transaction_date"].min()
    else:
        first_dates = pd.Series(pd.NaT, index=sel.index)
    out = (
        pd.DataFrame({"max_cumulative_amount": sel})
        .merge(first_dates.rename("first_cross_date"), left_index=True, right_index=True, how="left")
        .reset_index()
        .sort_values("max_cumulative_amount", ascending=(op in {"lt", "le"}))
    )
    return out

# TOTAL spend per window (non-cumulative)
def totals_by_account(frame: pd.DataFrame, window=None, merchant_norm: Optional[str] = None) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    f = f if not merchant_norm else f[f["merchant_name_norm"] == merchant_norm]
    if f.empty:
        return pd.DataFrame(columns=["account_number", "total_spend"])
    sums = f.groupby("account_number", dropna=False)["transaction_amount"].sum()
    return sums.reset_index(name="total_spend").sort_values("total_spend", ascending=False)

def list_accounts_total_over(frame: pd.DataFrame, threshold: float, window=None, merchant_norm: Optional[str] = None) -> pd.DataFrame:
    t = totals_by_account(frame, window=window, merchant_norm=merchant_norm)
    return t[t["total_spend"] > float(threshold)].reset_index(drop=True)

def known_merchants(frame: pd.DataFrame) -> set[str]:
    return set(frame["merchant_name_norm"].dropna().unique())

def match_merchant_in_query(q: str, frame: pd.DataFrame, cutoff: float = 0.8) -> Optional[str]:
    ql = q.lower()
    merchants = list(known_merchants(frame))
    # direct contains
    for m in merchants:
        if m and m in ql:
            return m
    # fuzzy n-gram
    tokens = [t for t in re.split(r"[^a-z0-9]+", ql) if t]
    candidates = [" ".join(tokens[i:j]) for i in range(len(tokens)) for j in range(i + 1, min(i + 5, len(tokens)) + 1)]
    for cand in candidates:
        match = difflib.get_close_matches(cand, merchants, n=1, cutoff=cutoff)
        if match:
            return match[0]
    return None

def filter_by_merchant(frame: pd.DataFrame, merchant_norm: Optional[str]) -> pd.DataFrame:
    if not merchant_norm: return frame
    f = frame[frame["merchant_name_norm"] == merchant_norm]
    if f.empty:
        f = frame[frame["merchant_name_norm"].str.contains(re.escape(merchant_norm), na=False)]
    return f.copy()

def top_accounts_total(frame: pd.DataFrame, window=None, merchant_norm: Optional[str] = None, top_k: int = 1, ascending: bool = False) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    f = filter_by_merchant(f, merchant_norm)
    if f.empty:
        cols = ["account_number", "total_spend"]
        if merchant_norm: cols.append("merchant")
        return pd.DataFrame(columns=cols)
    sums = f.groupby("account_number", dropna=False)["transaction_amount"].sum().sort_values(ascending=ascending)
    out = sums.head(top_k).reset_index(name="total_spend")
    if merchant_norm:
        out["merchant"] = merchant_norm
    return out

def total_spend(frame: pd.DataFrame, window=None, merchant_norm: Optional[str] = None) -> float:
    f = frame if window is None else filter_window(frame, window)
    f = filter_by_merchant(f, merchant_norm)
    if f.empty: return 0.0
    return float(f["transaction_amount"].sum())

# NEW: operator-aware TOTAL spend helpers
def list_accounts_total_compare(
    frame: pd.DataFrame,
    threshold: float,
    op: Literal["gt", "ge", "lt", "le"] = "gt",
    window=None,
    merchant_norm: Optional[str] = None,
) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    f = f if not merchant_norm else f[f["merchant_name_norm"] == merchant_norm]
    if f.empty:
        return pd.DataFrame(columns=["account_number", "total_spend"])
    sums = f.groupby("account_number", dropna=False)["transaction_amount"].sum().reset_index(name="total_spend")
    if op == "gt":
        out = sums[sums["total_spend"] >  threshold]; asc = False
    elif op == "ge":
        out = sums[sums["total_spend"] >= threshold]; asc = False
    elif op == "lt":
        out = sums[sums["total_spend"] <  threshold]; asc = True
    else:
        out = sums[sums["total_spend"] <= threshold]; asc = True
    return out.sort_values("total_spend", ascending=asc).reset_index(drop=True)

def count_accounts_total_compare(
    frame: pd.DataFrame,
    threshold: float,
    op: Literal["gt", "ge", "lt", "le"] = "gt",
    window=None,
    merchant_norm: Optional[str] = None,
) -> int:
    df_cmp = list_accounts_total_compare(frame, threshold=threshold, op=op, window=window, merchant_norm=merchant_norm)
    return int(len(df_cmp))

# Offers (existing deterministic eligibility for O1/O2/O3)
def offer1_prev_month(frame: pd.DataFrame) -> pd.DataFrame:
    start, end = prev_month_window()
    f = filter_window(frame, (start, end))
    spend = f.groupby("account_number", dropna=False)["transaction_amount"].sum()
    return spend[spend >= 3000].reset_index(name="total_spend_prev_month")

def offer2_last_3_months(frame: pd.DataFrame) -> pd.DataFrame:
    start, end = last_3_complete_months()
    f = filter_window(frame, (start, end))
    spend = f.groupby("account_number", dropna=False)["transaction_amount"].sum()
    return spend[spend >= 5000].reset_index(name="total_spend_last_3_months")

def offer3_last_year(frame: pd.DataFrame) -> pd.DataFrame:
    start, end = last_year_window()
    f = filter_window(frame, (start, end))
    g = f.groupby("account_number", dropna=False)
    spend = g["transaction_amount"].sum()
    has_sia = g["merchant_name"].apply(lambda s: s.str.contains("Singapore Airlines", case=False, na=False).any())
    mask = (spend >= 10000) & has_sia.reindex(spend.index, fill_value=False)
    return spend[mask].reset_index(name="total_spend_last_year")

# ───────────────────────────────────────────────────────────────────────────────
# 4) Parsers & Normalization
# ───────────────────────────────────────────────────────────────────────────────
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1)}
MONTHS.update({m[:3].lower(): i for m, i in MONTHS.items()})  # Jan, Feb, ...

def _normalize(q: str) -> str:
    ql = q.lower()
    ql = re.sub(r"[\#\*\n]+", " ", ql)
    ql = ql.replace("mnth", "month").replace("mth", "month")
    ql = ql.replace("yr", "year").replace(" ytd", " ytd")
    ql = ql.replace("in the ", " ").replace(" during ", " ").replace(" in ", " ")
    # week/day typos & abbreviations
    ql = ql.replace("weak", "week").replace("wek", "week").replace("weaks", "weeks")
    ql = ql.replace("wtd", "week to date")
    ql = ql.replace("todays", "today").replace("tody", "today")
    ql = ql.replace("yester day", "yesterday")
    # offer misspellings / variants
    ql = ql.replace("offres", "offers").replace("offre", "offer").replace("ofers", "offers")
    ql = re.sub(r"\s+", " ", ql).strip()
    return ql

GREETINGS = {"hi", "hello", "hey", "hola", "namaste"}
TRANSACTION_WORDS = {"transaction", "transactions", "txn", "txns", "sale", "sales", "purchase", "purchases", "swipes"}
DISTINCT_WORDS = {"distinct", "unique", "different"}
SPEND_WORDS = {"spend", "spent", "spending", "amount"}

def is_greeting(q: str) -> bool:
    return _normalize(q) in GREETINGS

def is_offer_list_query(q: str) -> bool:
    ql = _normalize(q)
    return ("offer list" in ql) or (" many offers" in ql) or ("list of all offer" in ql) or ("list of offer" in ql) or \
           ("list offer" in ql) or ("predefined offers" in ql) or ("what offers" in ql) or ("offers available" in ql) or \
           ("available offers" in ql)

def is_offers_eligibility_query(q: str) -> bool:
    ql = _normalize(q)
    return (("offer" in ql) and (not is_offer_list_query(q))) or ("eligible" in ql) or ("eligibility" in ql)

# UPDATED: allow any offer number (not limited to 1–3)
def parse_offer_number(q: str) -> Optional[int]:
    """
    Parses 'offer 1', 'offer1', 'o1', 'offer 12', etc. and returns the integer.
    """
    ql = _normalize(q)
    m = re.search(r"\boffer\s*(\d+)\b", ql)
    if m:
        return int(m.group(1))
    m = re.search(r"\bo(\d+)\b", ql)
    if m:
        return int(m.group(1))
    return None

def is_offer_detail_query(q: str) -> bool:
    """
    Returns True for queries asking details of a specific offer number,
    like 'what is offer12', 'offer 18 details', 'o5 detail', etc.
    """
    ql = _normalize(q)
    if parse_offer_number(q) is not None:
        if "eligible" in ql or "eligibility" in ql:
            return False
        return True
    if "offer details" in ql or "offers details" in ql:
        return True
    return False

# UPDATED: Offer list from catalog
def offer_list_df() -> pd.DataFrame:
    """
    Build the display list from the offers catalog if available,
    else fallback to a minimal message.
    """
    if offers_cat is not None and not offers_cat.empty:
        out = offers_cat[["id", "title", "description"]].copy()
        out["Offer"] = out["id"].str.upper().str.replace("O", "Offer ")
        out = out[["Offer", "title", "description"]].rename(columns={"title": "Name", "description": "Description"})
        return out.reset_index(drop=True)
    return pd.DataFrame([("No catalog", "Offers catalog not loaded")], columns=["Offer", "Description"])

# UPDATED: Offer details from catalog
def offer_details_df(n: int) -> pd.DataFrame:
    """
    Map 'offer N' to catalog id 'ON' and return the row with useful fields.
    """
    if offers_cat is not None and not offers_cat.empty:
        oid = f"o{n}"
        row = offers_cat[offers_cat["id_norm"] == oid]
        if not row.empty:
            cols = ["id","title","merchants","mechanic","cashback_pct","cap","min_cart","valid_days",
                    "expected_uplift_pct","description","eligibility","eligibility_window",
                    "threshold_amount","required_merchant"]
            existing = [c for c in cols if c in row.columns]
            return row[existing].reset_index(drop=True)
    return pd.DataFrame(columns=["id","title","description"])

def parse_time_window(q: str) -> Optional[Tuple[date, date]]:
    """Relative windows (now with week/day + 'last N ...' support)."""
    ql = _normalize(q)
    # simple day/week keywords
    if "yesterday" in ql: return yesterday_window()
    if "day before yesterday" in ql or "dby" in ql: return day_before_yesterday_window()
    if ("this week" in ql) or ("current week" in ql) or ("week to date" in ql): return this_week_window()
    if ("last week" in ql) or ("previous week" in ql) or ("prev week" in ql) or ("past week" in ql): return last_week_window()
    if ("last 3 months" in ql) or ("last three months" in ql) or ("previous 3 months" in ql) or ("prev 3 months" in ql): return last_3_complete_months()
    if ("last year" in ql) or ("previous year" in ql) or ("prev year" in ql) or ("last yr" in ql) or ("prev yr" in ql): return last_year_window()
    if ("this year" in ql) or ("current year" in ql) or ("this yr" in ql) or ("ytd" in ql) or ("year to date" in ql): return this_year_window()
    if ("last month" in ql) or ("previous month" in ql) or ("prev month" in ql): return prev_month_window()
    if ("this month" in ql) or ("mtd" in ql) or ("month to date" in ql): return this_month_window()
    if "today" in ql: return today_window()

    # generalized 'last/previous/past N <unit>'
    def _extract_n(text: str) -> Optional[int]:
        nums = re.findall(r"\b\d+\b", text)
        if not nums:
            return None
        return int(nums[-1])

    if re.search(r"\b(last|previous|prev|past)\b.*\bday(s)?\b", ql):
        n = _extract_n(ql)
        if n:
            if re.search(r"\b(previous|prev)\b", ql):
                return previous_n_days(n)
            return last_n_days_inclusive(n)

    if re.search(r"\b(last|previous|prev|past)\b.*\bweek(s)?\b", ql):
        n = _extract_n(ql)
        if n:
            return last_n_weeks_complete(n)

    if re.search(r"\b(last|previous|prev|past)\b.*\bmonth(s)?\b", ql):
        n = _extract_n(ql)
        if n:
            return last_n_complete_months(n)

    if re.search(r"\b(last|previous|prev|past)\b.*\byear(s)?\b", ql):
        n = _extract_n(ql)
        if n:
            y = TODAY.year - 1
            start = date(y - n + 1, 1, 1)
            end = date(y, 12, 31)
            return start, end

    return None

def _month_year_window(month: int, year: int) -> Tuple[date, date]:
    last_day = monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)

def parse_explicit_window(q: str) -> Optional[Tuple[date, date]]:
    """
    Explicit windows like:
    - "in 2024"
    - "December 2024" / "Dec 2024"
    - "from 2024-05-01 to 2024-12-31" / "between 1/5/2024 and 12/31/2024"
    - "in 2024 December" (tolerated)
    """
    txt = _normalize(q)
    m = re.search(r"(from|between)\s+([0-9/\-\s]+)\s+(to|and)\s+([0-9/\-\s]+)", txt)
    if m:
        s, e = m.group(2).strip(), m.group(4).strip()
        try:
            sd = pd.to_datetime(s, errors="raise").date()
            ed = pd.to_datetime(e, errors="raise").date()
            if sd > ed: sd, ed = ed, sd
            return sd, ed
        except Exception:
            pass
    for token in MONTHS:
        if token in txt:
            ym = re.search(rf"{token}\s+(\d{{4}})", txt)
            if ym:
                year = int(ym.group(1)); month = MONTHS[token]
                return _month_year_window(month, year)
            my = re.search(r"(\d{4})\s+" + token, txt)
            if my:
                year = int(my.group(1)); month = MONTHS[token]
                return _month_year_window(month, year)
    y = re.search(r"\b(20\d{2})\b", txt)
    if y and (" 20" in f" {txt} "):
        year = int(y.group(1))
        return date(year, 1, 1), date(year, 12, 31)
    return None

def parse_threshold(q: str) -> Optional[float]:
    text = _normalize(q).replace(",", " ")
    m_k = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text)  # '15k' or '15 k'
    if m_k:
        return float(m_k.group(1)) * 1000.0
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums: return None
    return float(max(nums, key=lambda s: float(s)))

def parse_accounts(q: str) -> List[str]:
    return re.findall(r"\b[A-Za-z]\d{6,}\b", q)

def parse_top_k(q: str) -> int:
    m = re.search(r"\btop\s+(\d+)\b", _normalize(q))
    if m: return int(m.group(1))
    return 1

def mentions_transactions(ql: str) -> bool:
    return any(w in ql for w in TRANSACTION_WORDS)

def mentions_distinct(ql: str) -> bool:
    return any(w in ql for w in DISTINCT_WORDS)

def mentions_spendish(ql: str) -> bool:
    return any(w in ql for w in SPEND_WORDS)

def has_greater_than_semantics(ql: str) -> bool:
    patterns = [
        "more than","more then","greater than","greater then","above","over","at least","atleast","minimum","min ",">=","=>","≥",">"
    ]
    return any(p in ql for p in patterns)

def has_less_than_semantics(ql: str) -> bool:
    patterns = [
        "less than", "less then", "below", "under","at most", "atmost", "maximum", "max ","upto", "up to", "no more than", "not more than","<=","=<","≤","<"
    ]
    return any(p in ql for p in patterns)

def has_threshold_semantics(ql: str) -> bool:
    return has_greater_than_semantics(ql) or has_less_than_semantics(ql)

def threshold_operator_from_query(ql: str) -> Literal["gt", "ge", "lt", "le"]:
    ql = ql.strip()
    if any(p in ql for p in ["<=", "=<", "≤", "at most", "atmost", "no more than", "not more than", "up to", "upto", "maximum", "max "]):
        return "le"
    if any(p in ql for p in [">=", "=>", "≥", "at least", "atleast", "minimum", "min "]):
        return "ge"
    if any(p in ql for p in ["<", " less than", " less then", " below", " under"]):
        return "lt"
    return "gt"

def mentions_account_word(ql: str) -> bool:
    return any(x in ql for x in ["accounts", "account", "accts", "acct", "a/c", " a c "])

def is_cumulative_query(q: str) -> bool:
    ql = _normalize(q)
    keys = [
        "cumulative","cummulative","cumsum","running total","running sum","accumulated","accumulative","cumulated",
        "to date","ytd","mtd","rolling sum","progressive"
    ]
    if any(k in ql for k in keys):
        return True
    superlatives = ["highest","maximum","top","max ","minimum","lowest","least","min ","smallest"]
    if any(s in ql for s in superlatives) and mentions_spendish(ql):
        return True
    return False

def detect_order(ql: str) -> Literal["highest","lowest"]:
    if any(w in ql for w in ["minimum","lowest","least","min ","smallest","lower","minimal"]):
        return "lowest"
    return "highest"

# ───────────────────────────────────────────────────────────────────────────────
# 5) Azure OpenAI (optional) — intent extraction / agent
# ───────────────────────────────────────────────────────────────────────────────
class CumIntent(BaseModel):
    intent: Literal["cumulative","other"] = "other"
    metric: Optional[Literal["count_over_threshold","list_running","top_account"]] = None
    window: Optional[Literal["last_month","last_3_months","last_year","this_year","this_month","today","all"]] = None
    threshold: Optional[float] = None
    merchant: Optional[str] = None
    order: Optional[Literal["highest","lowest"]] = None
    top_k: Optional[int] = 1

INTENT_SYSTEM = """You are an intent extraction assistant.
Return JSON for:
- intent="cumulative" if the user asks about cumulative/cummulative/cumsum/running total/sum/accumulated/progressive/YTD/MTD/to date.
- metric: "count_over_threshold" \\ "list_running" \\ "top_account"
- window: last_month \\ last_3_months \\ last_year \\ this_year \\ this_month \\ today \\ all
- threshold: number (ignore currency symbols)
- merchant: free text if explicitly mentioned (do not invent)
- order: "highest" \\ "lowest"
- top_k: integer for "top N", default 1
If not cumulative, use intent="other".
"""

@st.cache_resource(show_spinner=False)
def get_intent_llm() -> Optional["AzureChatOpenAI"]:
    if AzureChatOpenAI and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_KEY:
        return AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0,
        )
    return None

@st.cache_resource(show_spinner=False)
def build_agent(df_: pd.DataFrame):
    if df_ is None:
        return None, "No data loaded."
    if not ALLOW_CODE or not create_pandas_dataframe_agent or not get_intent_llm():
        return None, ("Agent disabled or Azure OpenAI settings missing. Deterministic engine still works. "
                      "Set AZURE_OPENAI_* in `.env` to enable agent.")
    llm = get_intent_llm()
    agent_ = create_pandas_dataframe_agent(
        llm, df_,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        verbose=False,
        number_of_head_rows=int(HEAD_ROWS),
        max_iterations=int(MAX_ITER),
        prefix=f"""
You are a data-only assistant for a pandas DataFrame named `df`
with columns: ['account_number','transaction_date','mcc_code','merchant_name','transaction_amount'].
Today's date is {datetime.now():%Y-%m-%d}. Use ONLY the provided DataFrame.
If the question cannot be answered strictly from `df`, reply exactly: Don't Found.
Definitions:
- "this year": Jan 1–today of {CURRENT_YEAR}.
- "last year": Jan 1–Dec 31 of {CURRENT_YEAR-1}.
- "this month": first day of this month to today.
- "last 3 months": the three COMPLETE months before this month (exclude current month).
- "today": only today's date.
Cumulative spend: sort by account_number then transaction_date and compute
group-wise running total on 'transaction_amount' when explicitly requested.
Offers:
- Offer 1: previous month total ≥ 3000.
- Offer 2: last 3 complete months total ≥ 5000.
- Offer 3: last year total ≥ 10000 AND at least one transaction with 'Singapore Airlines'.
All monetary values are INR (ignore currency symbols).
Dataframe index shuld be start from 1 rather 0.
""",
        return_intermediate_steps=False,
        handle_parsing_errors=True,
    )
    return agent_, None

agent, build_err = build_agent(df)

# ───────────────────────────────────────────────────────────────────────────────
# NEW: ChatGPT-like Suggestion System (deterministic & data-aware)
# ───────────────────────────────────────────────────────────────────────────────
def _recent_window_for_suggestions() -> Tuple[date, date]:
    return last_3_complete_months()

def _top_merchants(frame: pd.DataFrame, win: Tuple[date, date], k: int = 2) -> list[str]:
    f = filter_window(frame, win)
    if f.empty:
        return []
    g = (f.assign(merchant_name_norm=f["merchant_name"].astype(str).str.lower().str.strip())
           .groupby("merchant_name_norm", dropna=False)["transaction_amount"].sum()
           .sort_values(ascending=False)
           .head(k))
    return [m for m in g.index if isinstance(m, str) and m]

def _percentile_threshold(frame: pd.DataFrame, win: Tuple[date, date], q: float = 0.9) -> int:
    f = totals_by_account(filter_window(frame, win), window=None)
    if f.empty:
        return 50000
    import numpy as np
    p = float(np.percentile(f["total_spend"], q * 100))
    step = 500
    return int(step * round(p / step)) or 5000
def _offer_merchants_from_catalog(k: int = 2) -> list[str]:
    if offers_cat is None or offers_cat.empty:
        return []
    ms = (offers_cat.loc[offers_cat["merchants"].str.lower().ne("any"), "merchants"]
                      .str.lower().str.strip().dropna().unique().tolist())
    return ms[:k]

def generate_suggestions(user_query: str) -> list[str]:
    """
    Deterministic, data-aware, catalog-aware suggestions (3–6 items).
    Fully-specified thresholds & windows to avoid 'I don't know'.
    """
    if df is None or df.empty:
        return ["Offer list", "Show eligibility for Offer 1", "Distinct merchants"]
    win3 = _recent_window_for_suggestions(); win3_text = "last 3 months"
    thr90 = _percentile_threshold(df, win3, 0.90)
    thr75 = _percentile_threshold(df, win3, 0.75)
    # merchant suggestions
    tms = _top_merchants(df, win3, k=2)
    merchantSug = [f"How many accounts have TOTAL spend ≥ {thr75:,} at {m} in {win3_text}?" for m in tms]
    # offer suggestions
    offerSug = ["Offer list"]
    offerSug += [f"What offers are available for {m}?" for m in _offer_merchants_from_catalog(k=2)]
    offerSug += ["Show eligibility for Offer 1", "Show eligibility for Offer 2", "Show eligibility for Offer 3"]
    # general data exploration
    dataSug = [
        f"How many accounts have TOTAL spend ≥ {thr90:,} in {win3_text}?",
        f"Top 5 accounts by total spend in {win3_text}",
    ]
    # dedupe & limit
    out = dataSug + merchantSug + offerSug
    seen, dedup = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
        if len(dedup) >= 6:
            break
    return dedup

# ───────────────────────────────────────────────────────────────────────────────
# NEW: Offer lookup by merchant helpers (catalog-driven)
# ───────────────────────────────────────────────────────────────────────────────
def _merchant_from_query(q: str) -> Optional[str]:
    return match_merchant_in_query(q, df) if df is not None else None

def is_offer_lookup_by_merchant(q: str) -> Optional[str]:
    ql = _normalize(q)
    if "offer" in ql and any(w in ql for w in ["for", "on", "available for", "available on"]):
        m = _merchant_from_query(q)  # fuzzy match using df merchants (may return "domino's")
        if m:
            return _clean_token(m)     # normalize apostrophes → 'dominos'
        # fallback: take text after the preposition and clean it
        m2 = re.sub(r".*\b(available for|available on|for|on)\b\s*", "", ql).strip()
        m2 = _clean_token(m2)          # remove punctuation, lower-case
        return m2 or None
    return None

def _clean_token(s: str) -> str:
    # normalize brand tokens by removing punctuation & collapsing spaces
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

def offers_for_merchant_df(merchant: str) -> pd.DataFrame:
    """
    Return merchant-specific catalog entries first.
    If none found, fall back to global 'any' offers.
    """
    if offers_cat is None or offers_cat.empty:
        return pd.DataFrame(columns=["id","title","mechanic","merchants","valid_days","description"])

    mn = _clean_token(merchant)

    # Build a cleaned series on-the-fly to match variants like "domino's" vs "dominos"
    merchants_clean = (offers_cat["merchants"]
                       .astype(str).str.lower()
                       .str.replace(r"[^a-z0-9 ]", "", regex=True)
                       .str.strip())

    # STRICT: exact or contains (no 'any' yet)
    mask_specific = merchants_clean.eq(mn) | merchants_clean.str.contains(mn, na=False)

    out = offers_cat.loc[mask_specific, ["id","title","mechanic","merchants","valid_days","description"]].copy()

    # FALLBACK: if nothing matched, include 'any'
    if out.empty:
        mask_any = offers_cat["merchants_norm"].eq("any")
        out = offers_cat.loc[mask_any, ["id","title","mechanic","merchants","valid_days","description"]].copy()

    return out.reset_index(drop=True)

# ───────────────────────────────────────────────────────────────────────────────
# 6) Cached deterministic compute
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_cumulative_by(frame: pd.DataFrame, by=("account_number",), window=None) -> pd.DataFrame:
    return cumulative_by(frame, by=by, window=window)

@st.cache_data(show_spinner=False)
def cached_count_over_anytime(frame: pd.DataFrame, threshold: float, window=None) -> int:
    return count_accounts_cum_over_anytime(frame, threshold=threshold, window=window)

@st.cache_data(show_spinner=False)
def cached_accounts_over_anytime(frame: pd.DataFrame, threshold: float, window=None) -> pd.DataFrame:
    return accounts_cum_over_anytime(frame, threshold=threshold, window=window)

@st.cache_data(show_spinner=False)
def cached_top_accounts_total(frame: pd.DataFrame, window=None, merchant_norm: Optional[str] = None, top_k: int = 1, ascending: bool = False) -> pd.DataFrame:
    return top_accounts_total(frame, window=window, merchant_norm=merchant_norm, top_k=top_k, ascending=ascending)

@st.cache_data(show_spinner=False)
def cached_total_spend(frame: pd.DataFrame, window=None, merchant_norm: Optional[str] = None) -> float:
    return total_spend(frame, window=window, merchant_norm=merchant_norm)

@st.cache_data(show_spinner=False)
def cached_totals_by_account(frame: pd.DataFrame, window=None, merchant_norm: Optional[str] = None) -> pd.DataFrame:
    return totals_by_account(frame, window=window, merchant_norm=merchant_norm)

@st.cache_data(show_spinner=False)
def cached_list_accounts_total_over(frame: pd.DataFrame, threshold: float, window=None, merchant_norm: Optional[str] = None) -> pd.DataFrame:
    return list_accounts_total_over(frame, threshold=threshold, window=window, merchant_norm=merchant_norm)

# NEW caches: cumulative operator-aware
@st.cache_data(show_spinner=False)
def cached_count_cum_compare_anytime(frame: pd.DataFrame, threshold: float, op: str, window=None) -> int:
    return count_accounts_cum_compare_anytime(frame, threshold=threshold, op=op, window=window)

@st.cache_data(show_spinner=False)
def cached_accounts_cum_compare_anytime(frame: pd.DataFrame, threshold: float, op: str, window=None) -> pd.DataFrame:
    return accounts_cum_compare_anytime(frame, threshold=threshold, op=op, window=window)

# NEW caches: TOTAL operator-aware
@st.cache_data(show_spinner=False)
def cached_list_accounts_total_compare(
    frame: pd.DataFrame, threshold: float, op: str, window=None, merchant_norm: Optional[str] = None
) -> pd.DataFrame:
    return list_accounts_total_compare(frame, threshold=threshold, op=op, window=window, merchant_norm=merchant_norm)

@st.cache_data(show_spinner=False)
def cached_count_accounts_total_compare(
    frame: pd.DataFrame, threshold: float, op: str, window=None, merchant_norm: Optional[str] = None
) -> int:
    return count_accounts_total_compare(frame, threshold=threshold, op=op, window=window, merchant_norm=merchant_norm)

# ───────────────────────────────────────────────────────────────────────────────
# Proactive Insights Section
# ───────────────────────────────────────────────────────────────────────────────
if df is not None:
    st.markdown("## 🔔 Proactive Insights")
    try:
        win = this_month_window()
        res = accounts_cum_compare_anytime(df, threshold=50000, op="gt", window=win)
        if not res.empty:
            st.warning(f"Accounts with cumulative spend > ₹50,000 this month ({win[0]} to {win[1]}):")
            res.index = range(1, len(res) + 1)
            st.dataframe(res, use_container_width=True)
        offers_snapshot = {
            "Offer 1": len(offer1_prev_month(df)),
            "Offer 2": len(offer2_last_3_months(df)),
            "Offer 3": len(offer3_last_year(df))
        }
        st.info(f"Offer eligibility snapshot: {offers_snapshot}")
    except Exception as e:
        st.error(f"Error computing proactive insights: {e}")

# ───────────────────────────────────────────────────────────────────────────────
# 7) UI: Replay previous chat
# ───────────────────────────────────────────────────────────────────────────────
for m in st.session_state["messages"]:
    role = m.get("role", "assistant")
    mtype = m.get("type", "text")
    if role == "user":
        st.chat_message("user").write(m.get("content", ""))
    else:
        with st.chat_message("assistant"):
            if mtype == "dataframe":
                try:
                    df_msg = pd.DataFrame(m.get("data", []), columns=m.get("columns", []))
                except Exception:
                    df_msg = pd.DataFrame(m.get("data", []))
                df_msg.index = range(1, len(df_msg) + 1)
                st.dataframe(df_msg, use_container_width=True)
                cap = m.get("caption")
                if cap: st.caption(cap)
            else:
                st.write(m.get("content", ""))

# ───────────────────────────────────────────────────────────────────────────────
# 8) Chat Input + Router (with priority cumulative)
# ───────────────────────────────────────────────────────────────────────────────
user_q = st.chat_input("Ask your query")
queued_user = st.session_state.pop("queued_user", None)
if not user_q and queued_user:
    user_q = queued_user

def _window_from_query(q: str) -> Optional[Tuple[date, date]]:
    return parse_time_window(q) or parse_explicit_window(q)

def _merchant_from_query_local(q: str) -> Optional[str]:
    return match_merchant_in_query(q, df)

def _apply_filters(base: pd.DataFrame, win: Optional[Tuple[date, date]], merchant_norm: Optional[str], accounts: Optional[List[str]]) -> pd.DataFrame:
    f = base.copy()
    if win is not None:
        f = filter_window(f, win)
    if merchant_norm:
        f = filter_by_merchant(f, merchant_norm)
    if accounts:
        f = f[f["account_number"].isin(accounts)]
    return f

if user_q:
    st.session_state["messages"].append({"role": "user", "type": "text", "content": user_q})
    st.chat_message("user").write(user_q)
    with st.chat_message("assistant"):
        if df is None:
            st.error("No data loaded.")
            add_assistant_text("No data loaded.")
            st.stop()

        q = user_q.strip()
        ql = _normalize(q)

        # A) Greetings
        if is_greeting(q):
            text = "Hi there! 👋"
            st.write(text)
            add_assistant_text(text)

        # B) Offer lookup by merchant (catalog-driven) — NEW ROUTER BRANCH
        elif is_offer_lookup_by_merchant(q):
            m_lookup = is_offer_lookup_by_merchant(q)
            res = offers_for_merchant_df(m_lookup or "")
            if res.empty:
                st.write("Not Found"); add_assistant_text("Not Found")
            else:
                res.index = range(1, len(res) + 1)
                st.dataframe(res, use_container_width=True)
                add_assistant_df(res, caption=f"Offers relevant to '{m_lookup}'")

        # B1) Offer list (catalog-driven)
        elif is_offer_list_query(q):
            out = offer_list_df()
            out.index = range(1, len(out) + 1)
            st.dataframe(out, use_container_width=True)
            add_assistant_df(out)

        
        # B2) Offer detail (any N, catalog-driven)
        elif is_offer_detail_query(q):
            n = parse_offer_number(q)
            if n is None:
                out = offer_list_df()
                out.index = range(1, len(out) + 1)
                st.dataframe(out, use_container_width=True)
                add_assistant_df(out)
            else:
                det = offer_details_df(n)
                if det.empty:
                    st.write("Not Found"); add_assistant_text("Not Found")
                else:
                    st.subheader(f"Offer {n} — Details")
                    det.index = range(1, len(det) + 1)
                    st.dataframe(det, use_container_width=True)
                    add_assistant_df(det)
      
        
        # C) Offer eligibility (existing deterministic O1/O2/O3)
        elif is_offers_eligibility_query(q):
            try:
                if 'offer 1' in ql:
                    st.subheader("Offer 1 — prev month ≥ 3000")
                    res = offer1_prev_month(df)
                elif 'offer 2' in ql:
                    st.subheader("Offer 2 — last 3 complete months ≥ 5000")
                    res = offer2_last_3_months(df)
                elif 'offer 3' in ql:
                    st.subheader("Offer 3 — last year ≥ 10000 + Singapore Airlines")
                    res = offer3_last_year(df)
                else:
                    st.subheader("Offer 1 — prev month ≥ 3000")
                    st.dataframe(offer1_prev_month(df), use_container_width=True)
                    st.subheader("Offer 2 — last 3 complete months ≥ 5000")
                    st.dataframe(offer2_last_3_months(df), use_container_width=True)
                    st.subheader("Offer 3 — last year ≥ 10000 + Singapore Airlines")
                    st.dataframe(offer3_last_year(df), use_container_width=True)
                    res = None

                if res is not None:
                    if res.empty:
                        st.write("Not Found")
                        add_assistant_text("Not Found")
                    else:
                        res.index = range(1, len(res) + 1)
                        st.dataframe(res, use_container_width=True)
                        add_assistant_df(res)
            except Exception as e:
                st.error(f"Error while computing eligibility: {e}")
                add_assistant_text(f"Error while computing eligibility: {e}")

        
        # D) PRIORITY: Explicit/implicit cumulative questions FIRST
        elif is_cumulative_query(q):
            try:
                # derive parameters (intent optional)
                win_cum = parse_time_window(q) or parse_explicit_window(q) or _window_from_query(q)
                thr_cum = parse_threshold(q)
                merchant_cum = _merchant_from_query_local(q)

                # metric detection
                metric = None
                if ("how many" in ql and mentions_account_word(ql) and (mentions_spendish(ql) or "cumulative" in ql)
                    and (has_threshold_semantics(ql) or re.search(r"\d", ql))):
                    metric = "count_over_threshold"
                if metric is None:
                    if ("which account" in ql and mentions_spendish(ql)) or any(w in ql for w in ["highest","maximum","top","minimum","lowest","least"]):
                        metric = "top_account"
                    else:
                        metric = "list_running"

                if DEBUG_ROUTING:
                    st.caption(
                        f"🧭 metric={metric or '-'} threshold={thr_cum if thr_cum is not None else '-'} "
                        f"window={(win_cum[0], win_cum[1]) if win_cum else 'all'} merchant={merchant_cum or 'ALL'}"
                    )

                if metric == "count_over_threshold":
                    df_eff = filter_by_merchant(df, merchant_cum)
                    if thr_cum is None:
                        st.write("I don't know"); add_assistant_text("I don't know")
                    else:
                        op = threshold_operator_from_query(ql)  # "gt", "ge", "lt", "le"
                        cnt = cached_count_cum_compare_anytime(df_eff, threshold=thr_cum, op=op, window=win_cum)
                        st.write(str(cnt)); add_assistant_text(str(cnt))
                        details_df = cached_accounts_cum_compare_anytime(df_eff, threshold=thr_cum, op=op, window=win_cum)
                        if not details_df.empty:
                            sym = _symbol_for_op(op)
                            cap = (
                                f"Accounts whose MAX running cumulative {sym} {thr_cum:,.0f} "
                                + (f"from {win_cum[0]} to {win_cum[1]}" if win_cum else "(all data)")
                                + (f" — filtered by {merchant_cum}" if merchant_cum else "")
                            )
                            details_df.index = range(1, len(details_df) + 1)
                            st.dataframe(details_df, use_container_width=True); st.caption(cap); add_assistant_df(details_df, caption=cap)

                elif metric == "top_account":
                    order = detect_order(ql)
                    ascending = (order == "lowest")
                    top_k = parse_top_k(q)
                    # FIX: honor top_k
                    top_df = cached_top_accounts_total(df, window=win_cum, merchant_norm=merchant_cum, top_k=top_k, ascending=ascending)
                    overall_total = cached_total_spend(df, window=win_cum, merchant_norm=merchant_cum)
                    if top_df.empty:
                        st.write("Not Found"); add_assistant_text("Not Found")
                    else:
                        label = "Minimum" if ascending else "Top"
                        head = (
                            f"{label} account(s) by total spend"
                            + (f" at {merchant_cum}" if merchant_cum else "")
                            + (f" for {win_cum[0]} to {win_cum[1]}" if win_cum else "")
                        )
                        st.write(head)
                        if merchant_cum:
                            total_line = (
                                f"Overall total spend at **{merchant_cum}** "
                                + (f"from **{win_cum[0]}** to **{win_cum[1]}**" if win_cum else "**(all data)**")
                                + f": **₹{overall_total:,.2f}**"
                            )
                            st.caption(total_line)
                            add_assistant_text(head + " \n" + total_line)
                        else:
                            add_assistant_text(head)
                        top_df.index = range(1, len(top_df) + 1)
                        st.dataframe(top_df, use_container_width=True)
                        add_assistant_df(top_df, caption=(f"Filtered by {merchant_cum}") if merchant_cum else None)

                else:
                    df_eff = filter_by_merchant(df, merchant_cum)
                    cum = cached_cumulative_by(df_eff, by=("account_number",), window=win_cum)
                    if cum.empty:
                        st.write("Not Found"); add_assistant_text("Not Found")
                    else:
                        cap = (
                            "Cumulative by account_number "
                            + (f"for {win_cum[0]} to {win_cum[1]}" if win_cum else "(all data)")
                            + (f" — filtered by {merchant_cum}" if merchant_cum else "")
                        )
                        cum.index = range(1, len(cum) + 1)
                        st.dataframe(cum, use_container_width=True); st.caption(cap); add_assistant_df(cum, caption=cap)

                st.session_state["cum_context"] = {"window": win_cum, "metric": metric, "threshold": thr_cum, "merchant": merchant_cum}
            except Exception as e:
                st.error(f"Error while computing cumulative analytics: {e}")
                add_assistant_text(f"Error while computing cumulative analytics: {e}")

        # E) Deterministic engine for non-cumulative questions (TOTAL spend, etc.)
        else:
            # Common parses
            win = _window_from_query(q)
            merchant_norm = _merchant_from_query_local(q)
            acct_list = parse_accounts(q)
            top_k = parse_top_k(q)
            threshold = parse_threshold(q)
            handled = False

            # E1) DISTINCTS
            if mentions_distinct(ql):
                if "merchant" in ql:
                    if "how many" in ql or "count" in ql or "number of" in ql:
                        n = df["merchant_name"].nunique()
                        st.write(str(int(n)))
                        add_assistant_text(str(int(n)))
                    else:
                        uniq = df["merchant_name"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
                        out = pd.DataFrame({"merchant_name": uniq})
                        out.index = range(1, len(out) + 1)
                        st.dataframe(out, use_container_width=True)
                        add_assistant_df(out)
                    handled = True
                elif "account" in ql or "accounts" in ql:
                    if "how many" in ql or "count" in ql or "number of" in ql:
                        n = df["account_number"].nunique()
                        st.write(str(int(n)))
                        add_assistant_text(str(int(n)))
                    else:
                        uniq = df["account_number"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
                        out = pd.DataFrame({"account_number": uniq})
                        out.index = range(1, len(out) + 1)
                        st.dataframe(out, use_container_width=True)
                        add_assistant_df(out)
                    handled = True

            # E2) TRANSACTION COUNTS (sales)
            if not handled and (mentions_transactions(ql) and ("how many" in ql or "count" in ql or "number of" in ql)):
                f = _apply_filters(df, win, merchant_norm, acct_list or None)
                st.write(str(int(len(f))))
                add_assistant_text(str(int(len(f))))
                handled = True

            # E3) SUMS — “how much amount spent…”
            if not handled and (mentions_spendish(ql) or ql.startswith("sum ")):
                if ("how many" in ql and mentions_account_word(ql) and has_threshold_semantics(ql) and threshold is not None):
                    op = threshold_operator_from_query(ql)
                    cnt = cached_count_accounts_total_compare(df, threshold=threshold, op=op, window=win, merchant_norm=merchant_norm)
                    st.write(str(cnt))
                    add_assistant_text(str(cnt))
                    res = cached_list_accounts_total_compare(df, threshold=threshold, op=op, window=win, merchant_norm=merchant_norm)
                    if not res.empty:
                        sym = _symbol_for_op(op)
                        cap = (
                            f"Accounts with TOTAL spend {sym} {threshold:,.0f} "
                            + (f"from {win[0]} to {win[1]}" if win else "(all data)")
                            + (f" — filtered by {merchant_norm}" if merchant_norm else "")
                        )
                        res.index = range(1, len(res) + 1)
                        st.dataframe(res, use_container_width=True)
                        st.caption(cap)
                        add_assistant_df(res, caption=cap)
                    handled = True
                elif ("list" in ql and mentions_account_word(ql) and has_threshold_semantics(ql) and threshold is not None):
                    op = threshold_operator_from_query(ql)
                    res = cached_list_accounts_total_compare(df, threshold=threshold, op=op, window=win, merchant_norm=merchant_norm)
                    if res.empty:
                        st.write("Not Found"); add_assistant_text("Not Found")
                    else:
                        sym = _symbol_for_op(op)
                        cap = (
                            f"{len(res)} account(s) with TOTAL spend {sym} {threshold:,.0f} "
                            + (f"from {win[0]} to {win[1]}" if win else "(all data)")
                            + (f" — filtered by {merchant_norm}" if merchant_norm else "")
                        )
                        res.index = range(1, len(res) + 1)
                        st.dataframe(res, use_container_width=True)
                        st.caption(cap)
                        add_assistant_df(res, caption=cap)
                    handled = True
                elif acct_list:
                    f = _apply_filters(df, win, merchant_norm, acct_list)
                    total = float(f["transaction_amount"].sum()) if not f.empty else 0.0
                    st.write(f"{total:.2f}")
                    add_assistant_text(f"{total:.2f}")
                    handled = True
                elif ("which account has maximum cumulative spend" in ql) or ("highest cumulative spend" in ql):
                    order = "highest"
                    ascending = False
                    top_df = cached_top_accounts_total(df, window=win, merchant_norm=merchant_norm, top_k=top_k, ascending=ascending)
                    if top_df.empty:
                        st.write("Not Found"); add_assistant_text("Not Found")
                    else:
                        top_df.index = range(1, len(top_df) + 1)
                        st.dataframe(top_df, use_container_width=True); add_assistant_df(top_df)
                    handled = True
                else:
                    if "cumulative" in ql:
                        pass
                    else:
                        f = _apply_filters(df, win, merchant_norm, None)
                        total = float(f["transaction_amount"].sum()) if not f.empty else 0.0
                        st.write(f"{total:.2f}")
                        add_assistant_text(f"{total:.2f}")
                        handled = True

            # E4) MIN/MAX transaction AMOUNT (single txn)
            if not handled and mentions_transactions(ql) and any(k in ql for k in ["minimum", "maximum", "lowest", "highest", "max ", "min "]):
                f = _apply_filters(df, win, merchant_norm, acct_list or None)
                if f.empty:
                    st.write("Not Found"); add_assistant_text("Not Found")
                else:
                    if detect_order(ql) == "lowest":
                        row = f.loc[f["transaction_amount"].idxmin()].to_dict()
                        label = "Minimum transaction"
                    else:
                        row = f.loc[f["transaction_amount"].idxmax()].to_dict()
                        label = "Maximum transaction"
                    out = pd.DataFrame([row])
                    st.write(f"{label}: ₹{row['transaction_amount']:.2f}")
                    out.index = range(1, len(out) + 1)
                    st.dataframe(out[["account_number","transaction_date","merchant_name","transaction_amount"]], use_container_width=True)
                    add_assistant_df(out[["account_number","transaction_date","merchant_name","transaction_amount"]], caption=label)
                handled = True

            # E5) “which account make highest transaction?”
            if not handled and ("which account" in ql and "highest" in ql and mentions_transactions(ql) and "cumulative" not in ql):
                f = _apply_filters(df, win, merchant_norm, None)
                if f.empty:
                    st.write("Not Found"); add_assistant_text("Not Found")
                else:
                    row = f.loc[f["transaction_amount"].idxmax()]
                    st.write(f"{row['account_number']} (₹{row['transaction_amount']:.2f})")
                    add_assistant_text(f"{row['account_number']} (₹{row['transaction_amount']:.2f})")
                handled = True

            # E6) “maximum and minimum number of transaction done by which account?”
            if not handled and ("number of transaction" in ql or ("transactions" in ql and any(w in ql for w in ["maximum","minimum","highest","lowest"]))):
                f = _apply_filters(df, win, merchant_norm, None)
                if f.empty:
                    st.write("Not Found"); add_assistant_text("Not Found")
                else:
                    counts = f.groupby("account_number", dropna=False)["transaction_amount"].size().reset_index(name="txn_count")
                    max_row = counts.loc[counts["txn_count"].idxmax()]
                    min_row = counts.loc[counts["txn_count"].idxmin()]
                    out = pd.DataFrame([max_row, min_row]).reset_index(drop=True)
                    st.dataframe(out, use_container_width=True)
                    cap = (f"From {win[0]} to {win[1]}" if win else "(all data)") + (f" — filtered by {merchant_norm}" if merchant_norm else "")
                    st.caption(cap)
                    add_assistant_df(out, caption=cap)
                handled = True

            # E7) Per-account transaction COUNT
            if not handled and acct_list and mentions_transactions(ql) and ("how many" in ql or "count" in ql):
                f = _apply_filters(df, win, merchant_norm, acct_list)
                cnt = int(len(f))
                st.write(str(cnt)); add_assistant_text(str(cnt))
                handled = True

            # F) Fallback → agent or Not Found
            if not handled:
                if build_err:
                    st.error(build_err); add_assistant_text(build_err)
                elif agent is None:
                    st.error("Agent unavailable."); add_assistant_text("Agent unavailable.")
                else:
                    try:
                        with st.spinner("Thinking…"):
                            response = agent.run(q)
                    except Exception as e:
                        response = f"Error: {e}"
                    if isinstance(response, (dict, list)):
                        st.json(response); add_assistant_text(str(response))
                    else:
                        st.write(str(response)); add_assistant_text(str(response))

        # After answering, show proactive suggestions as buttons
        suggestions = generate_suggestions(user_q)
        if suggestions:
            st.markdown("**💡 You might also ask:**")
            cols = st.columns(len(suggestions))
            for i, s in enumerate(suggestions):
                cols[i].button(s, key=f"suggest_{i}", on_click=_queue_user_text, args=(s,))
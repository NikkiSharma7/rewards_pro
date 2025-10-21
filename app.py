# filename: app.py
# -*- coding: utf-8 -*-
"""
AI Financial Assistant — Updated (Catalog load + Dynamic Schema Q&A + LLM Narrative + Dashboard)
What's new:
- Auto-load transactions from data/ or TXN_DATA_PATH (.env)
- Dynamic schema reporting (“what type of data u have?”)
- LLM narrative summary (Azure OpenAI), safe deterministic fallback when not configured
- Quick Dashboard (KPIs + charts)
- Agent prompt now injects df.columns dynamically (no hard-coded list)
Existing functionality retained:
- Offers catalog (list, detail, lookup-by-merchant)
- Deterministic analytics (windows, totals, cumulative, thresholds)
- Proactive suggestions (now context-aware)
- Agent fallback (LangChain + Azure OpenAI) if configured
"""
from __future__ import annotations
import os
import re
import glob
import difflib
from datetime import date, datetime, timedelta
from calendar import monthrange
from typing import Optional, Tuple, Literal, List, Dict
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Optional LangChain + Azure OpenAI (used if configured)
try:
    from langchain_core.prompts import ChatPromptTemplate  # newer LC
except Exception:
    ChatPromptTemplate = None
try:
    from langchain_openai import AzureChatOpenAI
    from langchain_experimental.agents import create_pandas_dataframe_agent
except Exception:
    AzureChatOpenAI = None
    create_pandas_dataframe_agent = None

# --------------------------------------------------------------------------------------------
# 0) Configuration & Page
# --------------------------------------------------------------------------------------------
load_dotenv(override=True)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
TXN_DATA_PATH = os.getenv("TXN_DATA_PATH", "").strip()
ALLOW_CODE = True
HEAD_ROWS = int(os.getenv("AGENT_HEAD_ROWS", "10"))
MAX_ITER = int(os.getenv("AGENT_MAX_ITER", "10"))
TODAY = date.today()
CURRENT_YEAR = TODAY.year
DEBUG_ROUTING = False
st.set_page_config(page_title="AI Assistant", page_icon="💳")
st.markdown(
    """
    <h1 style='font-family: "Roboto", sans-serif; color: #FFFFFF; font-size:26px'>
        💳 Welcome to Citibank Rewards and Loyalty Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stApp {
        
         background-color: #004685;
        }
    .stApp11 {
             background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSQCvCEqMnhrmjNsxT52Murg5ggT-52I7zOgo-zTqL5_pbCNZkEE83wKE-ZCPIgAC0zk8&usqp=CAU");
             background-attachment: fixed;
             background-size: cover
         }
   
    .st-info {
        
         color: #fffff;
        }
        label {
        
         color: #fffff;
        }
         </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------------------------------------------------
# 1) Session State (light chat memory)
# --------------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "type": "text", "content": "How can I help you ?"}
    ]
if "cum_context" not in st.session_state:
    st.session_state["cum_context"] = {"window": None, "metric": None, "threshold": None, "merchant": None}

def add_assistant_text(text: str) -> None:
    st.session_state["messages"].append({"role": "assistant", "type": "text", "content": text})

def add_assistant_df(df: pd.DataFrame, caption: Optional[str] = None) -> None:
    df.index = range(1, len(df) + 1)
    st.session_state["messages"].append({
        "role": "assistant",
        "type": "dataframe",
        "columns": list(df.columns),
        "data": df.to_dict("records"),
        "caption": caption
    })

def _queue_user_text(text: str):
    st.session_state["queued_user"] = text

def add_assistant_chart(fig, caption: Optional[str] = None) -> None:
    """Persist a Plotly chart in session messages by storing its JSON."""
    try:
        import plotly.io as pio
        fig_json = fig.to_json()
    except Exception:
        fig_json = None
    st.session_state["messages"].append({
        "role": "assistant",
        "type": "chart",
        "format": "plotly_json",
        "data": fig_json,
        "caption": caption
    })

# --------------------------------------------------------------------------------------------
# 2) Data Loading — Catalog-style (no uploader; always from data/ or TXN_DATA_PATH)
# --------------------------------------------------------------------------------------------
 
import csv  # used for delimiter sniffing (kept for compatibility)

# Canonical schema we want everywhere
CANONICAL_COLS = ["account_number", "transaction_date", "mcc_code", "merchant_name", "transaction_amount"]

# Synonyms for flexible column names (kept small but practical)
SYNONYMS = {
    "account_number": {"account_number", "account_no", "acct_no", "account", "acct", "card_account"},
    # If you want 'customer_id' to be mapped to account_number, add "customer_id" above.
    "transaction_date": {"transaction_date", "txn_date", "date", "purchase_date", "posted_date", "trans_date", "datetime"},
    "mcc_code": {"mcc_code", "mcc", "merchant_category_code", "category_code"},
    "merchant_name": {"merchant_name", "merchant", "store", "shop", "vendor", "brand"},
    "transaction_amount": {"transaction_amount", "amount", "amt", "value", "price", "spend"},
    # debit/credit fallbacks if transaction_amount is missing
    "_debit": {"debit", "debit_amount", "withdrawal", "dr"},
    "_credit": {"credit", "credit_amount", "deposit", "cr", "refund_amount"},
    # optional card (if present, we’ll keep it)
    "card_name": {"card_name", "card", "product", "product_name"},
}

def _normalize_header(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

def _canonical_for(col: str) -> Optional[str]:
    """Map an incoming column name to our canonical name using exact/stripped matching."""
    c = _normalize_header(col)
    for canonical, names in SYNONYMS.items():
        for n in names:
            if c == _normalize_header(n):
                return canonical
    # contains heuristic (light)
    for canonical, names in SYNONYMS.items():
        for n in names:
            if _normalize_header(n) in c or c in _normalize_header(n):
                return canonical
    return None

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename incoming columns to our canonical names when possible; keep extras intact."""
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        canon = _canonical_for(col)
        if canon and canon not in rename_map.values():
            rename_map[col] = canon
    out = df.rename(columns=rename_map)
    return out

def _clean_token(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

def _clean_merchant_token(s: str) -> str:
    """Lowercase + remove punctuation to harmonize brand names (“Domino’s” -> “dominos”)."""
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

# --- ensure local cleaner is available before offers catalog loader uses it ---
def _clean_token_local(s: str) -> str:
    return _clean_token(s)

@st.cache_data(ttl="2h", show_spinner="Loading data…")
def load_transactions_any(path_or_buffer) -> pd.DataFrame:
    """
    Read CSV/Excel; auto-map column synonyms; validate/synthesize canonical fields;
    coerce types; normalize merchants; sort deterministically.
    """
    name = getattr(path_or_buffer, "name", str(path_or_buffer)).lower()
    # Read CSV/Excel
    if name.endswith(".csv"):
        df = pd.read_csv(path_or_buffer)
    elif name.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
        df = pd.read_excel(path_or_buffer)
    else:
        raise ValueError(f"Unsupported file format: {name}")
    # Auto-map to our canonical names
    df = _auto_map_columns(df)
    # If transaction_amount missing, try to synthesize from debit/credit
    if "transaction_amount" not in df.columns:
        debit_col = next((c for c in df.columns if _canonical_for(c) == "_debit"), None)
        credit_col = next((c for c in df.columns if _canonical_for(c) == "_credit"), None)
        deb = pd.to_numeric(df[debit_col], errors="coerce") if debit_col in df.columns else 0.0
        cre = pd.to_numeric(df[credit_col], errors="coerce") if credit_col in df.columns else 0.0
        # Convention: positive = outflow (debit - credit)
        df["transaction_amount"] = pd.Series(deb).fillna(0.0) - pd.Series(cre).fillna(0.0)
    # Validate presence of canonical columns (mcc_code can be absent → we add NaN)
    if "mcc_code" not in df.columns:
        df["mcc_code"] = pd.NA
    missing = set(CANONICAL_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns after normalization: "
            + ", ".join(sorted(missing))
            + f"\nDetected columns: {', '.join(map(str, df.columns))}"
        )
    # Type coercions
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    # If your files are mostly DD/MM/YYYY, uncomment the next line for India-style dates:
    # df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", dayfirst=True)
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce").fillna(0.0)
    # MCC as nullable Int; safe for filters like "MCC 5411"
    try:
        df["mcc_code"] = pd.to_numeric(df["mcc_code"], errors="coerce").astype("Int32")
    except Exception:
        pass  # leave as-is if coercion fails
    # Merchant normalization for matching
    df["merchant_name_norm"] = df["merchant_name"].apply(_clean_merchant_token)
    # Sort deterministically
    df = df.sort_values(
        ["account_number", "transaction_date", "merchant_name", "transaction_amount"],
        kind="mergesort"
    ).reset_index(drop=True)
    return df

def _discover_transactions_path() -> Optional[str]:
    """
    Resolve the transactions file path in this order:
    1) Env var TXN_DATA_PATH
    2) data/citi_5years_transactions.csv
    3) First CSV/Excel in ./data/
    """
    if TXN_DATA_PATH and os.path.exists(TXN_DATA_PATH):
        return TXN_DATA_PATH
    default_path = "data/citi_5years_transactions.csv"
    if os.path.exists(default_path):
        return default_path
    candidates = []
    for pat in ("data/*.csv", "data/*.xls", "data/*.xlsx", "data/*.xlsm", "data/*.xlsb"):
        candidates.extend(glob.glob(pat))
    return candidates[0] if candidates else None

@st.cache_data(ttl="2h", show_spinner="Loading transactions…")
def load_transactions_catalog(path: Optional[str] = None) -> pd.DataFrame:
    """Catalog-style loader (no uploader) mirroring offers catalog behavior."""
    use_path = path or _discover_transactions_path()
    if not use_path:
        raise FileNotFoundError(
            "No transactions file found. Put one at `data/citi_5years_transactions.csv` or set TXN_DATA_PATH."
        )
    df_local = load_transactions_any(use_path)
    return df_local

# Actually load the data
df: Optional[pd.DataFrame] = None
try:
    df = load_transactions_catalog()
    #st.info(f"Using local transactions: `{os.path.basename(_discover_transactions_path() or '')}`")
    #with st.expander("Preview data", expanded=False):
        #st.dataframe(df.head(25), use_container_width=True)
except Exception as e:
    st.error(f"Could not load transactions: {e}")
    df = None

# --------------------------------------------------------------------------------------------
# Offers Catalog (catalog-driven UI)
# --------------------------------------------------------------------------------------------
@st.cache_data(ttl="2h", show_spinner=False)
def load_offers_catalog(path: str = "data/offers_catalog_updated.csv") -> Optional[pd.DataFrame]:
    """
    Reads offers catalog from CSV and normalizes fields used by the app:
    - merchants_norm: cleaned lowercased merchant tokens
    - id_norm: normalized offer id like 'o1'
    - required_merchant_norm: cleaned token to match df.merchant_name_norm
    - threshold_amount: numeric coercion
    """
    try:
        cat = pd.read_csv(path)
        # Ensure consistent types
        for c in [
            "id", "title", "merchants", "mechanic", "description",
            "valid_days", "eligibility", "eligibility_window",
            "threshold_amount", "required_merchant", "cashback_pct", "cap", "min_cart",
            "expected_uplift_pct"
        ]:
            if c in cat.columns:
                cat[c] = cat[c].astype(str)
        # Normalized IDs and merchants
        cat["id_norm"] = cat["id"].astype(str).str.strip().str.lower()
        cat["merchants_norm"] = (
            cat["merchants"].astype(str).str.lower()
            .str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
        )
        # Required merchant normalized to match df['merchant_name_norm']
        if "required_merchant" in cat.columns:
            cat["required_merchant_norm"] = cat["required_merchant"].apply(
                lambda x: _clean_token_local(x) if pd.notna(x) and str(x).strip() != "" else pd.NA
            )
        # Numeric threshold
        if "threshold_amount" in cat.columns:
            cat["threshold_amount"] = pd.to_numeric(cat["threshold_amount"], errors="coerce")
        return cat
    except Exception as e:
        st.warning(f"Offers catalog not loaded: {e}")
        return None

offers_cat = load_offers_catalog()

# --------------------------------------------------------------------------------------------
# 3) Time Windows & Filters (used by dashboard and analytics)
# --------------------------------------------------------------------------------------------
def month_begin(d: date) -> date:
    return d.replace(day=1)

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

def filter_window(frame: pd.DataFrame, window: Tuple[date, date]) -> pd.DataFrame:
    start, end = window
    mask = (frame["transaction_date"] >= pd.to_datetime(start)) & (frame["transaction_date"] <= pd.to_datetime(end))
    return frame.loc[mask].copy()

# --------------------------------------------------------------------------------------------
# 4) Parsers, Normalization & NEW Schema Q&A
# --------------------------------------------------------------------------------------------
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1)}
MONTHS.update({m[:3].lower(): i for m, i in MONTHS.items()})  # Jan, Feb, ...

def _normalize(q: str) -> str:
    ql = q.lower()
    ql = re.sub(r"[\#\*\n]+", " ", ql)
    ql = ql.replace("mnth", "month").replace("mth", "month")
    ql = ql.replace("yr", "year").replace(" ytd", " ytd")
    ql = ql.replace("in the ", " ").replace(" during ", " ").replace(" in ", " ")
    ql = ql.replace("weak", "week").replace("wek", "week").replace("weaks", "weeks")
    ql = ql.replace("wtd", "week to date")
    ql = ql.replace("todays", "today").replace("tody", "today")
    ql = ql.replace("yester day", "yesterday")
    ql = ql.replace("offres", "offers").replace("offre", "offer").replace("ofers", "offers")
    ql = re.sub(r"\s+", " ", ql).strip()
    return ql

GREETINGS = {"hi", "hello", "hey", "hola", "namaste"}

def is_greeting(q: str) -> bool:
    return _normalize(q) in GREETINGS

# --- NEW: Schema question detector ---
def is_schema_query(q: str) -> bool:
    ql = _normalize(q)
    triggers = [
        "what type of data u have",
        "what type of data do you have",
        "what data do you have",
        "what columns do you have",
        "show schema",
        "what fields are present",
        "what headers are present",
        "list columns",
        "data dictionary",
        "describe data",
    ]
    return any(t in ql for t in triggers)

# --- NEW: Describe actual dataframe columns, dtypes, canonical mapping, non-null counts, example ---
def describe_schema_df(df_: pd.DataFrame) -> pd.DataFrame:
    def first_example(series: pd.Series):
        try:
            return series.dropna().iloc[0]
        except Exception:
            return pd.NA
    rows = []
    for col in df_.columns:
        rows.append({
            "Column": col,
            "Canonical": _canonical_for(col) or "-",
            "Type": str(df_[col].dtype),
            "Non-null": int(df_[col].notna().sum()),
            "Example": first_example(df_[col]),
        })
    out = pd.DataFrame(rows).sort_values("Column").reset_index(drop=True)
    return out

# Merchant helpers (reused)
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
    if not merchant_norm:
        return frame
    f = frame[frame["merchant_name_norm"] == merchant_norm]
    if f.empty:
        f = frame[frame["merchant_name_norm"].str.contains(re.escape(merchant_norm), na=False)]
    return f.copy()

# ---- Catalog-driven eligibility helpers ----
def window_from_token(token: Optional[str]):
    """
    Map catalog window tokens to your existing time-window functions.
    Supported tokens (case/spacing/underscore-insensitive):
    prev_month, last_month, previous_month
    last_3_months, last_three_months, previous_3_months, prev_3_months, last_3_complete_months
    last_year, previous_year, prev_year
    this_month, mtd, month_to_date
    this_year, ytd, year_to_date, current_year
    today
    """
    if not token:
        return None
    t = _normalize(token)
    if t in {"prev_month", "previous_month", "last_month"}:
        return prev_month_window()
    if t in {"last_3_months", "last_three_months", "previous_3_months", "prev_3_months", "last_3_complete_months"}:
        return last_3_complete_months()
    if t in {"last_year", "previous_year", "prev_year"}:
        return last_year_window()
    if t in {"this_month", "mtd", "month_to_date"}:
        return this_month_window()
    if t in {"this_year", "ytd", "year_to_date", "current_year"}:
        return this_year_window()
    if t in {"today"}:
        return today_window()
    return None

def eligible_accounts_for_offer_row(row: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate a single catalog offer row and return eligible accounts based on:
    - eligibility_window: time window token
    - threshold_amount: minimum cumulative spend in window (optional)
    - required_merchant: merchant string (optional); normalized to match df['merchant_name_norm']
    Rules:
    - If threshold present: account-level SUM(transaction_amount) >= threshold in window (and merchant filter if given).
    - Else if required_merchant present: >=1 txn for that merchant in window.
    - Else: no eligibility rule → empty.
    Returns columns: account_number + ('total_spend' or 'txn_count'), may include meta fields for context.
    """
    f = frame
    # 1) Apply window
    win_token = row.get("eligibility_window") if isinstance(row, (dict, pd.Series)) else None
    win = window_from_token(win_token)
    if win is not None:
        f = filter_window(f, win)
    if f.empty:
        return pd.DataFrame(columns=["account_number"])
    # 2) Merchant filter, using normalized value if provided
    req_norm = None
    if "required_merchant_norm" in row.index and pd.notna(row["required_merchant_norm"]):
        req_norm = row["required_merchant_norm"]
    elif "required_merchant" in row.index and str(row["required_merchant"]).strip():
        req_norm = _clean_token_local(row["required_merchant"])
    if req_norm:
        f = filter_by_merchant(f, req_norm)
    if f.empty:
        return pd.DataFrame(columns=["account_number"])
    # 3) Threshold or presence
    thr = None
    if "threshold_amount" in row.index:
        try:
            thr = float(row["threshold_amount"])
        except Exception:
            thr = None
    if thr is not None and not pd.isna(thr):
        g = f.groupby("account_number", dropna=False)["transaction_amount"].sum()
        out = g[g >= thr].reset_index(name="total_spend")
        return out.sort_values("total_spend", ascending=False)
    if req_norm:
        g = f.groupby("account_number", dropna=False).size().reset_index(name="txn_count")
        out = g[g["txn_count"] > 0].sort_values("txn_count", ascending=False)
        return out
    # No explicit eligibility fields → return empty
    return pd.DataFrame(columns=["account_number"])

def eligibility_for_offer_id(offer_id_or_num: str, frame: pd.DataFrame, catalog: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Evaluate one offer by its id (e.g., 'o3', 'offer 3', '3').
    """
    if catalog is None or catalog.empty:
        return pd.DataFrame(columns=["account_number"])
    q = str(offer_id_or_num).strip().lower()
    m = re.search(r"\b(?:offer\s*)?(\d+)\b", q)
    oid_norm = f"o{m.group(1)}" if m else (q if q.startswith("o") else q)
    id_col = "id_norm" if "id_norm" in catalog.columns else "id"
    row = catalog[catalog[id_col] == oid_norm]
    if row.empty:
        return pd.DataFrame(columns=["account_number"])
    r = row.iloc[0]
    out = eligible_accounts_for_offer_row(r, frame)
    if out.empty:
        return out
    # Attach context fields for display
    meta_cols = [c for c in [
        "id","title","mechanic","cashback_pct","cap","min_cart","valid_days",
        "eligibility_window","threshold_amount","required_merchant"
    ] if c in r.index]
    for c in meta_cols:
        out[c] = r[c]
    return out

def eligibility_all_offers(frame: pd.DataFrame, catalog: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Evaluate eligibility for every catalog row; returns a long table:
    account_number, id, title, plus qualifying metric.
    """
    if catalog is None or catalog.empty:
        return pd.DataFrame(columns=["account_number", "id", "title"])
    results = []
    for _, r in catalog.iterrows():
        el = eligible_accounts_for_offer_row(r, frame)
        if not el.empty:
            el = el.copy()
            el["id"] = r.get("id")
            el["title"] = r.get("title")
            # carry rule context for traceability
            for c in ["eligibility_window", "threshold_amount", "required_merchant"]:
                if c in r.index:
                    el[c] = r[c]
            results.append(el)
    if not results:
        return pd.DataFrame(columns=["account_number", "id", "title"])
    out = pd.concat(results, ignore_index=True)
    sort_cols = [c for c in ["total_spend", "txn_count"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False)
    return out

# --------------------------------------------------------------------------------------------
# Offer list/detail via catalog
# --------------------------------------------------------------------------------------------
def offer_list_df() -> pd.DataFrame:
    if offers_cat is not None and not offers_cat.empty:
        out = offers_cat[["id", "title", "description"]].copy()
        out["Offer"] = out["id"].str.upper().str.replace("O", "Offer ")
        out = out[["Offer", "title", "description"]].rename(columns={"title": "Name", "description": "Description"})
        return out.reset_index(drop=True)
    return pd.DataFrame([("No catalog","Offers catalog not loaded")], columns=["Offer","Description"])

def offer_details_df(n: int) -> pd.DataFrame:
    if offers_cat is not None and not offers_cat.empty:
        oid = f"o{n}"
        row = offers_cat[offers_cat["id_norm"] == oid]
        if not row.empty:
            cols = ["id","title","merchants","mechanic","cashback_pct","cap","min_cart","valid_days",
                    "expected_uplift_pct","description","eligibility","eligibility_window","threshold_amount","required_merchant"]
            existing = [c for c in cols if c in row.columns]
            return row[existing].reset_index(drop=True)
    return pd.DataFrame(columns=["id","title","description"])

def _merchant_from_query(q: str) -> Optional[str]:
    return match_merchant_in_query(q, df) if df is not None else None

def is_offer_lookup_by_merchant(q: str) -> Optional[str]:
    ql = _normalize(q)
    if "offer" in ql and any(w in ql for w in ["for", "on", "available for", "available on"]):
        m = _merchant_from_query(q)  # fuzzy match using df merchants
        if m:
            return _clean_token_local(m)
        m2 = re.sub(r".*\b(available for|available on|for|on)\b\s*", "", ql).strip()
        m2 = _clean_token_local(m2)
        return m2 or None
    return None

def offers_for_merchant_df(merchant: str) -> pd.DataFrame:
    if offers_cat is None or offers_cat.empty:
        return pd.DataFrame(columns=["id","title","mechanic","merchants","valid_days","description"])
    mn = _clean_token_local(merchant)
    merchants_clean = (
        offers_cat["merchants"].astype(str).str.lower()
        .str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
    )
    # BUGFIX: use OR to match exact or contains
    mask_specific = merchants_clean.eq(mn) | merchants_clean.str.contains(mn, na=False)
    out = offers_cat.loc[mask_specific, ["id","title","mechanic","merchants","valid_days","description"]].copy()
    if out.empty:
        mask_any = offers_cat["merchants_norm"].eq("any")
        out = offers_cat.loc[mask_any, ["id","title","mechanic","merchants","valid_days","description"]].copy()
    return out.reset_index(drop=True)

# --------------------------------------------------------------------------------------------
# 5) Deterministic Analytics (cumulative, totals, thresholds)
# --------------------------------------------------------------------------------------------
def sort_for_cumsum(frame: pd.DataFrame, by_cols=("account_number",), date_col="transaction_date") -> pd.DataFrame:
    return frame.sort_values(list(by_cols) + [date_col], kind="mergesort")

def cumulative_by(frame: pd.DataFrame, by=("account_number",), window: Optional[Tuple[date, date]] = None) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    if f.empty: return f
    f = sort_for_cumsum(f, by_cols=by)
    f["cumulative_amount"] = f.groupby(list(by), dropna=False)["transaction_amount"].cumsum()
    return f

def totals_by_account(frame: pd.DataFrame, window=None, merchant_norm: Optional[str] = None) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    f = f if not merchant_norm else f[f["merchant_name_norm"] == merchant_norm]
    if f.empty:
        return pd.DataFrame(columns=["account_number","total_spend"])
    sums = f.groupby("account_number", dropna=False)["transaction_amount"].sum()
    return sums.reset_index(name="total_spend").sort_values("total_spend", ascending=False)

def top_accounts_total(frame: pd.DataFrame, window=None, merchant_norm: Optional[str] = None, top_k: int = 1, ascending: bool = False) -> pd.DataFrame:
    f = frame if window is None else filter_window(frame, window)
    f = filter_by_merchant(f, merchant_norm)
    if f.empty:
        cols = ["account_number","total_spend"]
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

# --------------------------------------------------------------------------------------------
# 6) Azure OpenAI (optional) — LLM access, agent & NEW dataset narrative
# --------------------------------------------------------------------------------------------
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
    # --- DYNAMIC columns instead of hard-coded list ---
    cols_list = ", ".join([repr(c) for c in df_.columns.tolist()])
    agent_ = create_pandas_dataframe_agent(
        llm, df_,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        verbose=False,
        number_of_head_rows=int(HEAD_ROWS),
        max_iterations=int(MAX_ITER),
        prefix=f"""
You are a data-only assistant for a pandas DataFrame named `df`
with columns: [{cols_list}].
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
All monetary values are INR (ignore currency symbols).
Dataframe index shuld be start from 1 rather 0.
""",
        return_intermediate_steps=False,
        handle_parsing_errors=True,
    )
    return agent_, None

# NEW: LLM narrative of dataset (fallback deterministic if LLM unavailable)
def narrate_dataset(df_: pd.DataFrame) -> str:
    cols = ", ".join(df_.columns.tolist())
    n_txn = len(df_)
    n_accts = df_["account_number"].nunique() if "account_number" in df_.columns else df_.shape[0]
    n_merch = df_["merchant_name"].nunique() if "merchant_name" in df_.columns else 0
    dmin = pd.to_datetime(df_["transaction_date"], errors="coerce").min() if "transaction_date" in df_.columns else pd.NaT
    dmax = pd.to_datetime(df_["transaction_date"], errors="coerce").max() if "transaction_date" in df_.columns else pd.NaT
    base = (
        f"This dataset contains Citi credit-card transactions.\n"
        f"- Rows (transactions): {n_txn}\n"
        f"- Accounts: {n_accts} \nMerchants: {n_merch}\n"
        f"- Date range: {dmin.date() if pd.notna(dmin) else '-'} to {dmax.date() if pd.notna(dmax) else '-'}\n"
        f"- Columns: {cols}\n"
    )
    llm = get_intent_llm()
    if llm is None:
        return base  # deterministic fallback
    prompt = (
        "You are summarizing a Citi credit-card transactions DataFrame. "
        "Write 3–5 concise bullet points describing: coverage (dates), volume, "
        "key columns, and 1-2 notable aggregates (e.g., peak month by total spend). "
        "Do NOT invent columns; use exactly these columns:\n"
        f"{cols}\n"
        "Keep it factual and short."
    )
    try:
        if ChatPromptTemplate:
            p = ChatPromptTemplate.from_messages([("system", prompt), ("user", "Summarize the dataset.")])
            resp = llm.invoke(p.format_messages())
            text = resp.content if hasattr(resp, "content") else str(resp)
            return text or base
        else:
            resp = llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            return text or base
    except Exception as e:
        return base + f"\n(Note: LLM summary unavailable: {e})"

# Build agent (once)
agent, build_err = build_agent(df) if df is not None else (None, "No data loaded.")

# --------------------------------------------------------------------------------------------
# 7) 📊 Quick Dashboard (KPIs + charts + optional LLM narrative)
# --------------------------------------------------------------------------------------------
if df is not None:
    st.markdown("## 📊 Quick Dashboard")
    # --- Time window selector ---
    win_choice = st.selectbox(
        "Time window",
        ["All", "This month", "Last 3 complete months", "This year", "Last year"],
        index=0
    )
    win_map = {
        "This month": this_month_window(),
        "Last 3 complete months": last_3_complete_months(),
        "This year": this_year_window(),
        "Last year": last_year_window(),
    }
    df_eff = df.copy()
    if win_choice in win_map:
        df_eff = filter_window(df_eff, win_map[win_choice])

    # remember the selected window in session context for suggestions
    st.session_state["current_window_choice"] = win_choice
    st.session_state["current_window"] = win_map.get(win_choice) if win_choice in win_map else None

    # --- KPIs ---
    total_spend_v = float(df_eff["transaction_amount"].sum()) if "transaction_amount" in df_eff.columns else 0.0
    txn_count = int(len(df_eff))
    acct_count = int(df_eff["account_number"].nunique()) if "account_number" in df_eff.columns else txn_count
    merch_count = int(df_eff["merchant_name"].nunique()) if "merchant_name" in df_eff.columns else 0
    avg_txn = (total_spend_v / txn_count) if txn_count else 0.0
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total spend (₹)", f"{total_spend_v:,.0f}")
    k2.metric("Transactions", f"{txn_count:,}")
    k3.metric("Accounts", f"{acct_count:,}")
    k4.metric("Merchants", f"{merch_count:,}")
    k5.metric("Avg txn (₹)", f"{avg_txn:,.0f}")

    # --- Charts (Plotly) ---
    import plotly.express as px
    # Monthly spend trend
    if "transaction_date" in df_eff.columns and "transaction_amount" in df_eff.columns:
        m = df_eff.assign(month=pd.to_datetime(df_eff["transaction_date"], errors="coerce").dt.to_period("M"))
        m = m.groupby("month", dropna=False)["transaction_amount"].sum().reset_index()
        m["month_ts"] = m["month"].astype("datetime64[ns]")
        fig_trend = px.line(m, x="month_ts", y="transaction_amount", title="Monthly total spend (INR)")
        fig_trend.update_layout(xaxis_title="Month", yaxis_title="Spend (₹)")
        st.plotly_chart(fig_trend, use_container_width=True)
    # Top merchants by spend
    if "merchant_name" in df_eff.columns and "transaction_amount" in df_eff.columns:
        tm = (df_eff.groupby("merchant_name", dropna=False)["transaction_amount"]
              .sum().sort_values(ascending=False).head(10).reset_index())
        fig_merch = px.bar(tm, x="merchant_name", y="transaction_amount", title="Top merchants by spend")
        fig_merch.update_layout(xaxis_title="Merchant", yaxis_title="Spend (₹)")
        st.plotly_chart(fig_merch, use_container_width=True)
    # MCC split (donut)
    if "mcc_code" in df_eff.columns and "transaction_amount" in df_eff.columns:
        mc = (df_eff.groupby("mcc_code", dropna=False)["transaction_amount"].sum().reset_index())
        mc["mcc_code"] = mc["mcc_code"].astype(str)
        fig_mcc = px.pie(mc, values="transaction_amount", names="mcc_code", title="Spend by MCC", hole=.45)
        st.plotly_chart(fig_mcc, use_container_width=True)
    # Top accounts by total spend
    if "account_number" in df_eff.columns and "transaction_amount" in df_eff.columns:
        ta = (df_eff.groupby("account_number", dropna=False)["transaction_amount"]
              .sum().sort_values(ascending=False).head(10).reset_index())
        fig_acct = px.bar(ta, x="account_number", y="transaction_amount", title="Top accounts by spend")
        fig_acct.update_layout(xaxis_title="Account", yaxis_title="Spend (₹)")
        st.plotly_chart(fig_acct, use_container_width=True)

    # --- Optional: LLM narrative about the dataset ---
    try:
        summary_text = narrate_dataset(df_eff)
        with st.expander("🧠 Dataset summary", expanded=False):
            st.write(summary_text)
    except Exception as e:
        st.caption(f"(Narrative unavailable: {e})")

# --------------------------------------------------------------------------------------------
# 8) Chat UI: replay existing messages
# --------------------------------------------------------------------------------------------
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
            elif mtype == "chart":
                try:
                    import plotly.io as pio
                    if m.get("format") == "plotly_json" and m.get("data"):
                        fig = pio.from_json(m["data"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Chart unavailable")
                except Exception as e:
                    st.caption(f"(Chart render issue: {e})")
                cap = m.get("caption")
                if cap: st.caption(cap)
            else:
                st.write(m.get("content", ""))

# --------------------------------------------------------------------------------------------
# 9) Chat Input + Router (greetings + NEW schema Q&A + offers + analytics + agent fallback)
# --------------------------------------------------------------------------------------------
user_q = st.chat_input("Ask your query")
queued_user = st.session_state.pop("queued_user", None)
if not user_q and queued_user:
    user_q = queued_user

# --- Context helpers (NEW) ---
def _set_last_intent(intent: str) -> None:
    st.session_state["last_intent"] = intent

def _record_last_result(intent: str,
                        window: Optional[Tuple[date, date]],
                        metric: str,
                        dim_col: Optional[str] = None,
                        dim_label: str = "",
                        top_item: Optional[str] = None) -> None:
    """Persist summary of last computed result to guide next suggestions."""
    st.session_state["last_result"] = {
        "intent": intent,
        "window": window,
        "metric": metric,
        "dim_col": dim_col,
        "dim_label": dim_label,
        "top_item": top_item,
    }

def update_cum_context_from_query(q: str, frame: pd.DataFrame) -> None:
    """Extract lightweight context from the user query to guide suggestions."""
    try:
        win = parse_time_window(q)
        merchant = _merchant_from_query(q)
        metric = parse_metric(q)  # "sales" or "count"
        ctx = st.session_state.get("cum_context", {"window": None, "metric": None, "threshold": None, "merchant": None})
        if win is not None:
            ctx["window"] = win
        if merchant:
            ctx["merchant"] = merchant
        if metric:
            ctx["metric"] = metric
        st.session_state["cum_context"] = ctx
    except Exception:
        pass
def _window_label(win: Optional[Tuple[date, date]]) -> str:
    if not win:
        return "all time"
    s, e = win
    today = TODAY
    if (s, e) == this_month_window(today):
        return "this month"
    if (s, e) == last_3_complete_months(today):
        return "last 3 complete months"
    if (s, e) == this_year_window(today):
        return "this year"
    if (s, e) == last_year_window(today):
        return "last year"
    return f"{s:%d %b %Y}–{e:%d %b %Y}"

def _effective_window_and_label() -> Tuple[Optional[Tuple[date, date]], str]:
    """
    Decide which time window to reflect in suggestions and produce a human label.
    - If dashboard selection is "All": return (None, "all time").
    - Else prefer dashboard's current_window; fallback to cum_context.window; else None.
    """
    choice = st.session_state.get("current_window_choice")
    if choice == "All":
        return None, "all time"
    win = st.session_state.get("current_window") or st.session_state.get("cum_context", {}).get("window")
    label = _window_label(win) if win else "all time"
    return win, label

def _top_value(frame: pd.DataFrame, col: str, win: Optional[Tuple[date, date]]) -> Optional[str]:
    """Returns the top category value by sales in a window, for a given column."""
    try:
        f = frame if win is None else filter_window(frame, win)
        if f.empty or col not in f.columns or "transaction_amount" not in f.columns:
            return None
        s = f.groupby(col, dropna=False)["transaction_amount"].sum().sort_values(ascending=False)
        if s.empty:
            return None
        top_key = s.index[0]
        return str(top_key)
    except Exception:
        return None

def _display_merchant_from_last_query() -> Optional[str]:
    """
    Return the merchant as the user wrote it in the last query (preserve casing),
    falling back to normalized token if needed.
    """
    last_q = st.session_state.get("last_query", "") or ""
    if not last_q:
        return None
    # Try to find a merchant token from df using direct contains on original names
    try:
        if "merchant_name" in (df.columns if df is not None else []):
            qlow = last_q.lower()
            # pick the longest merchant that appears in query to avoid partials
            merchants = sorted(df["merchant_name"].dropna().unique().tolist(), key=lambda x: len(str(x)), reverse=True)
            for m in merchants:
                if str(m).lower() in qlow:
                    # return the exact casing found in query (best-effort)
                    idx = qlow.find(str(m).lower())
                    return last_q[idx: idx + len(str(m))]
    except Exception:
        pass
    # fallback to normalized token captured earlier
    mc = st.session_state.get("cum_context", {}).get("merchant")
    return mc

def _catalog_offer_ids() -> List[str]:
    """Return available offer IDs in 'oN' normalized form."""
    ids: List[str] = []
    try:
        if offers_cat is None or offers_cat.empty:
            return ids
        id_col = "id_norm" if "id_norm" in offers_cat.columns else "id"
        for raw in offers_cat[id_col].dropna().tolist():
            s = str(raw).strip().lower()
            ids.append(s if s.startswith("o") else f"o{s}")
    except Exception:
        pass
    return sorted(list(dict.fromkeys(ids)))  # de-dupe + stable order

def _intent_template_bank(last_intent: str,
                          win_label: str,
                          metric: str,
                          merchant_ctx: Optional[str]) -> List[str]:
    """
    Return suggestions tailored to the active intent and context.
    Keeps the phrases router-compatible with your existing handlers.
    """
    sugs: List[str] = []
    offer_ids = _catalog_offer_ids()
    merch_disp = merchant_ctx or _display_merchant_from_last_query()

    # --- OFFER-CENTRIC FLOWS ---
    if last_intent in ("offer_list", "offer_detail", "offer_lookup", "eligibility"):
        # If we have offer IDs, suggest details/eligibility next
        if offer_ids:
            # prioritize the first few offers
            head = offer_ids[:3]
            # Details for first offers
            sugs.extend([f"Offer {oid[1:]} — Details" for oid in head])              # e.g., "Offer 1 — Details"
            # Eligibility for first offers
            sugs.extend([f"Eligible accounts for Offer {oid}" for oid in head])      # e.g., "Eligible accounts for Offer o1"

        # Merchant-aware follow-ups
        if merch_disp:
            sugs.append(f"Show offers for {merch_disp}")
            sugs.append(f"Total spend for {merch_disp} {win_label}")
            sugs.append(f"Top 5 accounts for {merch_disp} {win_label}")
        else:
            # Suggest looking up offers for the top merchant this window
            sugs.append("Show offers for <merchant>")
            sugs.append(f"Top 5 merchants {win_label}")

        # A couple of analytics nudges to connect offers ↔ usage
        sugs.append(f"Compare {metric} {win_label} by merchant")
        sugs.append(f"Compare {metric} {win_label} by MCC")
        return sugs

    # --- MERCHANT-CENTRIC (user typed a brand) ---
    if merch_disp:
        sugs.append(f"Total spend for {merch_disp} {win_label}")
        sugs.append(f"Top 5 accounts for {merch_disp} {win_label}")
        sugs.append(f"Show offers for {merch_disp}")
        sugs.append(f"Compare {metric} {win_label} by MCC")
        return sugs

    # --- RANK FLOW ---
    if last_intent == "rank":
        sugs.append(f"Compare {metric} {win_label} by merchant")
        sugs.append(f"Top 5 accounts {win_label}")
        sugs.append(f"Compare {metric} {win_label} by MCC")
        sugs.append(f"Show offers for <merchant>")
        return sugs

    # --- COMPARE FLOW ---
    if last_intent == "compare":
        sugs.append(f"Top 5 merchants {win_label}")
        sugs.append(f"Top 5 accounts {win_label}")
        sugs.append(f"Compare {metric} {win_label} by MCC")
        sugs.append(f"Show offers for <merchant>")
        return sugs

    # --- SCHEMA FLOW ---
    if last_intent == "schema":
        sugs.append("Monthly spend trend")
        sugs.append(f"Top 5 merchants {win_label}")
        sugs.append(f"Compare {metric} {win_label} by merchant")
        sugs.append("Offer list")
        return sugs

    # --- DEFAULT / AGENT / FALLBACK ---
    sugs.extend([
        f"Top 5 merchants {win_label}",
        f"Compare {metric} {win_label} by merchant",
        f"Top 5 accounts {win_label}",
        f"Show offers for <merchant>",
    ])
    return sugs

def parse_offer_number(q: str) -> Optional[int]:
    ql = _normalize(q)
    m = re.search(r"\boffer\s*(\d+)\b", ql)
    if m: return int(m.group(1))
    m = re.search(r"\bo(\d+)\b", ql)
    if m: return int(m.group(1))
    return None

def is_offer_list_query(q: str) -> bool:
    ql = _normalize(q)
    return ("offer list" in ql) or ("available offers" in ql) or ("list offers" in ql) or ("predefined offers" in ql)

def is_offers_eligibility_query(q: str) -> bool:
    ql = _normalize(q)
    return (("offer" in ql) and (not is_offer_list_query(q))) or ("eligible" in ql) or ("eligibility" in ql)

def parse_time_window(q: str) -> Optional[Tuple[date, date]]:
    ql = _normalize(q)
    if "yesterday" in ql:
        return today_window(TODAY - timedelta(days=1))
    if "this week" in ql or "current week" in ql or "week to date" in ql:
        start = TODAY - timedelta(days=(TODAY.weekday() % 7))
        return (start, TODAY)
    if "last week" in ql or "previous week" in ql or "prev week" in ql or "past week" in ql:
        this_start = TODAY - timedelta(days=(TODAY.weekday() % 7))
        prev_start = this_start - timedelta(days=7)
        prev_end = this_start - timedelta(days=1)
        return (prev_start, prev_end)
    if "last 3 months" in ql or "last three months" in ql or "previous 3 months" in ql or "prev 3 months" in ql:
        return last_3_complete_months()
    if "last year" in ql or "previous year" in ql or "prev year" in ql or "last yr" in ql or "prev yr" in ql:
        return last_year_window()
    if "this year" in ql or "current year" in ql or "this yr" in ql or "ytd" in ql or "year to date" in ql:
        return this_year_window()
    if "last month" in ql or "previous month" in ql or "prev month" in ql:
        return prev_month_window()
    if "this month" in ql or "mtd" in ql or "month to date" in ql:
        return this_month_window()
    if "today" in ql:
        return today_window()
    # quarter parsing e.g., "Q1 2024", "this quarter", "last quarter"
    q_match = re.search(r"\bq([1-4])\s*(20\d{2})\b", ql)
    if "this quarter" in ql or "current quarter" in ql:
        m = TODAY.month
        qn = (m - 1) // 3 + 1
        y = TODAY.year
        start_m = 3 * (qn - 1) + 1
        end_m = start_m + 2
        last_day = monthrange(y, end_m)[1]
        return (date(y, start_m, 1), date(y, end_m, last_day))
    if "last quarter" in ql or "previous quarter" in ql:
        m = TODAY.month
        qn = (m - 1) // 3 + 1
        y = TODAY.year
        qn -= 1
        if qn == 0:
            qn = 4; y -= 1
        start_m = 3 * (qn - 1) + 1
        end_m = start_m + 2
        last_day = monthrange(y, end_m)[1]
        return (date(y, start_m, 1), date(y, end_m, last_day))
    if q_match:
        qn = int(q_match.group(1)); y = int(q_match.group(2))
        start_m = 3 * (qn - 1) + 1
        end_m = start_m + 2
        last_day = monthrange(y, end_m)[1]
        return (date(y, start_m, 1), date(y, end_m, last_day))
    # explicit year or month-year parsing
    y = re.search(r"\b(20\d{2})\b", ql)
    if y and (" 20" in f" {ql} "):
        year = int(y.group(1))
        return (date(year,1,1), date(year,12,31))
    for token in MONTHS:
        if token in ql:
            ym = re.search(rf"{token}\s+(\d{{4}})", ql)
            if ym:
                year = int(ym.group(1)); month = MONTHS[token]
                last_day = monthrange(year, month)[1]
                return (date(year,month,1), date(year,month,last_day))
            my = re.search(rf"(\d{{4}})\s+{token}", ql)  # rf: fixed syntax
            if my:
                year = int(my.group(1)); month = MONTHS[token]
                last_day = monthrange(year, month)[1]
                return (date(year,month,1), date(year,month,last_day))
    return None

# ---- Rank/Compare intent detection ----
def is_rank_query(q: str) -> bool:
    ql = _normalize(q)
    triggers = ["top", "worst", "lowest", "bottom", "most used", "popular", "most popular"]
    return any(t in ql for t in triggers) or re.search(r"\btop\s*\d+\b", ql) is not None

def is_compare_by_query(q: str) -> bool:
    ql = _normalize(q)
    # “compare ... by merchant/product/mcc/account/card ...”
    return ("compare" in ql) and (" by " in ql)

# ---- Extractors ----
def parse_n(q: str, default_top: int = 5) -> int:
    ql = _normalize(q)
    m = re.search(r"\b(top|worst|lowest|bottom)\s*(\d+)\b", ql)
    if m:
        try:
            return max(1, int(m.group(2)))
        except Exception:
            pass
    m2 = re.search(r"\b(\d+)\b", ql)
    if m2 and any(k in ql for k in ["top", "worst", "lowest", "bottom"]):
        try:
            return max(1, int(m2.group(1)))
        except Exception:
            pass
    # one-offs like "most used/popular"
    if "most used" in ql or "most popular" in ql or "popular" in ql:
        return 1
    return default_top

def parse_polarity(q: str) -> Literal["top","bottom"]:
    ql = _normalize(q)
    if any(k in ql for k in ["worst", "lowest", "bottom", "least"]):
        return "bottom"
    return "top"

def parse_metric(q: str) -> Literal["sales","count"]:
    """sales => sum(transaction_amount); count => #transactions (used/popular)."""
    ql = _normalize(q)
    if any(k in ql for k in ["used", "popular", "usage", "count", "frequency"]):
        return "count"
    # default if the user says "sales", "revenue", "spend", "amount"
    if any(k in ql for k in ["sales", "revenue", "spend", "amount", "value"]):
        return "sales"
    # fallback default
    return "sales"

def _has_col(df_: pd.DataFrame, name: str) -> bool:
    return name in df_.columns

def choose_dimension(q: str, df_: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Map user-requested entity to a dataframe column.
    Returns (column_name, pretty_label). If not found, (None, '').
    """
    ql = _normalize(q)
    # cards/products
    if any(k in ql for k in ["card", "cards", "product", "products"]):
        for cand in ["card_name", "product", "product_name"]:
            if _has_col(df_, cand):
                return cand, "Card" if "card" in ql else "Product"
    # soft fallback to merchant if nothing else specified
    if _has_col(df_, "merchant_name"):
        return "merchant_name", "Merchant"
    # merchant
    if "merchant" in ql or "merchants" in ql:
        if _has_col(df_, "merchant_name"):
            return "merchant_name", "Merchant"
    # mcc
    if "mcc" in ql:
        if _has_col(df_, "mcc_code"):
            return "mcc_code", "MCC"
    # account
    if "account" in ql or "card account" in ql:
        if _has_col(df_, "account_number"):
            return "account_number", "Account"
    # generic fallbacks by availability
    for cand, label in [("card_name","Card"), ("product","Product"), ("product_name","Product"),
                        ("merchant_name","Merchant"), ("mcc_code","MCC"), ("account_number","Account")]:
        if _has_col(df_, cand):
            return cand, label
    return None, ""

def _window_from_query_or_default(q: str, default_to_last_year: bool = False) -> Optional[Tuple[date, date]]:
    w = parse_time_window(q)
    if w:
        return w
    ql = _normalize(q)
    if default_to_last_year or "last year" in ql or "previous year" in ql:
        return last_year_window()
    return None

# ---- Aggregation helpers ----
def aggregate_by(frame: pd.DataFrame, dim_col: str, metric: Literal["sales","count"]) -> pd.DataFrame:
    f = frame.copy()
    if metric == "count":
        out = f.groupby(dim_col, dropna=False).size().reset_index(name="value")
    else:
        out = f.groupby(dim_col, dropna=False)["transaction_amount"].sum().reset_index(name="value")
    return out

def pretty_metric_name(metric: str) -> str:
    return "Transactions" if metric == "count" else "Sales (₹)"

def handle_rank_query(q: str, frame: pd.DataFrame) -> None:
    import plotly.express as px
    n = parse_n(q, default_top=5)
    polarity = parse_polarity(q)
    metric = parse_metric(q)
    dim_col, dim_label = choose_dimension(q, frame)
    if dim_col is None:
        st.write("Not Found")
        add_assistant_text("Not Found")
        return
    # time window (if any)
    win = _window_from_query_or_default(q, default_to_last_year=False)
    f = frame if win is None else filter_window(frame, win)
    if f.empty:
        st.write("Not Found")
        add_assistant_text("Not Found")
        return
    agg = aggregate_by(f, dim_col, metric)
    if agg.empty:
        st.write("Not Found")
        add_assistant_text("Not Found")
        return
    ascending = (polarity == "bottom")
    agg_sorted = agg.sort_values("value", ascending=ascending)
    sel = agg_sorted.head(n)
    sel = sel.rename(columns={dim_col: dim_label, "value": pretty_metric_name(metric)})
    sel.index = range(1, len(sel) + 1)
    # Title and subtitle
    title_prefix = "Top" if polarity == "top" else "Lowest"
    metric_name = "transactions" if metric == "count" else "sales"
    time_suffix = ""
    if win is not None:
        start, end = win
        time_suffix = f" ({start:%d %b %Y}–{end:%d %b %Y})"
    title = f"{title_prefix} {len(sel)} {dim_label.lower()}s by {metric_name}{time_suffix}"
    # Chart
    fig = px.bar(sel, x=dim_label, y=pretty_metric_name(metric), title=title)
    fig.update_layout(xaxis_title=dim_label, yaxis_title=pretty_metric_name(metric))
    # Render + persist
    st.dataframe(sel, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)
    add_assistant_df(sel, caption=title)
    add_assistant_chart(fig, caption=title)
    # Record last result for suggestions
    top_item = None
    try:
        if not sel.empty and dim_label in sel.columns:
            top_item = str(sel.iloc[0][dim_label])
    except Exception:
        top_item = None
    _record_last_result(intent="rank", window=win, metric=metric, dim_col=dim_col, dim_label=dim_label, top_item=top_item)

def handle_compare_by_query(q: str, frame: pd.DataFrame) -> None:
    """
    Examples:
    - Compare all sales data in last year by product
    - Compare all sales data this month by merchant
    Produces a distribution bar chart across categories (limited to top 10 by default).
    """
    import plotly.express as px
    metric = parse_metric(q)  # “sales” or “count”
    dim_col, dim_label = choose_dimension(q, frame)
    if dim_col is None:
        st.write("Not Found")
        add_assistant_text("Not Found")
        return
    # Window: default to last year if the phrase includes 'last year' or user expects a time window
    win = _window_from_query_or_default(q, default_to_last_year=True)
    f = frame if win is None else filter_window(frame, win)
    if f.empty:
        st.write("Not Found")
        add_assistant_text("Not Found")
        return
    agg = aggregate_by(f, dim_col, metric)
    if agg.empty:
        st.write("Not Found")
        add_assistant_text("Not Found")
        return
    # Take top 10 categories to keep the chart legible
    out = agg.sort_values("value", ascending=False).head(10)
    out = out.rename(columns={dim_col: dim_label, "value": pretty_metric_name(metric)})
    out.index = range(1, len(out) + 1)
    time_suffix = ""
    if win is not None:
        start, end = win
        time_suffix = f" ({start:%d %b %Y}–{end:%d %b %Y})"
    metric_name = "transactions" if metric == "count" else "sales"
    title = f"Comparison by {dim_label.lower()} — {metric_name}{time_suffix}"
    fig_plot = px.bar(out, x=dim_label, y=pretty_metric_name(metric), title=title)
    fig_plot.update_layout(xaxis_title=dim_label, yaxis_title=pretty_metric_name(metric))
    st.dataframe(out, use_container_width=True)
    st.plotly_chart(fig_plot, use_container_width=True)
    add_assistant_df(out, caption=title)
    add_assistant_chart(fig_plot, caption=title)
    # Record last result for suggestions
    top_item = None
    try:
        if not out.empty and dim_label in out.columns:
            top_item = str(out.iloc[0][dim_label])
    except Exception:
        top_item = None
    _record_last_result(intent="compare", window=win, metric=metric, dim_col=dim_col, dim_label=dim_label, top_item=top_item)

def parse_top_k(q: str) -> int:
    m = re.search(r"\btop\s+(\d+)\b", _normalize(q))
    if m: return int(m.group(1))
    return 1

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
        # NEW: remember last natural-language query
        st.session_state["last_query"] = q
        # Update context gleaned from this query
        update_cum_context_from_query(q, df)

        # A) Greetings
        if is_greeting(q):
            text = "Hi there! 👋"
            st.write(text)
            add_assistant_text(text)
            _set_last_intent("greeting")

        # A1) NEW: Schema / columns / data dictionary
        elif is_schema_query(q):
            _set_last_intent("schema")
            try:
                schema_df = describe_schema_df(df)
                schema_df.index = range(1, len(schema_df) + 1)
                st.subheader("Dataset schema (from loaded transactions)")
                st.dataframe(schema_df, use_container_width=True)
                add_assistant_df(schema_df, caption="Schema: actual columns, canonical mapping, types, examples")
                _record_last_result(intent="schema", window=st.session_state.get("current_window"), metric=st.session_state.get("cum_context", {}).get("metric", "sales"))
            except Exception as e:
                st.error(f"Error describing schema: {e}")
                add_assistant_text(f"Error describing schema: {e}")

        # B) Offer lookup by merchant (catalog-driven)
        elif is_offer_lookup_by_merchant(q):
            _set_last_intent("offer_lookup")
            m_lookup = is_offer_lookup_by_merchant(q)
            res = offers_for_merchant_df(m_lookup or "")
            if res.empty:
                st.write("Not Found"); add_assistant_text("Not Found")
            else:
                res.index = range(1, len(res) + 1)
                st.dataframe(res, use_container_width=True)
                add_assistant_df(res, caption=f"Offers relevant to '{m_lookup}'")
                _record_last_result(intent="offer_lookup", window=st.session_state.get("current_window"), metric=st.session_state.get("cum_context", {}).get("metric", "sales"))

        # B1) Offer list (catalog-driven)
        elif is_offer_list_query(q):
            _set_last_intent("offer_list")
            out = offer_list_df()
            out.index = range(1, len(out) + 1)
            st.dataframe(out, use_container_width=True)
            add_assistant_df(out)
            _record_last_result(intent="offer_list", window=st.session_state.get("current_window"), metric=st.session_state.get("cum_context", {}).get("metric", "sales"))

        # B2) Offer detail (any N, catalog-driven)
        elif parse_offer_number(q) is not None:
            _set_last_intent("offer_detail")
            n = parse_offer_number(q)
            det = offer_details_df(n)
            if det.empty:
                st.write("Not Found"); add_assistant_text("Not Found")
            else:
                st.subheader(f"Offer {n} — Details")
                det.index = range(1, len(det) + 1)
                st.dataframe(det, use_container_width=True)
                add_assistant_df(det)
                _record_last_result(intent="offer_detail", window=st.session_state.get("current_window"), metric=st.session_state.get("cum_context", {}).get("metric", "sales"))

        # C) Offer eligibility (existing deterministic O1/O2/O3)
        elif is_offers_eligibility_query(q):
            _set_last_intent("eligibility")
            try:
                # If the user referenced a specific offer number, compute for that one
                n = parse_offer_number(q)
                if n is not None:
                    res = eligibility_for_offer_id(f"o{n}", df, offers_cat)
                    if res is None or res.empty:
                        st.write("Not Found"); add_assistant_text("Not Found")
                    else:
                        res.index = range(1, len(res) + 1)
                        st.dataframe(res, use_container_width=True)
                        add_assistant_df(res, caption=f"Eligibility — Offer {n}")
                else:
                    # Otherwise, evaluate eligibility for all offers in the catalog
                    res_all = eligibility_all_offers(df, offers_cat)
                    if res_all is None or res_all.empty:
                        st.write("Not Found"); add_assistant_text("Not Found")
                    else:
                        res_all.index = range(1, len(res_all) + 1)
                        st.dataframe(res_all, use_container_width=True)
                        add_assistant_df(res_all, caption="Eligibility — All offers")
                _record_last_result(intent="eligibility", window=st.session_state.get("current_window"), metric=st.session_state.get("cum_context", {}).get("metric", "sales"))
            except Exception as e:
                st.error(f"Error while computing eligibility: {e}")
                add_assistant_text(f"Error while computing eligibility: {e}")

        # D) Rank
        elif is_rank_query(q):
            _set_last_intent("rank")
            try:
                handle_rank_query(q, df)
            except Exception as e:
                st.error(f"Error while ranking: {e}")
                add_assistant_text(f"Error while ranking: {e}")

        # E) Compare
        elif is_compare_by_query(q):
            _set_last_intent("compare")
            try:
                handle_compare_by_query(q, df)
            except Exception as e:
                st.error(f"Error while comparing by group: {e}")
                add_assistant_text(f"Error while comparing by group: {e}")

        # F) Fallback → agent or Not Found (uses dynamic df.columns in prompt)
        else:
            _set_last_intent("agent" if agent is not None else "fallback")
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

# --------------------------------------------------------------------------------------------
# 10) Proactive suggestion buttons (context-aware, v2)
# --------------------------------------------------------------------------------------------
def _context_aware_suggestions(max_sugs: int = 4) -> List[str]:
    """
    Context-aware suggestions v3:
    - Strongly intent-aware (offer/merchant/rank/compare/schema).
    - Merchant-aware using the user's last query text (preserve casing).
    - Rotates alternatives if previous set is identical.
    """
    if df is None or df.empty:
        return ["Show schema", "Offer list", "Top merchants by spend"][:max_sugs]

    # Window & label (respect "All")
    win, win_label = _effective_window_and_label()

    # Context primitives
    ctx = st.session_state.get("cum_context", {})
    last_intent = st.session_state.get("last_intent", "")
    metric = ctx.get("metric") or "sales"
    merchant_ctx = ctx.get("merchant")

    # Generate from intent template bank
    pool = _intent_template_bank(last_intent=last_intent, win_label=win_label, metric=metric, merchant_ctx=merchant_ctx)

    # De-dupe and trim
    seen = set()
    dedup = []
    for s in pool:
        if s and s not in seen:
            dedup.append(s)
            seen.add(s)
        if len(dedup) >= max_sugs:
            break

    # Avoid repeating identical sets across renders
    hist: List[List[str]] = st.session_state.get("suggestions_history", [])
    last_set = hist[-1] if hist else []
    if last_set and dedup == last_set:
        # Rotate with alt metric / dimension
        alt_metric = "count" if metric == "sales" else "sales"
        dedup.append(f"Compare {alt_metric} {win_label} by merchant")
        if "card_name" in (df.columns):
            dedup.append(f"Top 5 Cards {win_label}")
        # Trim again
        dedup = dedup[:max_sugs]

    hist.append(dedup)
    st.session_state["suggestions_history"] = hist[-5:]
    return dedup

sugs = _context_aware_suggestions()
if sugs:
    st.markdown("**💡 You might also ask:**")
    cols = st.columns(len(sugs))
    for i, s in enumerate(sugs):
        cols[i].button(s, key=f"suggest_{i}", on_click=_queue_user_text, args=(s,))





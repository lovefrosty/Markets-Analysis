# Equity Research Lab (Streamlit)

Interactive dashboard + auto-generated research note for U.S. equities.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional env vars:
- `FRED_API_KEY` for FRED macro series (10Y DGS10 risk-free, etc.)
- `SEC_USER_AGENT` like "you@example.com EquityResearchLab/1.0" for SEC politeness

> Educational use only. Not investment advice.


## Note on `pkg_resources`
`pkg_resources` (from setuptools) is **deprecated** in favor of `importlib.resources` and `importlib.metadata` (or their backports).  
This project does **not** use `pkg_resources`; templates are read from the filesystem and can easily be migrated to `importlib.resources` if packaged.

import os
import json
from typing import Any, Dict, Optional

import requests
import streamlit as st

"""
Streamlit ⇄ Marqo demo app (works through Cloudflare Access)
- Reads MARQO_URL and optional Cloudflare Access Service Token from
  Streamlit secrets (preferred) or environment variables.
- Lets you: health‑check, list/create/delete indexes, add simple text documents,
  and run searches.
"""

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

BASE_URL = get_secret("MARQO_URL", "https://marqo.logicafutura.com").rstrip("/")
CF_ID = get_secret("CF_ACCESS_CLIENT_ID")
CF_SECRET = get_secret("CF_ACCESS_CLIENT_SECRET")

def cf_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if CF_ID and CF_SECRET:
        headers["CF-Access-Client-Id"] = CF_ID
        headers["CF-Access-Client-Secret"] = CF_SECRET
    return headers

def http(method: str, path: str, *, json_body: Optional[Dict[str, Any]] = None, timeout: int = 30):
    url = f"{BASE_URL}{path}"
    resp = requests.request(method, url, headers=cf_headers(), json=json_body, timeout=timeout)
    try:
        data = resp.json()
    except ValueError:
        data = {"text": resp.text}
    return resp.status_code, data

st.set_page_config(page_title="Marqo Console", page_icon="🔎", layout="wide")
st.title("🔎 Marqo Console")
st.caption("Manage indexes, add docs, and search — via Cloudflare‑protected endpoint.")

with st.sidebar:
    st.header("Connection")
    st.text_input("Base URL", value=BASE_URL, help="Your Marqo URL (from Cloudflare)", key="base_url_input")
    if st.session_state.get("base_url_input"):
        BASE_URL = st.session_state["base_url_input"].rstrip("/")

    st.divider()
    st.subheader("Auth (optional)")
    st.text_input("CF-Access-Client-Id", value=CF_ID or "", type="password", key="cf_id")
    st.text_input("CF-Access-Client-Secret", value=CF_SECRET or "", type="password", key="cf_secret")
    if st.session_state.get("cf_id") and st.session_state.get("cf_secret"):
        CF_ID = st.session_state["cf_id"]
        CF_SECRET = st.session_state["cf_secret"]

    st.divider()
    if st.button("🔄 Health check"):
        code, data = http("GET", "/")
        st.toast(f"HTTP {code}")
        st.write(data)

st.subheader("📚 Indexes")
cols = st.columns([1, 1, 2])
with cols[0]:
    if st.button("List indexes", use_container_width=True):
        code, data = http("GET", "/indexes")
        st.code(json.dumps(data, indent=2))

with cols[1]:
    new_index = st.text_input("Create index (name)", placeholder="my-index")
    if st.button("Create", use_container_width=True, disabled=not new_index):
        body = {"indexName": new_index}
        code, data = http("PUT", f"/indexes/{new_index}", json_body={})
        if code >= 400:
            code, data = http("POST", "/indexes", json_body=body)
        st.toast(f"Create index {new_index}: HTTP {code}")
        st.code(json.dumps(data, indent=2))

with cols[2]:
    del_index = st.text_input("Delete index (name)", placeholder="my-index")
    if st.button("Delete", use_container_width=True, disabled=not del_index):
        code, data = http("DELETE", f"/indexes/{del_index}")
        st.toast(f"Delete index {del_index}: HTTP {code}")
        st.code(json.dumps(data, indent=2))

st.divider()

st.subheader("✍️ Add documents")
idx = st.text_input("Target index", placeholder="my-index", key="doc_index")
text = st.text_area("Text to index", placeholder="Enter a sentence or paragraph…", height=120)
doc_id = st.text_input("Optional document ID", placeholder="doc-1")

add_cols = st.columns([1, 1])
with add_cols[0]:
    if st.button("Add document", disabled=not (idx and text)):
        doc = {"_id": doc_id} if doc_id else {}
        doc.update({"text": text})
        payload = {"documents": [doc]}
        code, data = http("POST", f"/indexes/{idx}/documents", json_body=payload)
        st.toast(f"Add doc: HTTP {code}")
        st.code(json.dumps(data, indent=2))

with add_cols[1]:
    if st.button("Refresh index stats", disabled=not idx):
        code, data = http("GET", f"/indexes/{idx}")
        st.toast(f"HTTP {code}")
        st.code(json.dumps(data, indent=2))

st.divider()

st.subheader("🔍 Search")
search_idx = st.text_input("Index to search", value=idx or "", placeholder="my-index")
query = st.text_input("Query", placeholder="e.g., what is streamlit?")
top_k = st.number_input("Top K", min_value=1, max_value=100, value=5, step=1)

if st.button("Run search", disabled=not (search_idx and query)):
    payload = {"q": query, "searchMethod": "TENSOR", "limit": int(top_k)}
    code, data = http("POST", f"/indexes/{search_idx}/search", json_body=payload)
    st.toast(f"Search: HTTP {code}")
    st.code(json.dumps(data, indent=2))

st.write(f"— App connected to: `{BASE_URL}`")

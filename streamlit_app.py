import os
import io
import json
import uuid
import mimetypes
from typing import Any, Dict, Optional, List

import requests
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import boto3

# =============================================================
# Streamlit App: Visual Search with "same-type" gating (Marqo + R2)
# =============================================================

# -----------------------------
# Config (from secrets or ENV)
# -----------------------------

def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(key, default)

MARQO_URL: str = (_cfg("MARQO_URL", "https://marqo.logicafutura.com") or "").rstrip("/")
INDEX_NAME: str = _cfg("INDEX_NAME", "furniture-index") or "furniture-index"

TITLE_FIELD: str = _cfg("TITLE_FIELD", "name") or "name"
IMAGE_FIELD: str = _cfg("IMAGE_FIELD", "image") or "image"
ALT_IMAGE_FIELD: str = _cfg("ALT_IMAGE_FIELD", "image_url") or "image_url"
DESCRIPTION_FIELD: str = _cfg("DESCRIPTION_FIELD", "description") or "description"
SPEC_TEXT_FIELD: str = _cfg("SPEC_TEXT_FIELD", "spec_text") or "spec_text"
SEARCH_BLOB_FIELD: str = _cfg("SEARCH_BLOB_FIELD", "search_blob") or "search_blob"
# This is the HEX field for display, the filter will use r, g, b fields
DOM_COLOR_FIELD: str = _cfg("DOMINANT_COLOR_FIELD", "dominant_color") or "dominant_color"
OBJECT_TYPE_FIELD: str = _cfg("OBJECT_TYPE_FIELD", "object_type") or "object_type"
SKU_FIELD: str = _cfg("SKU_FIELD", "product_id") or "product_id"
CLICK_URL_FIELD: str = _cfg("CLICK_URL_FIELD", "product_url") or "product_url"

# Cloudflare Access headers (if your Marqo is behind Cloudflare Access)
CF_ACCESS_CLIENT_ID = _cfg("CF_ACCESS_CLIENT_ID")
CF_ACCESS_CLIENT_SECRET = _cfg("CF_ACCESS_CLIENT_SECRET")

# R2 (for uploading the query image and making it web-accessible to Marqo)
R2_ENDPOINT_URL: str = _cfg("R2_ENDPOINT_URL") or ""
R2_ACCESS_KEY_ID: Optional[str] = _cfg("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY: Optional[str] = _cfg("R2_SECRET_ACCESS_KEY")
R2_BUCKET: str = _cfg("R2_BUCKET", "streamlit098") or "streamlit098"
PUBLIC_BASE_URL: str = _cfg("PUBLIC_BASE_URL", "") or ""

HEADERS: Dict[str, str] = {"Content-Type": "application/json"}
if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
    HEADERS["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
    HEADERS["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET

# Tensor fields available in the current index (keep in sync with your indexer)
KNOWN_TENSOR_FIELDS = {"name", "description", "image", "spec_text", "search_blob"}


def sanitize_attrs(attrs: List[str]) -> List[str]:
    """Map aliases, drop unknowns, and ensure the list is not empty."""
    out: List[str] = []
    for a in attrs:
        a = (a or "").strip()
        if not a:
            continue
        # Map common alias
        if a == "title":
            a = "name"
        if a in KNOWN_TENSOR_FIELDS and a not in out:
            out.append(a)
    if not out:
        out = ["image"] if "image" in KNOWN_TENSOR_FIELDS else list(KNOWN_TENSOR_FIELDS)
    return out

# =============================================================
# R2 helpers (for uploading the query image)
# =============================================================

def _r2_client():
    if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_ENDPOINT_URL):
        raise RuntimeError("R2 credentials or endpoint missing.")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


def upload_query_image_to_r2(img_bytes: bytes, filename: str) -> str:
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL is not configured.")
    ext = os.path.splitext(filename)[1].lower() or ".jpg"
    key = f"queries/{uuid.uuid4()}{ext}"
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    r2 = _r2_client()
    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=img_bytes, ContentType=content_type)
    return f"{PUBLIC_BASE_URL.rstrip('/')}/{key}"

# =============================================================
# Color helpers (for analyzing the query image)
# =============================================================

def to_hex(rgb) -> str:
    """Converts an RGB tuple or list to a hex string."""
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def get_dominant_color(image_bytes: bytes) -> Optional[np.ndarray]:
    """Estimate the dominant object colour of the query image."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img.thumbnail((150, 150)) # Resize for faster processing
        pixels = np.array(img, dtype=np.float32) / 255.0
        pixels = pixels.reshape(-1, 3)

        # Remove black and white pixels from consideration
        non_bw_mask = (pixels.mean(axis=1) > 0.05) & (pixels.mean(axis=1) < 0.95)
        pixels = pixels[non_bw_mask]

        if len(pixels) == 0:
            return None

        kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0).fit(pixels)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_color_float = kmeans.cluster_centers_[unique[counts.argmax()]]

        return (dominant_color_float * 255).astype(int)
    except Exception:
        return None

# =============================================================
# Marqo search (HTTP API)
# =============================================================

def marqo_search(q: str, limit: int = 200, filter_string: Optional[str] = None, attrs: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Performs a tensor search on Marqo, now with pre-filtering."""
    searchable_attributes = sanitize_attrs(attrs or [IMAGE_FIELD, TITLE_FIELD, SEARCH_BLOB_FIELD])

    payload: Dict[str, Any] = {
        "limit": limit,
        "q": q,
        "searchMethod": "TENSOR",
        "searchableAttributes": searchable_attributes,
        "attributesToRetrieve": [
            "_id", TITLE_FIELD, IMAGE_FIELD, ALT_IMAGE_FIELD, DOM_COLOR_FIELD,
            SKU_FIELD, DESCRIPTION_FIELD, SPEC_TEXT_FIELD,
            SEARCH_BLOB_FIELD, CLICK_URL_FIELD
        ]
    }

    if filter_string:
        payload["filter"] = filter_string

    url = f"{MARQO_URL}/indexes/{INDEX_NAME}/search"
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API search error: {e}")
        if getattr(e, 'response', None) is not None and e.response.text:
            with st.expander("📄 Server Response"):
                st.code(e.response.text, language='json')
        return None


def fuse_hits(img_hits: List[Dict[str, Any]], txt_hits: List[Dict[str, Any]], alpha: float = 0.7) -> List[Dict[str, Any]]:
    """Weighted late-fusion of two hit lists by _id."""
    def to_map(hits):
        return {h.get('_id'): float(h.get('_score', 0.0)) for h in hits}
    imap, tmap = to_map(img_hits), to_map(txt_hits)
    ids = set(imap) | set(tmap)
    fused = []
    for _id in ids:
        s = alpha * imap.get(_id, 0.0) + (1 - alpha) * tmap.get(_id, 0.0)
        base = next((h for h in img_hits if h.get('_id') == _id), None) or next((h for h in txt_hits if h.get('_id') == _id), None)
        if not base:
            continue
        h = dict(base)
        h['_fused_score'] = s
        fused.append(h)
    fused.sort(key=lambda x: x.get('_fused_score', 0.0), reverse=True)
    return fused

# =============================================================
# UI
# =============================================================

st.set_page_config(page_title="Furniture Search", layout="wide")
st.title("🛋️ Furniture and Interior Element Search")
st.caption("The system primarily searches for visually similar products. Text provides an additional signal.")

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 0

# Sidebar controls
st.sidebar.header("Search Settings")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="uploader"
)
search_query = st.sidebar.text_input("🔍 Search by text:", "")

use_color_filter = st.sidebar.checkbox("Enable color filtering", value=True)
color_threshold = st.sidebar.slider("Color similarity threshold", 0, 150, 50, 10)

# Boosting weights
visual_weight = 0.85
text_weight = 0.35 if search_query.strip() else 0.0

final_hits = []
active_filters = []

# --- MODIFIED: Filter Building Logic ---

# 1. Build Color Filter
color_filter_string = None
if uploaded_file and use_color_filter:
    img_bytes = uploaded_file.getvalue()
    query_rgb = get_dominant_color(img_bytes)
    if query_rgb is not None:
        r, g, b = query_rgb
        r_min, r_max = max(0, r - color_threshold), min(255, r + color_threshold)
        g_min, g_max = max(0, g - color_threshold), min(255, g + color_threshold)
        b_min, b_max = max(0, b - color_threshold), min(255, b + color_threshold)
        color_filter_string = (
            f"(color_r:[{r_min} TO {r_max}] AND "
            f"color_g:[{g_min} TO {g_max}] AND "
            f"color_b:[{b_min} TO {b_max}])"
        )
        active_filters.append(color_filter_string)

# 2. Build Text Filter
if search_query.strip():
    # This filter ensures all search words are present in the 'name' field
    search_words = search_query.strip().split()
    text_filters_for_name = [f'{TITLE_FIELD}:*"{word}"*' for word in search_words] # Use quotes for phrase matching
    text_filter_string = " AND ".join(text_filters_for_name)
    active_filters.append(f"({text_filter_string})")

# 3. Combine all active filters
final_filter_string = " AND ".join(active_filters) if active_filters else None


# --- Main search logic ---
if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", width=180)
    if use_color_filter and color_filter_string:
         hex_color = to_hex(query_rgb)
         st.sidebar.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-family: 'Source Sans Pro', sans-serif; color: #262730;">
                <span style="font-size: 0.9rem;">Filtering by dominant color:</span>
                <div style="width: 25px; height: 25px; background-color: {hex_color}; border: 1px solid #ccc; border-radius: 4px;"></div>
            </div>
            """, unsafe_allow_html=True)
            
    try:
        query_url = upload_query_image_to_r2(img_bytes, uploaded_file.name)
    except Exception as e:
        st.error(f"Failed to upload to R2: {e}")
        st.stop()

    with st.spinner("Searching..."):
        # Image-based search (semantic)
        vis_res = marqo_search(query_url, limit=200, attrs=[IMAGE_FIELD], filter_string=final_filter_string)
        sem_res = marqo_search(query_url, limit=200, attrs=[TITLE_FIELD, SEARCH_BLOB_FIELD], filter_string=final_filter_string)
        vis_hits = vis_res.get("hits", []) if vis_res else []
        sem_hits = sem_res.get("hits", []) if sem_res else []
        fused_img = fuse_hits(vis_hits, sem_hits, alpha=visual_weight)

        # Text-based search (semantic + lexical filter)
        if search_query.strip():
            txt_res = marqo_search(search_query.strip(), limit=200, attrs=[TITLE_FIELD, SEARCH_BLOB_FIELD], filter_string=final_filter_string)
            txt_hits = txt_res.get("hits", []) if txt_res else []
            final_hits = fuse_hits(fused_img, txt_hits, alpha=1.0 - text_weight) if text_weight > 0 else fused_img
        else:
            final_hits = fused_img

elif search_query.strip():
    with st.spinner("Searching by text..."):
        txt_res = marqo_search(search_query.strip(), limit=200, attrs=[TITLE_FIELD, SEARCH_BLOB_FIELD], filter_string=final_filter_string)
        final_hits = txt_res.get("hits", []) if txt_res else []
else:
    st.info("Please upload an image or enter a search query.")


# =============================================================
# Render results
# =============================================================

if final_hits:
    st.subheader(f"Found {len(final_hits)} results")
    
    # Pagination
    page_size = 9
    total_pages = (len(final_hits) - 1) // page_size + 1 if final_hits else 0
    current_page = st.session_state.page

    if current_page >= total_pages:
        current_page = 0
        st.session_state.page = 0

    col_prev, col_pg, col_next = st.columns([2, 8, 2])
    if col_prev.button("⬅ Previous", disabled=(current_page == 0)):
        st.session_state.page -= 1
        st.rerun()
    
    col_pg.markdown(f"<div style='text-align:center;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    
    if col_next.button("Next ➡", disabled=(current_page >= total_pages - 1)):
        st.session_state.page += 1
        st.rerun()

    start = current_page * page_size
    end = start + page_size
    page_hits = final_hits[start:end]

    cols = st.columns(3)
    for i, h in enumerate(page_hits):
        with cols[i % 3]:
            img_url = h.get(IMAGE_FIELD) or h.get(ALT_IMAGE_FIELD)
            title = h.get(TITLE_FIELD, 'No title')
            _id = h.get('_id', 'N/A')
            score = h.get('_fused_score', h.get('_score', None))
            click_url = h.get(CLICK_URL_FIELD)

            if img_url:
                st.image(img_url, use_container_width=True)
            st.write(f"**{title}**")
            st.caption(f"ID: {_id}")
            if isinstance(score, (int, float)):
                st.caption(f"Similarity: {score:.3f}")
            if isinstance(click_url, str) and click_url:
                st.markdown(f"[🔗 Open Product]({click_url})")
            
            dom_color_hex = h.get(DOM_COLOR_FIELD)
            if dom_color_hex:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 20px; height: 20px; background-color: {dom_color_hex}; border: 1px solid #ccc; border-radius: 4px;"></div>
                    <span>{dom_color_hex}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('---')

elif uploaded_file or search_query:
    st.warning("No results found. Try adjusting your query or the filter settings.")

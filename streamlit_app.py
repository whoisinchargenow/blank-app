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
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

MARQO_URL: str = (_cfg("MARQO_URL", "https://marqo.logicafutura.com") or "").rstrip("/")
INDEX_NAME: str = _cfg("INDEX_NAME", "furniture-index") or "furniture-index"

TITLE_FIELD: str = _cfg("TITLE_FIELD", "name") or "name"
IMAGE_FIELD: str = _cfg("IMAGE_FIELD", "image") or "image"
ALT_IMAGE_FIELD: str = _cfg("ALT_IMAGE_FIELD", "image_url") or "image_url"
SEARCH_BLOB_FIELD: str = _cfg("SEARCH_BLOB_FIELD", "search_blob") or "search_blob"
DOM_COLOR_FIELD: str = _cfg("DOMINANT_COLOR_FIELD", "dominant_color") or "dominant_color"
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


# =============================================================
# Helper Functions
# =============================================================

def to_hex(rgb) -> str:
    """Converts an RGB tuple or list to a hex string."""
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def get_dominant_color(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img.thumbnail((150, 150))
        pixels = np.array(img, dtype=np.float32) / 255.0
        pixels = pixels.reshape(-1, 3)
        non_bw_mask = (pixels.mean(axis=1) > 0.05) & (pixels.mean(axis=1) < 0.95)
        pixels = pixels[non_bw_mask]
        if len(pixels) == 0: return None
        kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0).fit(pixels)
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_color_float = kmeans.cluster_centers_[unique[counts.argmax()]]
        return (dominant_color_float * 255).astype(int)
    except Exception:
        return None

def marqo_search(q: str, limit: int = 200, filter_string: Optional[str] = None, attrs: Optional[List[str]] = None, method: str = "TENSOR") -> Optional[Dict[str, Any]]:
    """Performs a search on Marqo, allowing method to be 'TENSOR' or 'LEXICAL'."""
    payload: Dict[str, Any] = {
        "limit": limit, "q": q, "searchMethod": method,
        "searchableAttributes": attrs or [IMAGE_FIELD, TITLE_FIELD, SEARCH_BLOB_FIELD],
        "attributesToRetrieve": ["_id", TITLE_FIELD, IMAGE_FIELD, ALT_IMAGE_FIELD, DOM_COLOR_FIELD, SKU_FIELD, CLICK_URL_FIELD, "_score"]
    }
    if filter_string: payload["filter"] = filter_string
    url = f"{MARQO_URL}/indexes/{INDEX_NAME}/search"
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API search error: {e.response.text if e.response else e}")
        return None

def fuse_hits(img_hits: List[Dict[str, Any]], txt_hits: List[Dict[str, Any]], alpha: float = 0.7) -> List[Dict[str, Any]]:
    def to_map(hits):
        return {h.get('_id'): float(h.get('_score', 0.0)) for h in hits}
    imap, tmap = to_map(img_hits), to_map(txt_hits)
    ids = set(imap) | set(tmap)
    fused = []
    for _id in ids:
        s = alpha * imap.get(_id, 0.0) + (1 - alpha) * tmap.get(_id, 0.0)
        base = next((h for h in img_hits if h.get('_id') == _id), None) or next((h for h in txt_hits if h.get('_id') == _id), None)
        if not base: continue
        h = dict(base); h['_fused_score'] = s
        fused.append(h)
    fused.sort(key=lambda x: x.get('_fused_score', 0.0), reverse=True)
    return fused

def upload_query_image_to_r2(img_bytes: bytes, filename: str) -> str:
    if not PUBLIC_BASE_URL: raise RuntimeError("PUBLIC_BASE_URL is not configured.")
    r2 = boto3.client("s3", endpoint_url=R2_ENDPOINT_URL, aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY)
    ext = os.path.splitext(filename)[1].lower() or ".jpg"
    key = f"queries/{uuid.uuid4()}{ext}"
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=img_bytes, ContentType=content_type)
    return f"{PUBLIC_BASE_URL.rstrip('/')}/{key}"

def render_pagination(total_pages: int, current_page_zerobased: int):
    """Renders the advanced pagination component."""
    if total_pages <= 1: return
    max_pages_to_show = 10
    nav = st.container()
    cols = nav.columns([1, 1, *(2 for _ in range(max_pages_to_show)), 1, 1])
    def set_page(page_num): st.session_state.page = page_num
    
    is_first_page = current_page_zerobased == 0
    cols[0].button("«", on_click=set_page, args=[0], disabled=is_first_page, use_container_width=True)
    cols[1].button("‹", on_click=set_page, args=[current_page_zerobased - 1], disabled=is_first_page, use_container_width=True)

    half_window = max_pages_to_show // 2
    start_page = max(0, current_page_zerobased - half_window)
    end_page = min(total_pages, start_page + max_pages_to_show)
    if end_page - start_page < max_pages_to_show: start_page = max(0, end_page - max_pages_to_show)
    
    for i, page_num in enumerate(range(start_page, end_page)):
        is_current = page_num == current_page_zerobased
        cols[i+2].button(f"{page_num + 1}", on_click=set_page, args=[page_num], disabled=is_current, type="primary" if is_current else "secondary", use_container_width=True)

    is_last_page = current_page_zerobased == total_pages - 1
    cols[-2].button("›", on_click=set_page, args=[current_page_zerobased + 1], disabled=is_last_page, use_container_width=True)
    cols[-1].button("»", on_click=set_page, args=[total_pages - 1], disabled=is_last_page, use_container_width=True)

# =============================================================
# UI and Application Flow
# =============================================================

st.set_page_config(page_title="Furniture Search", layout="wide")
st.title("🛋️ Furniture and Interior Element Search")

if 'page' not in st.session_state: st.session_state.page = 0
st.sidebar.header("Search Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
search_query = st.sidebar.text_input("🔍 Search by text")

final_hits = []

# --- Main Logic Branch: Image Search (+ optional text) ---
if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", width=180)
    st.sidebar.markdown("---")
    st.sidebar.write("**Image Options**")
    use_color_filter = st.sidebar.checkbox("Enable color filtering", value=True)
    color_threshold = st.sidebar.slider("Color similarity threshold", 0, 150, 50, 10)
    
    img_bytes = uploaded_file.getvalue()
    
    color_filter_string = None
    if use_color_filter:
        query_rgb = get_dominant_color(img_bytes)
        if query_rgb is not None:
            r, g, b = query_rgb
            r_min, r_max = max(0, r - color_threshold), min(255, r + color_threshold)
            g_min, g_max = max(0, g - color_threshold), min(255, g + color_threshold)
            b_min, b_max = max(0, b - color_threshold), min(255, b + color_threshold)
            color_filter_string = f"(color_r:[{r_min} TO {r_max}] AND color_g:[{g_min} TO {g_max}] AND color_b:[{b_min} TO {b_max}])"
            hex_color = to_hex(query_rgb)
            st.sidebar.markdown(f"""
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-family: 'Source Sans Pro', sans-serif; color: #262730;">
                    <span style="font-size: 0.9rem;">Filtering by dominant color:</span>
                    <div style="width: 25px; height: 25px; background-color: {hex_color}; border: 1px solid #ccc; border-radius: 4px;"></div>
                </div>
                """, unsafe_allow_html=True)

    with st.spinner("Searching..."):
        try:
            query_url = upload_query_image_to_r2(img_bytes, uploaded_file.name)
        except Exception as e:
            st.error(f"Failed to upload image: {e}"); st.stop()
        
        vis_res = marqo_search(query_url, attrs=[IMAGE_FIELD], filter_string=color_filter_string)
        sem_res = marqo_search(query_url, attrs=[TITLE_FIELD, SEARCH_BLOB_FIELD], filter_string=color_filter_string)
        
        # --- FIX: Safely get hits, providing an empty list if the search failed ---
        vis_hits = vis_res.get("hits", []) if vis_res else []
        sem_hits = sem_res.get("hits", []) if sem_res else []
        image_search_results = fuse_hits(vis_hits, sem_hits)

        if search_query.strip():
            txt_res = marqo_search(q=search_query.strip(), limit=1000, attrs=[TITLE_FIELD], method="LEXICAL")
            # Safely get hits from the text search
            text_search_hits = txt_res.get("hits", []) if txt_res else []
            
            image_result_ids = {hit['_id'] for hit in image_search_results}
            text_result_ids = {hit['_id'] for hit in text_search_hits}
            common_ids = image_result_ids.intersection(text_result_ids)
            final_hits = [hit for hit in image_search_results if hit['_id'] in common_ids]
        else:
            final_hits = image_search_results

# --- Main Logic Branch: Text-Only Search ---
elif search_query.strip():
    with st.spinner("Searching by text..."):
        txt_res = marqo_search(q=search_query.strip(), limit=1000, attrs=[TITLE_FIELD], method="LEXICAL")
        # Safely get hits from the text search
        final_hits = txt_res.get("hits", []) if txt_res else []

# --- Initial State ---
else:
    st.info("Please upload an image or enter a search query to begin.")

# =============================================================
# Render Results
# =============================================================

if final_hits:
    if "last_hit_count" not in st.session_state or st.session_state.last_hit_count != len(final_hits):
        st.session_state.page = 0
    st.session_state.last_hit_count = len(final_hits)
    
    st.subheader(f"Found {len(final_hits)} results")
    page_size = 9
    total_pages = (len(final_hits) - 1) // page_size + 1
    current_page = st.session_state.page
    
    render_pagination(total_pages, current_page)

    start_idx = current_page * page_size
    end_idx = start_idx + page_size
    page_hits = final_hits[start_idx:end_idx]

    cols = st.columns(3)
    for i, h in enumerate(page_hits):
        with cols[i % 3]:
            img_url = h.get(IMAGE_FIELD) or h.get(ALT_IMAGE_FIELD)
            title = h.get(TITLE_FIELD, 'No title')
            click_url = h.get(CLICK_URL_FIELD)
            dom_color_hex = h.get(DOM_COLOR_FIELD)
            
            if img_url: st.image(img_url, use_container_width=True)
            st.write(f"**{title}**")
            if click_url: st.markdown(f"[🔗 Open Product]({click_url})")
            if dom_color_hex:
                st.markdown(f"""<div style="display: flex; align-items: center; gap: 8px;">
                                <div style="width: 20px; height: 20px; background-color: {dom_color_hex}; border: 1px solid #ccc; border-radius: 4px;"></div>
                                <span>{dom_color_hex}</span></div>""", unsafe_allow_html=True)
            st.markdown('---')

elif uploaded_file or search_query:
    st.warning("No results found. Try adjusting your query or the filter settings.")

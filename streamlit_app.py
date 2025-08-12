import os
import io
import json
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

"""
STREAMLIT versija (online) – pilnai atkuria originalią logiką:
- ĮKĖLIMAS: galite įkelti paveikslėlį; app išsaugo failą į viešai pasiekiamą vietą
  ir perduoda jo URL į Marqo paieškai (q = public_image_url).
- TEKSTO paieška: galite ieškoti pagal pavadinimą / tipą / SKU.
- Spalvos filtras (KMeans dominuojanti spalva) ir rūšiavimas pagal balą – kaip originale.
- Filtravimas pagal objekto tipą (remiantis pirmu hitu) – kaip originale.
- Paveikslėliai rodomi iš VENDOR URL laukų, kurie jau yra Marqo dokumentuose.

BŪTINA ONLINE ĮKĖLIMUI: turėkite viešą statinį URL, kuris atitinka vietinį katalogą,
kuriame app išsaugo įkeltą failą (žiūrėkite secrets skiltį žemiau).
"""

# -----------------------------
# Konfigūracija (iš secrets arba ENV)
# -----------------------------

def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(key, default)

MARQO_URL: str = (_cfg("MARQO_URL", "https://marqo.logicafutura.com") or "").rstrip("/")
INDEX_NAME: str = _cfg("INDEX_NAME", "furniture-index") or "furniture-index"

# Laukų pavadinimai (pagal jūsų indeksavimo skriptą)
TITLE_FIELD: str = _cfg("TITLE_FIELD", "title") or "title"
IMAGE_FIELD: str = _cfg("IMAGE_FIELD", "image") or "image"
ALT_IMAGE_FIELD: str = _cfg("ALT_IMAGE_FIELD", "image_vendor_url") or "image_vendor_url"
DOM_COLOR_FIELD: str = _cfg("DOMINANT_COLOR_FIELD", "dominant_color") or "dominant_color"
SKU_FIELD: str = _cfg("SKU_FIELD", "sku") or "sku"

# Failų saugojimas įkėlimui → viešas URL
IMAGE_UPLOAD_DIR: str = _cfg("IMAGE_UPLOAD_DIR", "./public") or "./public"
PUBLIC_IMAGE_URL_PREFIX: str = _cfg("PUBLIC_IMAGE_URL_PREFIX", "http://host.docker.internal:8080/") or "http://host.docker.internal:8080/"
QUERY_FILENAME: str = _cfg("QUERY_FILENAME", "uploaded_query.jpg") or "uploaded_query.jpg"

# Cloudflare Access (jei Marqo saugomas per Access Service Token)
CF_ID = _cfg("CF_ACCESS_CLIENT_ID")
CF_SECRET = _cfg("CF_ACCESS_CLIENT_SECRET")

# HTTP antraštės
HEADERS: Dict[str, str] = {"Content-Type": "application/json"}
if CF_ID and CF_SECRET:
    HEADERS["CF-Access-Client-Id"] = CF_ID
    HEADERS["CF-Access-Client-Secret"] = CF_SECRET

# Žinomi tipai (kaip originale)
KNOWN_TYPES = ['table', 'lamp', 'rack', 'chair', 'sofa', 'bench', 'bed', 'cabinet', 'desk']

# -----------------------------
# Pagalbinės funkcijos
# -----------------------------

def get_dominant_color(image_bytes: bytes) -> np.ndarray:
    """Apskaičiuoja dominuojančią spalvą, ignoruojant beveik baltą/juodą foną."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((50, 50))
        data = np.array(img).reshape(-1, 3)

        def is_background(pixel):
            return np.all(pixel > 240) or np.all(pixel < 15)

        filtered = np.array([px for px in data if not is_background(px)])
        if filtered.size == 0:
            filtered = data

        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(filtered)
        return kmeans.cluster_centers_[0].astype(int)
    except Exception as e:
        st.warning(f"Nepavyko nustatyti spalvos: {e}")
        return np.array([0, 0, 0])


def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = (hex_color or "").lstrip('#')
    if len(hex_color) != 6:
        return np.array([0, 0, 0])
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])


def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.linalg.norm(c1 - c2))


def detect_object_type(title: str) -> str:
    t = (title or "").lower()
    for obj in KNOWN_TYPES:
        if obj in t:
            return obj
    return "other"


def marqo_search(query: str) -> Optional[Dict[str, Any]]:
    payload = {
        "limit": 1000,
        "q": query,
        "searchMethod": "TENSOR",
        "searchableAttributes": [IMAGE_FIELD, TITLE_FIELD],
        "attributesToRetrieve": ["_id", TITLE_FIELD, IMAGE_FIELD, ALT_IMAGE_FIELD, DOM_COLOR_FIELD, SKU_FIELD]
    }
    try:
        url = f"{MARQO_URL}/indexes/{INDEX_NAME}/search"
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        server_response = e.response.text if getattr(e, 'response', None) is not None else "Serveris neatsako."
        st.error(f"API paieškos klaida: {e}")
        st.error(f"Serverio atsakymas: {server_response}")
        return None

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Vaizdų paieška", layout="wide")
st.title("🖼️ Panašių vaizdų paieška (Marqo)")
st.markdown("Įkelkite nuotrauką ARBA įveskite raktažodį ir raskite panašius produktus iš tiekėjų svetainių.")

# Būsena
if 'last_upload_hash' not in st.session_state:
    st.session_state.last_upload_hash = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'query_color' not in st.session_state:
    st.session_state.query_color = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'detected_object_type' not in st.session_state:
    st.session_state.detected_object_type = None

# Valdikliai
uploaded_file = st.file_uploader("Pasirinkite paveikslėlį (JPG/PNG)", type=["jpg", "jpeg", "png"], key="uploader")
search_query = st.text_input("🔍 Ieškoti pagal pavadinimą / tipą / SKU (nebūtina):", "")

# Pagrindinė logika
if uploaded_file:
    st.image(uploaded_file, caption="Įkeltas paveikslėlis", width=200)
    color_threshold = st.slider(
        "Spalvos panašumo riba", 0, 200, 50, 10,
        format="%d", help="Kairėje – labiau panaši, dešinėje – mažiau panaši"
    )
    use_color_filter = st.checkbox("Įjungti spalvos filtravimą", value=True)

    img_bytes = uploaded_file.getvalue()
    current_hash = hash(img_bytes)

    if current_hash != st.session_state.last_upload_hash or color_threshold != st.session_state.get('last_color_threshold', -1):
        st.session_state.last_upload_hash = current_hash
        st.session_state.last_color_threshold = color_threshold
        st.session_state.current_page = 0

        # 1) Išsaugome failą į vietinį katalogą
        os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)
        query_path = os.path.join(IMAGE_UPLOAD_DIR, QUERY_FILENAME)
        with open(query_path, "wb") as f:
            f.write(img_bytes)

        # 2) Viešas URL, kurį Marqo galės pasiekti
        base = PUBLIC_IMAGE_URL_PREFIX
        if not base.endswith('/'):
            base += '/'
        query_url = base + QUERY_FILENAME

        # 3) Dominuojanti spalva
        query_color = get_dominant_color(img_bytes)
        st.session_state.query_color = query_color

        # 4) Paieška pagal įkelto vaizdo URL
        with st.spinner("Ieškoma panašių vaizdų..."):
            results = marqo_search(query_url)
            if results and results.get("hits"):
                raw_hits = results["hits"]
                for hit in raw_hits:
                    title = hit.get(TITLE_FIELD, hit.get("title", ""))
                    hit["object_type"] = detect_object_type(title)

                st.session_state.detected_object_type = raw_hits[0]["object_type"]

                filtered_hits = raw_hits
                if use_color_filter:
                    filtered_hits = []
                    for hit in raw_hits:
                        hit_color_hex = hit.get(DOM_COLOR_FIELD, "#000000")
                        hit_rgb = hex_to_rgb(hit_color_hex)
                        dist = color_distance(query_color, hit_rgb)
                        if dist <= color_threshold + 5:
                            filtered_hits.append(hit)

                    # Rūšiuojame pagal modelio balą – (spalvos atstumas / 441)
                    filtered_hits.sort(
                        key=lambda h: h.get('_score', 0) - (color_distance(query_color, hex_to_rgb(h.get(DOM_COLOR_FIELD, "#000000"))) / 441.0),
                        reverse=True
                    )

                # Paliekame tik to paties objekto tipo kaip 1‑as hitas
                filtered_hits = [h for h in filtered_hits if h["object_type"] == st.session_state.detected_object_type]
                st.session_state.search_results = {"hits": filtered_hits}
            else:
                st.session_state.search_results = {"hits": []}

else:
    # Teksto paieška
    if search_query.strip():
        with st.spinner("Ieškoma pagal raktažodį..."):
            results = marqo_search(search_query)
            if results and results.get("hits"):
                for hit in results["hits"]:
                    title = hit.get(TITLE_FIELD, hit.get("title", ""))
                    hit["object_type"] = detect_object_type(title)
                st.session_state.search_results = {"hits": results["hits"]}
            else:
                st.session_state.search_results = {"hits": []}
    else:
        st.session_state.search_results = None
        st.session_state.last_upload_hash = None
        st.session_state.query_color = None
        st.session_state.current_page = 0

# Rezultatų atvaizdavimas
results = st.session_state.search_results
if results and results.get("hits"):
    hits = results["hits"]

    # Jei yra ir įkėlimas, ir tekstas – papildomas filtravimas pagal pavadinimą
    if uploaded_file and search_query.strip():
        keyword = search_query.lower()
        hits = [h for h in hits if keyword in (h.get(TITLE_FIELD, h.get("title", "")).lower())]

    page_size = 9
    total_pages = (len(hits) - 1) // page_size + 1 if hits else 0
    current_page = st.session_state.current_page

    if total_pages == 0:
        st.info("Rezultatų nerasta. Pabandykite įkelti kitą paveikslėlį arba įvesti kitą raktažodį.")
    else:
        st.subheader(f"✅ Rodymas: puslapis {current_page + 1} iš {total_pages}")

        start = current_page * page_size
        end = start + page_size
        page_hits = hits[start:end]

        cols = st.columns(3)
        for i, hit in enumerate(page_hits):
            with cols[i % 3]:
                img_url = hit.get(IMAGE_FIELD) or hit.get(ALT_IMAGE_FIELD) or hit.get("image")
                # Senas atvejis: jei kažkur buvo 'host.docker.internal'
                if isinstance(img_url, str):
                    img_url = img_url.replace("host.docker.internal", "localhost")

                title = hit.get(TITLE_FIELD, hit.get('title', 'Be pavadinimo'))
                _id = hit.get('_id', 'Nėra')
                score = hit.get('_score')

                if img_url:
                    st.image(img_url, use_container_width=True)
                st.write(f"**Pavadinimas:** {title}")
                st.write(f"**Produkto ID:** {_id}")
                if isinstance(score, (int, float)):
                    st.write(f"**Panašumo įvertinimas:** {score:.2f}")

        col1, col2, col3 = st.columns([1, 5, 1])
        with col1:
            if current_page > 0:
                if st.button("⬅ Ankstesnis"):
                    st.session_state.current_page -= 1
                    st.rerun()

        with col2:
            visible_pages = list(range(current_page, min(current_page + 8, total_pages)))
            if visible_pages:
                btn_cols = st.columns(len(visible_pages))
                for i, p in enumerate(visible_pages):
                    if btn_cols[i].button(str(p + 1)):
                        st.session_state.current_page = p
                        st.rerun()
            else:
                st.info("Rezultatų nerasta. Bandykite kitą puslapį.")

        with col3:
            if current_page < total_pages - 1:
                if st.button("Kitas ➡"):
                    st.session_state.current_page += 1
                    st.rerun()
else:
    st.info("Rezultatų nerasta. Pabandykite įkelti paveikslėlį arba įvesti raktažodį.")

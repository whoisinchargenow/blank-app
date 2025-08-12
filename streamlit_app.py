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
STREAMLIT versija originalios paieškos programos:
- Leidžia ĮKELTI nuotrauką ir ieškoti panašių vaizdų (pagal nuotrauką) – kaip originale.
- Leidžia ieškoti pagal tekstą (produkto pavadinimą / tipą / SKU) – kaip originale.
- Spalvos filtravimas su slankikliu ir rūšiavimas pagal panašumą + spalvos atstumą – kaip originale.
- Filtravimas pagal aptiktą objekto tipą (table/lamp/…): pirmo hito tipas → rodom tik to paties tipo – kaip originale.
- Veikia su Marqo per HTTP API. UI terminologija lietuvių kalba.

PASTABA dėl įkelto vaizdo URL:
Originale įkeltas failas buvo pasiekiamas per lokalią HTTP nuorodą (pvz., host.docker.internal:8080).
Kad tai veiktų ONLINE aplinkoje, sukonfigūruokite viešą katalogą, kuriame Streamlit gali išsaugoti failą,
o viešasis URL (PUBLIC_IMAGE_URL_PREFIX) turi rodyti į tą katalogą. Pvz., CDN ar statinis serveris už Cloudflare.
"""

# -----------------------------
# Konfigūracija
# -----------------------------

def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(key, default)

# Marqo ir indeksas
MARQO_URL: str = (_get_secret("MARQO_URL", "https://marqo.logicafutura.com") or "").rstrip("/")
INDEX_NAME: str = _get_secret("INDEX_NAME", "furniture-index") or "furniture-index"

# Laukų pavadinimai (pritaikykite prie savo schemos)
TITLE_FIELD: str = _get_secret("TITLE_FIELD", "name") or "name"          # produkto pavadinimas
IMAGE_FIELD: str = _get_secret("IMAGE_FIELD", "image_url") or "image_url"  # vaizdo URL laukas
DOM_COLOR_FIELD: str = _get_secret("DOMINANT_COLOR_FIELD", "dominant_color") or "dominant_color"
SKU_FIELD: str = _get_secret("SKU_FIELD", "sku") or "sku"

# Kur įrašyti įkeltą vaizdą ir koks bus jo viešas URL
# BŪTINA: PUBLIC_IMAGE_URL_PREFIX turi baigtis "/" ir būti pasiekiamas iš Marqo
IMAGE_FOLDER: str = _get_secret("IMAGE_UPLOAD_DIR", "./public") or "./public"
PUBLIC_IMAGE_URL_PREFIX: str = _get_secret("PUBLIC_IMAGE_URL_PREFIX", "http://host.docker.internal:8080/") or "http://host.docker.internal:8080/"
QUERY_FILENAME: str = _get_secret("QUERY_FILENAME", "uploaded_query.jpg") or "uploaded_query.jpg"

# Cloudflare Access (jei Marqo saugomas per Access Service Token)
CF_ID = _get_secret("CF_ACCESS_CLIENT_ID")
CF_SECRET = _get_secret("CF_ACCESS_CLIENT_SECRET")

# HTTP antraštės
HEADERS: Dict[str, str] = {"Content-Type": "application/json"}
if CF_ID and CF_SECRET:
    HEADERS["CF-Access-Client-Id"] = CF_ID
    HEADERS["CF-Access-Client-Secret"] = CF_SECRET

# Žinomi tipai (kaip originale)
KNOWN_TYPES = ['table', 'lamp', 'rack', 'chair', 'sofa', 'bench', 'bed', 'cabinet', 'desk']

# -----------------------------
# Pagalbinės funkcijos (tokios pačios idėjos kaip originale)
# -----------------------------

def get_dominant_color(image_bytes: bytes) -> np.ndarray:
    """Apskaičiuoja dominuojančią spalvą; ignoruoja beveik baltą/juodą foną."""
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


def search_similar_images(query: str) -> Optional[Dict[str, Any]]:
    payload = {
        "limit": 1000,
        "q": query,
        "searchMethod": "TENSOR",
        "searchableAttributes": [IMAGE_FIELD, TITLE_FIELD],
        "attributesToRetrieve": ["_id", TITLE_FIELD, IMAGE_FIELD, DOM_COLOR_FIELD, SKU_FIELD]
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
st.title("🖼️ Panašių vaizdų paieška")
st.markdown("Įkelkite nuotrauką ARBA įveskite raktažodį, kad surastumėte vizualiai panašius produktus.")

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
uploaded_file = st.file_uploader("Pasirinkite paveikslėlį", type=["jpg", "jpeg", "png"], key="uploader")
search_query = st.text_input("🔍 Ieškoti pagal produkto pavadinimą ar tipą (nebūtina):", "")

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

        # 1) Išsaugome failą viešai pasiekiamame kataloge
        os.makedirs(IMAGE_FOLDER, exist_ok=True)
        query_path = os.path.join(IMAGE_FOLDER, QUERY_FILENAME)
        with open(query_path, "wb") as f:
            f.write(img_bytes)

        # 2) Konstruojame viešą URL, kurį Marqo galės pasiekti
        if not PUBLIC_IMAGE_URL_PREFIX.endswith('/'):
            PUBLIC_IMAGE_URL_PREFIX += '/'
        query_url = PUBLIC_IMAGE_URL_PREFIX + QUERY_FILENAME

        # 3) Apskaičiuojame įkelto vaizdo dominuojančią spalvą
        query_color = get_dominant_color(img_bytes)
        st.session_state.query_color = query_color

        # 4) Ieškome panašių vaizdų pagal įkelto vaizdo URL
        with st.spinner("Ieškoma panašių vaizdų..."):
            results = search_similar_images(query_url)
            if results and results.get("hits"):
                raw_hits = results["hits"]
                for hit in raw_hits:
                    title = hit.get(TITLE_FIELD, hit.get("title", ""))
                    hit["object_type"] = detect_object_type(title)

                # Nustatome bazinį tipą pagal pirmą hitą
                st.session_state.detected_object_type = raw_hits[0]["object_type"]

                # Spalvos filtras (jei įjungtas)
                filtered_hits = raw_hits
                if use_color_filter:
                    filtered_hits = []
                    for hit in raw_hits:
                        hit_color_hex = hit.get(DOM_COLOR_FIELD, "#000000")
                        hit_rgb = hex_to_rgb(hit_color_hex)
                        dist = color_distance(query_color, hit_rgb)
                        if dist <= color_threshold + 5:
                            filtered_hits.append(hit)

                    # Rūšiuojame: modelio balas minus normalizuotas spalvos atstumas (441 ≈ max RGB dist)
                    filtered_hits.sort(
                        key=lambda h: h.get('_score', 0) - (color_distance(query_color, hex_to_rgb(h.get(DOM_COLOR_FIELD, "#000000"))) / 441.0),
                        reverse=True
                    )

                # Filtruojame pagal aptiktą objekto tipą (kaip originale)
                filtered_hits = [h for h in filtered_hits if h["object_type"] == st.session_state.detected_object_type]
                st.session_state.search_results = {"hits": filtered_hits}
            else:
                st.session_state.search_results = {"hits": []}

else:
    if search_query.strip():
        with st.spinner("Ieškoma pagal raktažodį..."):
            results = search_similar_images(search_query)
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

    # Jei įkeltas failas + tekstas: papildomai filtruojame pagal tekstą pavadinime
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
                image_url = hit.get(IMAGE_FIELD, hit.get("image", ""))
                # Istorinis atvejis (jei indeksuota lokaliai):
                image_url = image_url.replace("host.docker.internal", "localhost") if isinstance(image_url, str) else image_url

                title = hit.get(TITLE_FIELD, hit.get('title', 'Be pavadinimo'))
                _id = hit.get('_id', 'Nėra')
                score = hit.get('_score')

                if image_url:
                    st.image(image_url, use_container_width=True)
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

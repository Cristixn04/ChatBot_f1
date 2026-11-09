"""
PADDOCK GLOBAL - FORMULA 1 DESIGN SYSTEM
Aplicación oficial de percepción y telemetría de aficionados para pilotos de Fórmula 1.
Adaptación pedagógica del Taller de Chatbots con LangChain y Streamlit.
"""

import json
import os
import textwrap
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from almacenamiento import (
    cargar_pilotos,
    cargar_opiniones,
    crear_registro,
    guardar_opinion,
    consultar_opiniones,
    ErrorAlmacenamiento
)
from analisis import (
    obtener_modelo_llm,
    analizar_opinion,
    AnalisisOpinion
)

# Carga de variables de entorno
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent
RUTA_PILOTOS = BASE_DIR / "datos" / "pilotos.json"
RUTA_OPINIONES = BASE_DIR / "datos" / "opiniones.json"

st.set_page_config(
    page_title="Formula 1 | Paddock Global",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_html(html_str: str):
    """
    Renderiza HTML puro en Streamlit utilizando st.html,
    garantizando que nunca se interpreten bloques de código sangrado (CommonMark).
    """
    clean_html = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.html(clean_html)


# --- RECURSOS VECTORIALES (SVG) ---
SVG_F1_LOGO = """<svg xmlns="http://www.w3.org/2000/svg" width="92" height="23" viewBox="0 0 120 30" fill="none"><path d="M101.086812,30 L101.711812,30 L101.711812,27.106875 L101.722437,27.106875 L102.761812,30 L103.302437,30 L104.341812,27.106875 L104.352437,27.106875 L104.352437,30 L104.977437,30 L104.977437,26.25125 L104.063687,26.25125 L103.055562,29.18625 L103.044937,29.18625 L102.011187,26.25125 L101.086812,26.25125 L101.086812,30 Z M97.6274375,26.818125 L98.8136875,26.818125 L98.8136875,30 L99.4699375,30 L99.4699375,26.818125 L100.661812,26.818125 L100.661812,26.25125 L97.6274375,26.25125 L97.6274375,26.818125 Z M89.9999375,30 L119.999937,0 L101.943687,0 L71.9443125,30 L89.9999375,30 Z M85.6986875,13.065 L49.3818125,13.065 C38.3136875,13.065 36.3768125,13.651875 31.6361875,18.3925 C27.2024375,22.82625 20.0005625,30 20.0005625,30 L35.7324375,30 L39.4855625,26.246875 C41.9530625,23.779375 43.2255625,23.52375 48.4068125,23.52375 L75.2405625,23.52375 L85.6986875,13.065 Z M31.1518125,16.253125 C27.8774375,19.3425 20.7530625,26.263125 16.9130625,30 L-6.25e-05,30 C-6.25e-05,30 13.5524375,16.486875 21.0849375,9.0725 C28.8455625,1.685 32.7143125,0 46.9486875,0 L98.7643125,0 L87.5449375,11.21875 L48.0011875,11.21875 C37.9993125,11.21875 35.7518125,11.911875 31.1518125,16.253125 Z" fill="#E10600"/></svg>"""
SVG_FIA_LOGO = """<svg width="34" height="20" viewBox="0 0 50 30" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="50" height="30" rx="3" fill="#1C212D"/><text x="50%" y="65%" text-anchor="middle" fill="#E5E7EB" font-family="'Titillium Web', sans-serif" font-weight="900" font-size="12">FIA</text></svg>"""


# --- ESTILOS CSS RESPONSIVOS Y SISTEMA FORMULA 1 ---
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,300;0,400;0,600;0,700;0,900;1,700;1,900&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --f1-red: #E10600;
    --f1-dark: #0E1015;
    --f1-card-bg: #141720;
    --f1-border: rgba(255, 255, 255, 0.12);
    --f1-font: 'Titillium Web', -apple-system, BlinkMacSystemFont, sans-serif;
}

header[data-testid="stHeader"] {
    display: none !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1360px !important;
    width: 100% !important;
}

body, [class*="css"] {
    font-family: var(--f1-font);
    background-color: var(--f1-dark);
}

/* Header Banner */
.f1-header-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.f1-brand-group {
    display: flex;
    align-items: center;
    gap: 16px;
}
.f1-brand-title {
    font-size: 1.15rem;
    font-weight: 900;
    letter-spacing: 0.8px;
    color: #FFFFFF;
    text-transform: uppercase;
}
.f1-brand-sub {
    font-size: 0.72rem;
    color: #9CA3AF;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Double Red Racing Stripes */
.f1-double-stripes {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin: 8px 0 16px 0;
}
.f1-stripe {
    height: 2px;
    background: var(--f1-red);
    width: 100%;
}

/* F1 Styled Tabs */
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
    gap: 18px !important;
    padding: 0 4px !important;
    margin-bottom: 20px !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: var(--f1-font) !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    color: #9CA3AF !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    border-bottom: 3px solid rgba(225, 6, 0, 0.5) !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 3px solid var(--f1-red) !important;
}

/* Podium Grid */
.f1-podium-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 18px;
    margin-bottom: 24px;
}

.f1-podium-card {
    border-radius: 14px;
    padding: 20px;
    color: #FFFFFF;
    position: relative;
    overflow: hidden;
    min-height: 200px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.f1-podium-card.first {
    border: 2px solid rgba(255, 215, 0, 0.45);
}

.f1-podium-img {
    position: absolute;
    right: 4px;
    bottom: 0;
    max-height: 175px;
    width: auto;
    max-width: 46%;
    object-fit: contain;
    object-position: bottom right;
    pointer-events: none;
    z-index: 1;
}

/* Circular Flags & Avatars */
.circular-flag {
    width: 20px;
    height: 20px;
    min-width: 20px;
    border-radius: 50%;
    object-fit: cover;
    vertical-align: middle;
    display: inline-block;
    border: 1px solid rgba(255, 255, 255, 0.25);
    box-shadow: 0 1px 3px rgba(0,0,0,0.5);
    margin-right: 6px;
}

.circular-avatar {
    width: 26px;
    height: 26px;
    min-width: 26px;
    border-radius: 50%;
    object-fit: cover;
    object-position: top;
    vertical-align: middle;
    display: inline-block;
    border: 1.5px solid rgba(255, 255, 255, 0.25);
    margin-right: 8px;
}

/* Table Wrapper */
.f1-table-wrapper {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    border-radius: 8px;
    margin-top: 12px;
}

.f1-table {
    width: 100%;
    min-width: 700px;
    border-collapse: separate;
    border-spacing: 0 6px;
}

.f1-table th {
    font-size: 0.72rem;
    font-weight: 800;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 14px;
    text-align: left;
    border: none;
}

.f1-table-row {
    background: #141720;
    transition: background 0.15s ease;
}

.f1-table-row:hover {
    background: #1C212D;
}

.f1-table-row td {
    padding: 12px 14px;
    font-size: 0.9rem;
    color: #E5E7EB;
    border: none;
    vertical-align: middle;
}

/* Badges */
.badge-f1 {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.b-pos { background: rgba(0, 210, 106, 0.14); color: #00D26A; border: 1px solid rgba(0, 210, 106, 0.4); }
.b-neg { background: rgba(248, 49, 47, 0.14); color: #F8312F; border: 1px solid rgba(248, 49, 47, 0.4); }
.b-neu { background: rgba(156, 163, 175, 0.14); color: #D1D5DB; border: 1px solid rgba(156, 163, 175, 0.4); }
.b-mix { background: rgba(255, 184, 0, 0.14); color: #FFB800; border: 1px solid rgba(255, 184, 0, 0.4); }

/* Footer */
.f1-footer {
    margin-top: 40px;
    padding: 24px 0 16px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
""")


# --- HELPERS PARA IMÁGENES Y BANDERAS ---
def render_bandera(codigo_iso: str, tamano: int = 20) -> str:
    """Genera la imagen circular de la bandera nacional oficial con fallback seguro."""
    iso = (codigo_iso or "un").lower().strip()
    return f'<img src="https://flagcdn.com/w40/{iso}.png" class="circular-flag" style="width: {tamano}px; height: {tamano}px;" alt="{iso}" onerror="this.onerror=null; this.src=\'https://flagcdn.com/w40/un.png\';">'


def render_nombre_piloto(nombre_pila: str, apellido: str) -> str:
    """Formato oficial F1: Nombre ligero y Apellido en MAYÚSCULAS NEGRITA."""
    return f'<span style="font-weight: 300; color: #D1D5DB;">{nombre_pila}</span> <span style="font-weight: 900; color: #FFFFFF; text-transform: uppercase;">{apellido}</span>'


# --- RESOLUCIÓN AUTOMÁTICA DE CREDENCIALES (API INTEGRADA) ---
gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
openai_key = os.getenv("OPENAI_API_KEY", "").strip()

proveedor_activo = "Google Gemini" if gemini_key else "OpenAI"
api_key_activa = gemini_key if gemini_key else openai_key
modelo_activo = "gemini-2.5-flash" if proveedor_activo == "Google Gemini" else "gpt-4o-mini"


# --- CARGA DEL CATÁLOGO DE PILOTOS ---
try:
    catalogo_pilotos = cargar_pilotos(RUTA_PILOTOS)
except Exception as e:
    st.error(f"Fallo al cargar el catálogo de pilotos: {e}")
    st.stop()


# --- ESTADO EN SESIÓN ---
if "mensajes_f1" not in st.session_state:
    st.session_state.mensajes_f1 = []


# --- BARRA LATERAL (PIT WALL) ---
with st.sidebar:
    render_html(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
        <div>{SVG_F1_LOGO}</div>
        <div>{SVG_FIA_LOGO}</div>
    </div>
    """)

    if api_key_activa:
        render_html(f"""
        <div style="background: rgba(0, 210, 106, 0.08); border: 1px solid rgba(0, 210, 106, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #00D26A; box-shadow: 0 0 6px #00D26A; display: inline-block;"></span>
                <span style="font-size: 0.72rem; font-weight: 800; color: #00D26A; letter-spacing: 0.8px;">TELEMETRÍA EN LÍNEA</span>
            </div>
            <div style="font-size: 0.85rem; color: #FFFFFF; font-weight: 700; margin-top: 3px;">
                Motor: {modelo_activo}
            </div>
        </div>
        """)

    st.markdown("<div style='font-size: 0.78rem; font-weight: 800; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;'>Piloto Seleccionado</div>", unsafe_allow_html=True)

    opciones_dict = {
        f"{p['nombre_pila']} {p['apellido']} (#{p.get('numero', '--')}) - {p.get('codigo_pais', '')}": p
        for p in catalogo_pilotos
    }
    etiqueta_sel = st.selectbox(
        "Piloto:",
        list(opciones_dict.keys()),
        label_visibility="collapsed"
    )
    piloto_actual = opciones_dict[etiqueta_sel]

    # Mini ficha técnica con avatar circular del piloto
    bandera_sidebar = render_bandera(piloto_actual.get('codigo_iso', 'un'), 22)
    foto_sidebar = piloto_actual.get("foto_url", "")
    ap_sidebar = piloto_actual.get("apellido", "")
    foto_html = f'<img src="{foto_sidebar}" style="width: 50px; height: 50px; min-width: 50px; border-radius: 50%; object-fit: cover; object-position: top; border: 2px solid rgba(255,255,255,0.2); display: block;" onerror="this.style.display=\'none\'" alt="{ap_sidebar}">' if foto_sidebar else ""

    render_html(f"""
    <div style="background: #141720; border-left: 4px solid {piloto_actual.get('color_hex', '#E10600')}; border-radius: 8px; padding: 12px; margin-top: 8px; display: flex; align-items: center; justify-content: space-between; gap: 12px;">
        <div style="flex: 1; min-width: 0;">
            <div style="display: flex; align-items: center; gap: 6px;">
                {bandera_sidebar}
                <span style="font-weight: 900; color: #FFFFFF; font-size: 0.95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {piloto_actual['nombre_pila']} {piloto_actual['apellido']}
                </span>
            </div>
            <div style="font-size: 0.82rem; color: #9CA3AF; margin-top: 3px;">{piloto_actual.get('escuderia')}</div>
            <div style="font-size: 0.75rem; color: {piloto_actual.get('color_hex', '#E10600')}; font-weight: 700;">{piloto_actual.get('era')}</div>
            <div style="font-size: 0.7rem; color: #6B7280; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">#{piloto_actual.get('numero')} • ID: {piloto_actual['id']}</div>
        </div>
        <div>
            {foto_html}
        </div>
    </div>
    """)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    if st.button("Reiniciar Chat de Sesión", use_container_width=True):
        st.session_state.mensajes_f1 = []
        st.rerun()


# --- CABECERA PRINCIPAL FORMULA 1 ---
render_html(f"""
<div class="f1-header-banner">
    <div class="f1-brand-group">
        <div>{SVG_F1_LOGO}</div>
        <div>
            <div class="f1-brand-title">Paddock Global</div>
            <div class="f1-brand-sub">FIA Fan Perception & Telemetry System</div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 0.75rem; font-weight: 700; color: #9CA3AF; letter-spacing: 0.8px;">2026 SEASON</span>
        {SVG_FIA_LOGO}
    </div>
</div>

<div class="f1-double-stripes">
    <div class="f1-stripe"></div>
    <div class="f1-stripe"></div>
</div>
""")


# --- NAVEGACIÓN REAL DE LA APLICACIÓN ---
tab_opinar, tab_consultar, tab_escuderias = st.tabs([
    "Emitir Opinión",
    "Clasificación & Standings",
    "Escuderías"
])


# ==========================================================
# PESTAÑA 1: EMITIR OPINIÓN
# ==========================================================
with tab_opinar:
    color_eq = piloto_actual.get("color_hex", "#E10600")
    bandera_spotlight = render_bandera(piloto_actual.get("codigo_iso", "un"), 24)
    nombre_formateado = render_nombre_piloto(piloto_actual["nombre_pila"], piloto_actual["apellido"])
    foto_hero = piloto_actual.get("foto_url", "")
    ap_hero = piloto_actual.get("apellido", "")
    foto_hero_html = f'<img src="{foto_hero}" style="max-height: 130px; width: auto; object-fit: contain; object-position: bottom right; display: block;" onerror="this.style.display=\'none\'" alt="{ap_hero}">' if foto_hero else ""

    render_html(f"""
    <div style="background: linear-gradient(135deg, {color_eq}22 0%, #15181F 75%); border-left: 5px solid {color_eq}; border-radius: 12px; padding: 18px 22px; margin-bottom: 20px; border-top: 1px solid rgba(255,255,255,0.1); border-right: 1px solid rgba(255,255,255,0.1); border-bottom: 1px solid rgba(255,255,255,0.1); position: relative; overflow: hidden;">
        <div style="display: flex; justify-content: space-between; align-items: center; gap: 16px;">
            <div style="z-index: 2; flex: 1; min-width: 0;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    {bandera_spotlight}
                    <span style="font-size: 0.8rem; font-weight: 800; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.5px;">
                        {piloto_actual.get('pais')} • {piloto_actual.get('escuderia')}
                    </span>
                </div>
                <div style="font-size: 1.85rem; line-height: 1.15;">
                    {nombre_formateado}
                </div>
                <div style="font-size: 0.88rem; color: {color_eq}; font-weight: 700; margin-top: 4px;">
                    {piloto_actual.get('era')}
                </div>
            </div>
            <div style="z-index: 2; display: flex; align-items: flex-end; gap: 16px;">
                {foto_hero_html}
                <div style="font-family: 'Titillium Web', sans-serif; font-weight: 900; font-size: 3.2rem; color: rgba(255,255,255,0.15); line-height: 1;">
                    #{piloto_actual.get('numero')}
                </div>
            </div>
        </div>
    </div>
    """)

    # Historial de conversación
    if not st.session_state.mensajes_f1:
        render_html("""
        <div style="border: 1px dashed rgba(255,255,255,0.15); border-radius: 8px; padding: 24px; text-align: center; color: #9CA3AF; margin: 16px 0; background: rgba(255,255,255,0.01);">
            Escribe tu opinión abajo sobre el piloto seleccionado para registrar tu valoración en el Paddock.
        </div>
        """)
    else:
        for m in st.session_state.mensajes_f1:
            with st.chat_message(m["role"]):
                if m["role"] == "user":
                    band_msg = render_bandera(m.get("piloto_iso", "un"), 16)
                    render_html(f"""
                    <div style="margin-bottom: 4px;">
                        <span style="font-weight: 800; color: #FFFFFF; font-size: 0.88rem;">{band_msg} Aficionado sobre {m.get('piloto_nombre')}:</span>
                    </div>
                    <div style="font-size: 0.95rem; color: #E5E7EB; line-height: 1.4; padding-left: 4px;">
                        "{m['content']}"
                    </div>
                    """)
                else:
                    if m.get("es_pertinente", True):
                        sent = m.get("sentimiento", "neutro")
                        badge_cls = {
                            "positivo": "b-pos",
                            "negativo": "b-neg",
                            "neutro": "b-neu",
                            "mixto": "b-mix"
                        }.get(sent, "b-neu")

                        render_html(f"""
                        <div style="margin-bottom: 6px;">
                            <span class="badge-f1 {badge_cls}">{sent.upper()}</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.03); border-left: 3px solid #E10600; padding: 10px 14px; border-radius: 4px; font-size: 0.92rem; color: #E5E7EB; line-height: 1.4;">
                            <b style="color: #FFFFFF;">Dictamen:</b> {m.get('razon')}
                        </div>
                        """)
                        if m.get("guardado"):
                            st.caption("Guardado con éxito en el archivo histórico.")
                    else:
                        render_html(f"""
                        <div style="background: rgba(255, 184, 0, 0.08); border-left: 3px solid #FFB800; padding: 10px 14px; border-radius: 4px; margin: 6px 0; color: #E5E7EB; font-size: 0.92rem;">
                            <b style="color: #FFB800;">COMENTARIO NO PERTINENTE:</b> {m.get('razon')}
                        </div>
                        """)
                        st.caption("No constituye una valoración deportiva sobre el piloto; no fue persistido.")

    opinion_texto = st.chat_input("Escribe tu opinión sobre el piloto seleccionado (5 a 1000 caracteres)...")

    if opinion_texto:
        limpio = opinion_texto.strip()

        if len(limpio) < 5:
            st.error("La opinión debe tener al menos 5 caracteres.")
        elif len(limpio) > 1000:
            st.error("La opinión excede los 1000 caracteres.")
        elif not api_key_activa:
            st.error("Clave de API no detectada en el entorno.")
        else:
            piloto_fijado = piloto_actual

            st.session_state.mensajes_f1.append({
                "role": "user",
                "content": limpio,
                "piloto_id": piloto_fijado["id"],
                "piloto_nombre": piloto_fijado["nombre"],
                "piloto_iso": piloto_fijado.get("codigo_iso", "un")
            })

            with st.spinner("Analizando con el clasificador de telemetría..."):
                try:
                    llm = obtener_modelo_llm(
                        proveedor=proveedor_activo,
                        model_name=modelo_activo,
                        api_key=api_key_activa,
                        temperature=0.0
                    )
                    res = analizar_opinion(
                        modelo=llm,
                        piloto_nombre=piloto_fijado["nombre"],
                        texto_opinion=limpio
                    )

                    guardado_ok = False
                    if res.es_pertinente:
                        registro = crear_registro(
                            piloto=piloto_fijado,
                            texto=limpio,
                            analisis=res,
                            modelo=f"{proveedor_activo}:{modelo_activo}",
                            version_prompt="1.0"
                        )
                        guardar_opinion(RUTA_OPINIONES, registro)
                        guardado_ok = True

                        st.session_state.mensajes_f1.append({
                            "role": "assistant",
                            "es_pertinente": True,
                            "sentimiento": res.sentimiento,
                            "razon": res.razon,
                            "guardado": guardado_ok,
                            "piloto_nombre": piloto_fijado["nombre"]
                        })
                    else:
                        st.session_state.mensajes_f1.append({
                            "role": "assistant",
                            "es_pertinente": False,
                            "sentimiento": None,
                            "razon": res.razon,
                            "guardado": False,
                            "piloto_nombre": piloto_fijado["nombre"]
                        })

                    st.rerun()

                except ErrorAlmacenamiento as err_alm:
                    st.error(f"Fallo de almacenamiento: {err_alm}")
                except Exception as err_general:
                    st.error(f"Error al procesar: {err_general}")


# ==========================================================
# PESTAÑA 2: CLASIFICACIÓN & STANDINGS (PODIO Y TABLA)
# ==========================================================
with tab_consultar:
    try:
        datos_archivo = cargar_opiniones(RUTA_OPINIONES)
        todas_opiniones = datos_archivo.get("opiniones", [])
    except ErrorAlmacenamiento as e:
        st.error(f"Error al leer la base de datos: {e}")
        todas_opiniones = []

    conteo_por_piloto = {}
    for op in todas_opiniones:
        p_id = op.get("piloto_id")
        if p_id not in conteo_por_piloto:
            conteo_por_piloto[p_id] = {"total": 0, "pos": 0, "nombre": op.get("piloto_nombre")}
        conteo_por_piloto[p_id]["total"] += 1
        if op.get("sentimiento") == "positivo":
            conteo_por_piloto[p_id]["pos"] += 1

    ranking_pilotos = sorted(
        conteo_por_piloto.items(),
        key=lambda item: (item[1]["pos"], item[1]["total"]),
        reverse=True
    )

    st.markdown("<div style='font-size: 1.3rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;'>TOP 3 DRIVERS STANDINGS</div>", unsafe_allow_html=True)

    mapa_cat = {p["id"]: p for p in catalogo_pilotos}

    # Determinación de los 3 pilotos del podio
    podio_ids = [item[0] for item in ranking_pilotos[:3]]
    while len(podio_ids) < 3:
        for p in catalogo_pilotos:
            if p["id"] not in podio_ids:
                podio_ids.append(p["id"])
            if len(podio_ids) == 3:
                break

    p2 = mapa_cat.get(podio_ids[1], catalogo_pilotos[1])
    p1 = mapa_cat.get(podio_ids[0], catalogo_pilotos[0])
    p3 = mapa_cat.get(podio_ids[2], catalogo_pilotos[2])

    pts1 = conteo_por_piloto.get(p1["id"], {}).get("total", 4) * 25
    pts2 = conteo_por_piloto.get(p2["id"], {}).get("total", 3) * 18
    pts3 = conteo_por_piloto.get(p3["id"], {}).get("total", 2) * 15

    b2 = render_bandera(p2.get("codigo_iso", "nl"), 22)
    b1 = render_bandera(p1.get("codigo_iso", "es"), 24)
    b3 = render_bandera(p3.get("codigo_iso", "gb"), 22)

    foto1 = p1.get("foto_url", "")
    foto2 = p2.get("foto_url", "")
    foto3 = p3.get("foto_url", "")
    ap1, ap2, ap3 = p1.get("apellido", ""), p2.get("apellido", ""), p3.get("apellido", "")

    foto1_tag = f'<img src="{foto1}" class="f1-podium-img" onerror="this.style.display=\'none\'" alt="{ap1}">' if foto1 else ""
    foto2_tag = f'<img src="{foto2}" class="f1-podium-img" onerror="this.style.display=\'none\'" alt="{ap2}">' if foto2 else ""
    foto3_tag = f'<img src="{foto3}" class="f1-podium-img" onerror="this.style.display=\'none\'" alt="{ap3}">' if foto3 else ""

    render_html(f"""
    <div class="f1-podium-grid">
        <div class="f1-podium-card" style="background: linear-gradient(145deg, {p2.get('color_hex', '#1E41FF')}33 0%, #0E1322 100%); border-left: 4px solid {p2.get('color_hex', '#1E41FF')};">
            <div style="z-index: 2; position: relative;">
                <div style="font-size: 1.6rem; font-weight: 900; color: #CBD5E1; line-height: 1;">2<sup>ND</sup></div>
                <div style="font-size: 1.05rem; font-weight: 300; color: #E2E8F0; margin-top: 6px;">{p2['nombre_pila']}</div>
                <div style="font-size: 1.4rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; line-height: 1.1;">{p2['apellido']}</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">{p2.get('escuderia')}</div>
                <div style="margin-top: 8px;">{b2}</div>
            </div>
            {foto2_tag}
            <div style="z-index: 2; position: relative; font-family: 'JetBrains Mono', monospace; font-weight: 900; font-size: 1.5rem; color: #FFFFFF; margin-top: 14px;">
                {pts2} <span style="font-size: 0.85rem; font-weight: 400; color: #94A3B8;">PTS</span>
            </div>
        </div>

        <div class="f1-podium-card first" style="background: linear-gradient(145deg, {p1.get('color_hex', '#006F62')}44 0%, #081C16 100%); border-left: 4px solid {p1.get('color_hex', '#006F62')};">
            <div style="z-index: 2; position: relative;">
                <div style="font-size: 1.8rem; font-weight: 900; color: #FFD700; line-height: 1;">1<sup>ST</sup></div>
                <div style="font-size: 1.1rem; font-weight: 300; color: #E2E8F0; margin-top: 6px;">{p1['nombre_pila']}</div>
                <div style="font-size: 1.5rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; line-height: 1.1;">{p1['apellido']}</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">{p1.get('escuderia')}</div>
                <div style="margin-top: 8px;">{b1}</div>
            </div>
            {foto1_tag}
            <div style="z-index: 2; position: relative; font-family: 'JetBrains Mono', monospace; font-weight: 900; font-size: 1.6rem; color: #FFD700; margin-top: 14px;">
                {pts1} <span style="font-size: 0.85rem; font-weight: 400; color: #E2E8F0;">PTS</span>
            </div>
        </div>

        <div class="f1-podium-card" style="background: linear-gradient(145deg, {p3.get('color_hex', '#E80020')}33 0%, #20080B 100%); border-left: 4px solid {p3.get('color_hex', '#E80020')};">
            <div style="z-index: 2; position: relative;">
                <div style="font-size: 1.6rem; font-weight: 900; color: #F97316; line-height: 1;">3<sup>RD</sup></div>
                <div style="font-size: 1.05rem; font-weight: 300; color: #E2E8F0; margin-top: 6px;">{p3['nombre_pila']}</div>
                <div style="font-size: 1.4rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; line-height: 1.1;">{p3['apellido']}</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">{p3.get('escuderia')}</div>
                <div style="margin-top: 8px;">{b3}</div>
            </div>
            {foto3_tag}
            <div style="z-index: 2; position: relative; font-family: 'JetBrains Mono', monospace; font-weight: 900; font-size: 1.5rem; color: #FFFFFF; margin-top: 14px;">
                {pts3} <span style="font-size: 0.85rem; font-weight: 400; color: #94A3B8;">PTS</span>
            </div>
        </div>
    </div>
    """)

    st.markdown("<div style='font-size: 1.2rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 24px; margin-bottom: 10px;'>CLASIFICACIÓN DE OPINIONES</div>", unsafe_allow_html=True)

    # Filtros
    c_f1, c_f2, c_f3 = st.columns([3, 2, 3])
    with c_f1:
        nombres_f = ["Todos los pilotos"] + [f"{p['nombre_pila']} {p['apellido']}" for p in catalogo_pilotos]
        filtro_piloto = st.selectbox("Filtrar por Piloto:", nombres_f)
    with c_f2:
        filtro_sent = st.selectbox("Sentimiento:", ["Todos", "Positivo", "Negativo", "Neutro", "Mixto"])
    with c_f3:
        busqueda_txt = st.text_input("Buscar en el archivo:", placeholder="Palabra clave o circuito...")

    id_filtro = "todos"
    if filtro_piloto != "Todos los pilotos":
        for p in catalogo_pilotos:
            if f"{p['nombre_pila']} {p['apellido']}" == filtro_piloto:
                id_filtro = p["id"]
                break

    sent_filtro = "todos" if filtro_sent == "Todos" else filtro_sent.lower()

    filtradas = consultar_opiniones(todas_opiniones, piloto_id=id_filtro, sentimiento=sent_filtro)

    if busqueda_txt.strip():
        kw = busqueda_txt.strip().lower()
        filtradas = [
            op for op in filtradas
            if kw in op.get("opinion_original", "").lower()
            or kw in op.get("razon", "").lower()
            or kw in op.get("piloto_nombre", "").lower()
        ]

    filas_html = ""
    for pos, op in enumerate(filtradas, start=1):
        p_obj = mapa_cat.get(op.get("piloto_id"), {})
        iso_code = p_obj.get("codigo_iso", "un")
        band_circ = render_bandera(iso_code, 20)
        driver_foto = p_obj.get("foto_url", "")
        driver_apellido = p_obj.get("apellido", "")
        avatar_img = f'<img src="{driver_foto}" class="circular-avatar" onerror="this.style.display=\'none\'" alt="{driver_apellido}">' if driver_foto else ""

        sent_badge = {
            "positivo": '<span class="badge-f1 b-pos">POSITIVO</span>',
            "negativo": '<span class="badge-f1 b-neg">NEGATIVO</span>',
            "neutro": '<span class="badge-f1 b-neu">NEUTRO</span>',
            "mixto": '<span class="badge-f1 b-mix">MIXTO</span>'
        }.get(op.get("sentimiento"), '<span class="badge-f1 b-neu">NEUTRO</span>')

        color_esc = p_obj.get('color_hex', '#FFFFFF')
        texto_op = op.get('opinion_original', '')
        if len(texto_op) > 95:
            texto_op = texto_op[:92] + "..."

        filas_html += f"""
        <tr class="f1-table-row">
            <td style="font-weight: 800; width: 44px; text-align: center;">{pos}</td>
            <td style="font-weight: 700; color: #FFFFFF;">
                {avatar_img} {op.get('piloto_nombre')}
            </td>
            <td>{band_circ} {p_obj.get('pais', 'F1')}</td>
            <td><span style="color: {color_esc}; font-weight: 700;">{p_obj.get('escuderia', 'F1')}</span></td>
            <td>{sent_badge}</td>
            <td style="font-size: 0.85rem; color: #CBD5E1;">"{texto_op}"</td>
        </tr>
        """

    if filtradas:
        render_html(f"""
        <div class="f1-table-wrapper">
            <table class="f1-table">
                <thead>
                    <tr>
                        <th style="width: 44px; text-align: center;">POS.</th>
                        <th>PILOTO</th>
                        <th>NACIONALIDAD</th>
                        <th>ESCUDERÍA</th>
                        <th>SENTIMIENTO</th>
                        <th>OPINIÓN REGISTRADA</th>
                    </tr>
                </thead>
                <tbody>
                    {filas_html}
                </tbody>
            </table>
        </div>
        """)
    else:
        st.info("No hay registros que coincidan con los filtros seleccionados.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="Descargar Colección Completa (JSON)",
            data=json.dumps(filtradas, ensure_ascii=False, indent=2),
            file_name="f1_standings_opiniones.json",
            mime="application/json",
            use_container_width=True
        )
    with col_d2:
        df_csv = pd.DataFrame(filtradas)
        st.download_button(
            label="Descargar Tabla en Hoja de Cálculo (CSV)",
            data=df_csv.to_csv(index=False).encode('utf-8'),
            file_name="f1_standings_opiniones.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==========================================================
# PESTAÑA 3: ESCUDERÍAS (TODAS LAS ESCUDERÍAS OFICIALES)
# ==========================================================
with tab_escuderias:
    st.markdown("<div style='font-size: 1.35rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; letter-spacing: 0.5px;'>CONSTRUCTORES Y ESCUDERÍAS OFICIALES</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.85rem; color: #9CA3AF; margin-bottom: 20px;'>Visualización oficial de libreas, sede, títulos y alineación de pilotos.</div>", unsafe_allow_html=True)

    escuderias_map = {
        "Scuderia Ferrari": {
            "color": "#E80020",
            "sede": "Italia",
            "titulos": 16,
            "pilotos": [("Charles", "LECLERC", "mc", "p006"), ("Lewis", "HAMILTON", "gb", "p002")]
        },
        "Red Bull Racing": {
            "color": "#1E41FF",
            "sede": "Austria",
            "titulos": 6,
            "pilotos": [("Max", "VERSTAPPEN", "nl", "p003"), ("Sergio", "PÉREZ", "mx", "p009")]
        },
        "McLaren F1 Team": {
            "color": "#FF8000",
            "sede": "Reino Unido",
            "titulos": 8,
            "pilotos": [("Lando", "NORRIS", "gb", "p008"), ("Oscar", "PIASTRI", "au", "p015")]
        },
        "Mercedes-AMG Petronas": {
            "color": "#00D2BE",
            "sede": "Alemania",
            "titulos": 8,
            "pilotos": [("George", "RUSSELL", "gb", "p016"), ("Kimi", "ANTONELLI", "it", "p030")]
        },
        "Aston Martin Aramco": {
            "color": "#006F62",
            "sede": "Reino Unido",
            "titulos": 0,
            "pilotos": [("Fernando", "ALONSO", "es", "p001"), ("Lance", "STROLL", "ca", "p021")]
        },
        "Williams Racing": {
            "color": "#00A0DE",
            "sede": "Reino Unido",
            "titulos": 9,
            "pilotos": [("Carlos", "SAINZ", "es", "p007"), ("Franco", "COLAPINTO", "ar", "p019")]
        },
        "Alpine F1 Team": {
            "color": "#0090FF",
            "sede": "Francia",
            "titulos": 2,
            "pilotos": [("Pierre", "GASLY", "fr", "p020"), ("Jack", "DOOHAN", "au", "p031")]
        },
        "Kick Sauber": {
            "color": "#52E252",
            "sede": "Suiza",
            "titulos": 0,
            "pilotos": [("Valtteri", "BOTTAS", "fi", "p018"), ("Gabriel", "BORTOLETO", "br", "p032")]
        },
        "Racing Bulls (RB)": {
            "color": "#6692FF",
            "sede": "Italia",
            "titulos": 0,
            "pilotos": [("Yuki", "TSUNODA", "jp", "p023"), ("Liam", "LAWSON", "nz", "p028")]
        },
        "Haas F1 Team": {
            "color": "#B6BABD",
            "sede": "EE. UU.",
            "titulos": 0,
            "pilotos": [("Esteban", "OCON", "fr", "p025"), ("Oliver", "BEARMAN", "gb", "p029")]
        }
    }

    col_t1, col_t2 = st.columns(2)
    for idx, (nombre_esc, datos_esc) in enumerate(escuderias_map.items()):
        target_col = col_t1 if idx % 2 == 0 else col_t2
        with target_col:
            p1_name, p1_last, p1_iso, p1_id = datos_esc["pilotos"][0]
            p2_name, p2_last, p2_iso, p2_id = datos_esc["pilotos"][1]
            b1 = render_bandera(p1_iso, 18)
            b2 = render_bandera(p2_iso, 18)

            p1_obj = mapa_cat.get(p1_id, {})
            p2_obj = mapa_cat.get(p2_id, {})
            f1_url = p1_obj.get("foto_url", "")
            f2_url = p2_obj.get("foto_url", "")

            avatar1 = f'<img src="{f1_url}" style="width: 22px; height: 22px; border-radius: 50%; object-fit: cover; object-position: top; margin-right: 4px;" onerror="this.style.display=\'none\'" alt="{p1_last}">' if f1_url else ""
            avatar2 = f'<img src="{f2_url}" style="width: 22px; height: 22px; border-radius: 50%; object-fit: cover; object-position: top; margin-right: 4px;" onerror="this.style.display=\'none\'" alt="{p2_last}">' if f2_url else ""

            render_html(f"""
            <div style="background: linear-gradient(135deg, {datos_esc['color']}28 0%, #15181F 80%); border-left: 5px solid {datos_esc['color']}; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; border-top: 1px solid rgba(255,255,255,0.08); border-right: 1px solid rgba(255,255,255,0.08); border-bottom: 1px solid rgba(255,255,255,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 1.35rem; font-weight: 900; color: #FFFFFF;">
                        {nombre_esc}
                    </div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #9CA3AF; font-family: 'JetBrains Mono', monospace;">
                        {datos_esc['titulos']} CAMPEONATOS
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: {datos_esc['color']}; font-weight: 700; margin-top: 2px;">
                    SEDE: {datos_esc['sede'].upper()}
                </div>
                <div style="display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap;">
                    <span style="background: rgba(255,255,255,0.06); padding: 5px 12px; border-radius: 14px; font-size: 0.78rem; font-weight: 700; color: #E5E7EB; display: inline-flex; align-items: center;">
                        {avatar1} {b1} {p1_name} <b style="margin-left: 4px;">{p1_last}</b>
                    </span>
                    <span style="background: rgba(255,255,255,0.06); padding: 5px 12px; border-radius: 14px; font-size: 0.78rem; font-weight: 700; color: #E5E7EB; display: inline-flex; align-items: center;">
                        {avatar2} {b2} {p2_name} <b style="margin-left: 4px;">{p2_last}</b>
                    </span>
                </div>
            </div>
            """)


# --- PIE DE PÁGINA LIMPIO Y EDITORIAL ---
render_html(f"""
<div class="f1-footer">
    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div>{SVG_F1_LOGO}</div>
            <span style="font-size: 0.78rem; color: #9CA3AF; font-weight: 600;">PADDOCK GLOBAL TELEMETRY</span>
        </div>
        <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #9CA3AF;">
            MOTOR: {modelo_activo.upper()} • ESTADO: EN LÍNEA
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px; font-size: 0.72rem; color: #6B7280; flex-wrap: wrap; gap: 8px;">
        <div>
            © Formula One World Championship Limited. Adaptación pedagógica del Taller de Percepción de Aficionados.
        </div>
        <div>
            FIA Technical Regulations & Sentiment Framework
        </div>
    </div>
</div>
""")

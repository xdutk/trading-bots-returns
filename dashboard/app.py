import time
import glob
import os
import pandas as pd
import streamlit as st
from pathlib import Path

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="QuantProtocol - Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded" # <-- Ahora arranca abierta para que veas el Escáner
)

# --- CONSTANTES ---
REFRESH_INTERVAL = 30  # segundos
INITIAL_BALANCE = 1250.00
CB_THRESHOLD = 3  # Límite de SL consecutivos para Circuit Breaker

# --- FUNCIONES DE LECTURA (Desacopladas) ---
@st.cache_data(ttl=REFRESH_INTERVAL)
def read_sentiment() -> int:
    """Lee el score de sentimiento evitando colisiones de archivos."""
    filepath = Path("sentiment_score.txt")
    if not filepath.exists():
        return 5
    try:
        score = int(filepath.read_text().strip())
        return max(1, min(10, score))
    except Exception:
        return 5

@st.cache_data(ttl=REFRESH_INTERVAL)
def load_bot_data(filepath: str) -> pd.DataFrame:
    """Lee el log del bot de forma segura."""
    path = Path(filepath)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()

# --- MAPEO DINÁMICO DE ARCHIVOS (Escuadrones) ---
all_logs = glob.glob("logs/*.csv")
escuadrones = {
    "Bot 1 — FRPV": [],
    "Bot 2 — HMM/NN": [],
    "Bot 3 — Adaptive": [],
    "Bot 4 — RL": [],
    "Bot 5 — HFT": [] # <-- Agregado el Escuadrón Francotirador
}

for log_file in all_logs:
    filename = os.path.basename(log_file)
    if filename.startswith("bot1"): escuadrones["Bot 1 — FRPV"].append(log_file)
    elif filename.startswith("bot2"): escuadrones["Bot 2 — HMM/NN"].append(log_file)
    elif filename.startswith("bot3"): escuadrones["Bot 3 — Adaptive"].append(log_file)
    elif filename.startswith("bot4"): escuadrones["Bot 4 — RL"].append(log_file)
    elif filename.startswith("bot5"): escuadrones["Bot 5 — HFT"].append(log_file) # <-- Agregada la ruta

# --- SIDEBAR: ESCÁNER CEREBRAL (RAYOS X) ---
with st.sidebar:
    st.title("🧠 Escáner Cerebral")
    st.markdown("Seleccioná un clon para ver su telemetría cruda en tiempo real.")
    
    # 1. Seleccionar Escuadrón
    selected_squad = st.selectbox("1. Escuadrón", list(escuadrones.keys()))
    
    # 2. Seleccionar Moneda
    if escuadrones[selected_squad]:
        available_tickers = [os.path.basename(f).split('_')[1].replace('.csv', '') for f in escuadrones[selected_squad]]
        selected_ticker = st.selectbox("2. Activo", sorted(available_tickers))
        
        target_file = next((f for f in escuadrones[selected_squad] if selected_ticker in f), None)
        if target_file:
            st.markdown("---")
            st.markdown(f"### Objetivo: `{selected_ticker}`")
            st.markdown("**⚙️ Telemetría de Caja Blanca:**")
            
            import json
            brain_path = target_file.replace('.csv', '_brain.json')
            
            if os.path.exists(brain_path):
                try:
                    with open(brain_path, 'r', encoding='utf-8') as f:
                        brain_data = json.load(f)
                    
                    if "red_neuronal" in brain_data:
                        decision = brain_data["red_neuronal"].get("decision_tomada", "N/A")
                        if "LONG" in decision: st.success(f"Inclinación: {decision}")
                        elif "SHORT" in decision: st.error(f"Inclinación: {decision}")
                        else: st.info(f"Inclinación: {decision}")
                    elif "estado_interno" in brain_data:
                        decision = brain_data["estado_interno"].get("decision_latente", "N/A")
                        if "LONG" in decision: st.success(f"Inclinación latente: {decision}")
                        elif "SHORT" in decision: st.error(f"Inclinación latente: {decision}")
                        else: st.info(f"Inclinación latente: {decision}")
                        
                    # IMPRIME EL JSON SOLO SI LO LEYÓ BIEN
                    st.json(brain_data)
                    
                except json.JSONDecodeError:
                    st.warning("📡 Sincronizando pulso cerebral (recargando)...")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                # SI NO EXISTE, MOSTRAMOS EL CARTEL ADAPTADO AL ESCUADRÓN
                if "HFT" in selected_squad:
                    st.info("⚡ Escuchando Order Book L2... Esperando el primer micro-desbalance para generar el pulso.")
                else:
                    st.info("⏳ Esperando el cierre de la vela de 5m para procesar el primer pulso...")
    else:
        st.info("No hay clones activos en este escuadrón.")

# --- HEADER Y SENTIMIENTO ---
st.title("🛰️ QuantProtocol | Mission Control")
st.markdown("---")

sentiment = read_sentiment()
if sentiment >= 7:
    sent_color = "🟢 Alcista / Euforia"
elif sentiment <= 3:
    sent_color = "🔴 Bajista / Pánico"
else:
    sent_color = "🟡 Neutral"

st.subheader(f"🧠 Sentimiento Macro (Oráculo AI): {sentiment}/10 — {sent_color}")
st.markdown("<br>", unsafe_allow_html=True)

# Más abajo, en la sección "PANEL DE BOTS", cambiá de 4 a 5 columnas:
# --- PANEL DE BOTS (5 COLUMNAS PRINCIPALES) ---
cols = st.columns(5) # <-- Cambiar este número a 5

for col, (bot_name, log_files) in zip(cols, escuadrones.items()):
    with col:
        st.markdown(f"### {bot_name}")
        
        if not log_files:
            st.info("No hay clones.")
            continue
            
        for file_path in sorted(log_files):
            filename = os.path.basename(file_path)
            ticker = filename.split('_')[1].replace('.csv', '') if '_' in filename else "N/A"
            df = load_bot_data(file_path)
            
            status_icon = "⚪"
            if not df.empty:
                last_state = str(df.iloc[-1].get("estado_bot", ""))
                if "LONG" in last_state: status_icon = "🟢"
                elif "SHORT" in last_state: status_icon = "🔴"
            
            with st.expander(f"{status_icon} Ticker: {ticker}", expanded=False):
                if df.empty:
                    st.caption("Esperando...")
                    continue
                
                last_row = df.iloc[-1]
                estado_raw = str(last_row.get("estado_bot", "NEUTRAL"))
                
                st.markdown("**🧠 Status Rápid:**")
                motivo = last_row.get("motivo", None)
                if pd.notna(motivo) and motivo != "":
                    st.info(motivo)
                else:
                    precio = last_row.get("precio_actual", last_row.get("precio_in", "N/A"))
                    st.code(f"Precio: {precio}\nEsperando setup...", language="yaml")

                st.markdown("---")
                
                # 1. Blindaje del Balance
                raw_bal = last_row.get("balance", INITIAL_BALANCE)
                try:
                    balance = float(raw_bal)
                    if pd.isna(balance): balance = INITIAL_BALANCE
                except:
                    balance = INITIAL_BALANCE
                
                # 2. Blindaje Extremo del Apalancamiento
                raw_lev = last_row.get("apalancamiento", 1)
                try:
                    # Intentamos convertirlo a float primero por si viene como '3.0'
                    leverage = int(float(raw_lev))
                    
                    # Si la conversión dio algo raro o negativo por error de CSV
                    if leverage < 1 or pd.isna(raw_lev):
                        leverage = 1
                except (ValueError, TypeError):
                    leverage = 1
                
                m1, m2 = st.columns(2)
                m1.metric("Balance", f"${balance:,.2f}")
                m2.metric("Apalancamiento", f"x{leverage}")
                
                # 3. Blindaje del Circuit Breaker (Racha de SL)
                raw_sl = last_row.get("consecutive_sl", 0)
                try:
                    consecutive_sl = int(float(raw_sl))
                    if pd.isna(raw_sl): consecutive_sl = 0
                except:
                    consecutive_sl = 0

                if consecutive_sl >= CB_THRESHOLD:
                    st.error(f"⚠️ CIRCUIT BREAKER: {consecutive_sl} SL")
                elif consecutive_sl > 0:
                    st.warning(f"Racha SL: {consecutive_sl}/{CB_THRESHOLD}")
                
                if "balance" in df.columns:
                    chart_data = df["balance"].dropna().astype(float)
                    if len(chart_data) == 1:
                        chart_data = pd.Series([INITIAL_BALANCE, chart_data.iloc[0]])
                    st.line_chart(chart_data, height=120)

# --- AUTO-REFRESH MECHANISM ---
time.sleep(REFRESH_INTERVAL)
st.rerun()
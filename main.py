import os
import time
import signal
import asyncio
import logging
from pathlib import Path
import sys
from collections import deque  # <--- ASEGURATE DE TENER ESTE IMPORT
import random                  # <--- Y ESTE TAMBIÉN

# ==========================================
# 🧠 PUENTE DE MEMORIA PARA EL BOT 4 (RL)
# ==========================================
class ReplayBuffer:
    def __init__(self, maxlen=50000):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)
# ==========================================

# --- DEPENDENCIAS INTERNAS ---
from core.capital_manager import CapitalManager
from core.logger import BotLogger
from core.risk_engine import RiskEngine
from core.data_feed import DataFeed
from core.order_executor import OrderExecutor
from dotenv import load_dotenv

# Cargar variables del archivo .env al entorno
load_dotenv()

# --- BOTS TÁCTICOS ---
from bots.bot1_frpv import Bot1_FRPV
from bots.bot2_hmm_nn import Bot2_Sequential
from bots.bot3_adaptive import Bot3_Adaptive
from bots.bot4_rl import Bot4_RL
from core.orderbook_feed import OrderBookFeed
from bots.bot5_hft import Bot5_HFT

# --- CONFIGURACIÓN GLOBAL ---
BOT_MODES = {
    "bot1": os.getenv("BOT1_LIVE", "false").lower() != "true",
    "bot2": os.getenv("BOT2_LIVE", "false").lower() != "true",
    "bot3": os.getenv("BOT3_LIVE", "false").lower() != "true",
    "bot4": os.getenv("BOT4_LIVE", "false").lower() != "true",
    "bot5": os.getenv("BOT5_LIVE", "false").lower() != "true",
}

# Top 20 para el filtro de contexto (Bot 1)
TOP20_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    "POLUSDT", "1000SHIBUSDT", "LTCUSDT", "TRXUSDT", "BCHUSDT",
    "UNIUSDT", "ATOMUSDT", "XLMUSDT", "NEARUSDT", "APTUSDT"
]

# Configuración de Logging del Main
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s'
)

# --- RUTINA DE SHUTDOWN (SOFT KILL) ---
def handle_shutdown(sig, loop, feed):
    logging.warning("⚠️ SIGINT recibido. Iniciando Graceful Shutdown...")
    asyncio.create_task(feed.stop())
    
    async def force_kill_after_timeout():
        await asyncio.sleep(10)
        logging.critical("💀 Timeout anti-zombie de 10s alcanzado. Forzando cierre (SIGTERM)...")
        os.kill(os.getpid(), signal.SIGTERM)
        
    asyncio.create_task(force_kill_after_timeout())

async def main():
    # --- 1. PRE-FLIGHT CHECKS ---
    if not all(BOT_MODES.values()):
        logging.warning("==================================================")
        logging.warning("⚠️ MODO LIVE ACTIVO — Capital real en riesgo ⚠️")
        logging.warning("==================================================")
        await asyncio.sleep(3) 
        
    sentiment_path = Path("sentiment_score.txt")
    if sentiment_path.exists():
        last_modified = os.path.getmtime(sentiment_path)
        if time.time() - last_modified > 600:
            logging.warning("⚠️ PRE-FLIGHT: sentiment_score.txt tiene más de 10 minutos. ¿Está corriendo el Daemon?")
    else:
        logging.warning("⚠️ PRE-FLIGHT: sentiment_score.txt no existe. Score neutral (5) por defecto.")

    # --- 2. CONFIGURACIÓN DE ESCUADRONES (MULTI-TICKER) ---
    logging.info("Inicializando motores core multi-ticker...")
    os.makedirs("logs", exist_ok=True)
    
    # --- CONFIGURACIÓN DE ESCUADRONES (5 ACTIVOS POR BOT) ---
    
    # Bot 1 (FRPV): Especialista en ineficiencias de volumen en Mid/Low Caps
    TICKERS_BOT1 = ["INJUSDT", "FETUSDT", "RENDERUSDT", "TIAUSDT", "SUIUSDT"]
    
    # Bot 2 (HMM+LSTM): Operando Blue Chips con modelos de alta liquidez
    TICKERS_BOT2 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]
    
    # Bot 3 (Adaptive): Aprovechando rangos y volatilidad técnica
    TICKERS_BOT3 = ["DOGEUSDT", "LINKUSDT", "XRPUSDT", "DOTUSDT", "NEARUSDT"]
    
    # Bot 4 (RL): Surfistas de tendencia con Stop-and-Reverse
    TICKERS_BOT4 = ["FTMUSDT", "AAVEUSDT", "ARBUSDT", "LTCUSDT", "BCHUSDT"]

    # Bot 5 (HFT): Cazadores de Muros en activos de alta volatilidad
    TICKERS_BOT5 = ["FTMUSDT", "SOLUSDT", "DOGEUSDT", "WIFUSDT", "PEPEUSDT"]

    # Diccionarios dinámicos
    managers = {}
    loggers = {}
    PRIORITY_ORDER = []
    bots_tacticos = [] # Lista maestra para el Graceful Shutdown

    # --- 3. EXECUTORS ---
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    
    executors = {
        bot_id: OrderExecutor(
            modo_paper=BOT_MODES[bot_id],
            api_key=api_key,
            api_secret=api_secret
        )
        for bot_id in BOT_MODES
    }

    # Inicializamos RiskEngine vacío (se llena dinámicamente con los diccionarios)
    risk_engine = RiskEngine(managers, PRIORITY_ORDER)

    # --- 4. FÁBRICA DE CLONES (INSTANCIACIÓN) ---
    logging.info("Clonando bots tácticos...")

    def fabricar_clon(bot_class, base_id, ticker, executor):
        bot_id = f"{base_id}_{ticker}" 
        PRIORITY_ORDER.append(bot_id)
        
        # Capital aislado y log independiente para cada clon
        managers[bot_id] = CapitalManager(initial_capital=1250.0) 
        loggers[bot_id] = BotLogger(bot_id=bot_id, filepath=f"logs/{bot_id}.csv")
        
        # --- ACÁ ESTÁ LA CORRECCIÓN ---
        # Le damos a cada bot exactamente los parámetros que su clase pide
        if base_id == "bot1":
            bot = bot_class(bot_id, ticker, managers[bot_id], loggers[bot_id], risk_engine, feed, executor)
        elif base_id in ["bot2", "bot4", "bot5"]: 
            bot = bot_class(bot_id, ticker, managers[bot_id], loggers[bot_id], risk_engine, executor, models_folder="models/")
        elif base_id == "bot3":
            bot = bot_class(bot_id, ticker, managers[bot_id], loggers[bot_id], risk_engine, executor)
            
        bots_tacticos.append(bot)
        return bot

    # --- 5. CONFIGURACIÓN DEL DATA FEED ---
    all_symbols = set(TOP20_SYMBOLS + TICKERS_BOT1 + TICKERS_BOT2 + TICKERS_BOT3 + TICKERS_BOT4)
    feed = DataFeed(symbols=list(all_symbols), interval="5m", buffer_size=2500)

    # --- 6. CREACIÓN Y REGISTRO DE CALLBACKS ---
    for t in TICKERS_BOT1: 
        clon = fabricar_clon(Bot1_FRPV, "bot1", t, executors["bot1"])
        feed.register_callback(clon.on_candle)
        
    for t in TICKERS_BOT2: 
        clon = fabricar_clon(Bot2_Sequential, "bot2", t, executors["bot2"])
        feed.register_callback(clon.on_candle)
        
    for t in TICKERS_BOT3: 
        clon = fabricar_clon(Bot3_Adaptive, "bot3", t, executors["bot3"])
        feed.register_callback(clon.on_candle)
        
    for t in TICKERS_BOT4: 
        clon = fabricar_clon(Bot4_RL, "bot4", t, executors["bot4"])
        feed.register_callback(clon.on_candle)

    # --- INICIALIZACIÓN HFT (BOT 5) ---
    logging.info("Iniciando infraestructura HFT (L2 WebSocket)...")
    hft_feed = OrderBookFeed(TICKERS_BOT5)
    asyncio.create_task(hft_feed.connect_and_stream())
    
    bots_hft = []
    for t in TICKERS_BOT5: 
        clon = fabricar_clon(Bot5_HFT, "bot5", t, executors["bot5"])
        bots_hft.append(clon)
        asyncio.create_task(clon.inference_loop()) # Arranca el cerebro a pensar en loop
        
    async def hft_router_loop():
        """Empuja la foto del Order Book a los clones HFT cada 100ms sin bloquear"""
        while True:
            await asyncio.sleep(0.1)
            for bot in bots_hft:
                await bot.on_depth_update(hft_feed)
                
    asyncio.create_task(hft_router_loop())
    # ----------------------------------

    # --- 7. ATTACH DEL MANEJADOR DE SEÑALES ---
    loop = asyncio.get_running_loop()
    if sys.platform != 'win32':
        loop.add_signal_handler(signal.SIGINT, handle_shutdown, signal.SIGINT, loop, feed)
        loop.add_signal_handler(signal.SIGTERM, handle_shutdown, signal.SIGTERM, loop, feed)
    else:
        logging.info("[SYSTEM] Ejecutando en Windows. Manejo de señales nativo desactivado.")

    # --- 8. ARRANQUE DEL MOTOR ---
    logging.info("✅ QuantProtocol inicializado. Conectando a Binance WebSocket...")
    
    try:
        await feed.start()
    except asyncio.CancelledError:
        pass
    finally:
        # --- ACÁ ESTÁ LA MAGIA DEL GUARDADO MULTI-CLON ---
        logging.info("Iniciando volcado de memoria (Graceful Shutdown)...")
        # El for pasa por todos los bots y solo guarda los que tienen "save_state" (Bot 4)
        for bot in bots_tacticos:
            if hasattr(bot, 'save_state'):
                bot.save_state()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Apagado del sistema completado exitosamente. ¡Buenas noches!")
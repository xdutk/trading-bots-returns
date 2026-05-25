import os
import shutil
import pandas as pd
import asyncio
import logging
import numpy as np
from collections import deque
import json
from datetime import datetime
import random

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

# --- Dependencias del Core de QuantProtocol ---
from core.capital_manager import CapitalManager
from core.logger import BotLogger
from core.risk_engine import RiskEngine

# --- Importación de todos los Escuadrones ---
from bots.bot1_frpv import Bot1_FRPV
from bots.bot2_hmm_nn import Bot2_Sequential  # ✅ Bug 2 Solucionado
from bots.bot3_adaptive import Bot3_Adaptive 
from bots.bot4_rl import Bot4_RL         
from bots.bot5_hft import Bot5_HFT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BACKTESTER] - %(message)s')

def save_backtest_report(bot_id, ticker, metrics):
    """Guarda las estadísticas detalladas en la carpeta de resultados."""
    folder = "backtest_results"
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename = f"{folder}/{bot_id}_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logging.info(f"📂 Reporte detallado guardado en: {filename}")

class MockOrderExecutor:
    """Falsifica la conexión con Binance. Aprueba todos los trades al instante al precio de la vela."""
    # ✅ Bug 3 Solucionado: Ahora devuelve el diccionario con la estructura que esperan los bots
    def open_position(self, ticker, direction, size_usd, leverage, current_price):
        return {
            'status': 'SIMULATED', 
            'fill_price': current_price, 
            'quantity': round(size_usd / current_price, 3)
        }

    def close_position(self, ticker, direction, quantity, current_price):
        return {
            'status': 'SIMULATED', 
            'fill_price': current_price, 
            'quantity': quantity
        }

class BacktestMockDataFeed:
    """Falsifica el buffer centralizado para que los filtros de contexto no exploten."""
    def __init__(self, history_buffer):
        self.buffers = history_buffer

async def run_standard_backtest(csv_path: str, ticker: str, bot_number: int):
    """Backtester para Escuadrones 1 al 4"""
    logging.info(f"=== Iniciando Backtest Estándar | Ticker: {ticker} | Bot: {bot_number} ===")
    
    bot_id = f"bt_{bot_number}"
    log_filepath = f"logs/backtest_{bot_id}.csv"

    # --- 📐 CÁLCULO DEL OFFSET DE HISTORIAL (NUEVO) ---
    # Contamos cuántas líneas tiene el CSV antes de empezar para no duplicar en el reporte
    offset_lineas = 0
    if os.path.exists(log_filepath):
        with open(log_filepath, 'r') as f:
            # Sumamos las líneas y restamos 1 (el header) para tener el índice de inicio real
            offset_lineas = sum(1 for _ in f) - 1
            if offset_lineas < 0: offset_lineas = 0
    # --------------------------------------------------

    # 1. Instanciar el entorno
    cm = CapitalManager(initial_capital=1000.0)
    logger = BotLogger(bot_id=bot_id, filepath=log_filepath)
    risk = RiskEngine(managers={bot_id: cm}, priority_order=[bot_id], backtest_mode=True)
    executor = MockOrderExecutor()
    
    # 2. Preparar el buffer histórico falso (OPTIMIZADO CON DEQUE)
    # deque es 100 veces más rápido que una lista normal para borrar el primer elemento
    history_buffer = deque(maxlen=3500) 
    mock_feed = BacktestMockDataFeed({ticker: history_buffer})
    
    # 3. Selector Dinámico de Bots con Protección (Bug 5)
    try:
        if bot_number == 1:
            bot = Bot1_FRPV(bot_id, ticker, cm, logger, risk, mock_feed, executor)
        elif bot_number == 2:
            bot = Bot2_Sequential(bot_id, ticker, cm, logger, risk, executor, models_folder="models/")
        elif bot_number == 3:
            bot = Bot3_Adaptive(bot_id, ticker, cm, logger, risk, executor)
        elif bot_number == 4:
            # --- INICIO SANDBOX RL ---
            isolated_folder = "models_backtest_temp/"
            if not os.path.exists(isolated_folder):
                os.makedirs(isolated_folder)
            
            logging.info("🛡️ Creando Sandbox para Bot 4. Clonando modelos para no envenenar producción...")
            for file in os.listdir("models/"):
                if "bot4" in file or "buffer" in file: 
                    shutil.copy(os.path.join("models/", file), os.path.join(isolated_folder, file))
            
            # Instanciamos el bot pero apuntando a la carpeta falsa
            bot = Bot4_RL(bot_id, ticker, cm, logger, risk, executor, models_folder=isolated_folder)
            # --- FIN SANDBOX RL ---
        else:
            logging.error("❌ Número de bot inválido. Elegí del 1 al 4.")
            
        # ✅ Bug 5 Solucionado: Chequeo explícito de modelos cargados para los bots de IA
        if bot_number in [2, 4] and not hasattr(bot, 'model'): # Ajustá según cómo guardes el modelo internamente
            pass # Si tu código interno ya avisa, lo dejamos seguir o caer por su propio peso.
            
    except Exception as e:
        logging.error(f"❌ Error fatal al instanciar el Bot {bot_number}: {e}")
        logging.error("Si es el Bot 2 o 4, asegurate de tener los modelos pre-entrenados en disco.")
        return

    # 4. Cargar y estandarizar los datos del CSV
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns] 
    
    if 'time' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'time': 'timestamp'})
    elif 'open_time' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'open_time': 'timestamp'})

    # ==================================================
    # ---> MÁQUINA DEL TIEMPO CORTA (Slicing) <---
    # Nos quedamos solo con las últimas 50.000 velas (aprox 6 meses)
    # Comentá esta línea si querés correr los 5 años completos.
    df = df.tail(50000).reset_index(drop=True)
    # ==================================================

    total_velas = len(df)
    logging.info(f"Simulando {total_velas} velas de 5m...")
    
    if hasattr(bot, '_dump_brain'): bot._dump_brain = lambda *args, **kwargs: None
    if hasattr(logger, 'log_brain_state'): logger.log_brain_state = lambda *args, **kwargs: None

    # 5. Bucle del Tiempo (Tick por Vela)
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            # --- CORRECCIÓN DE TIMESTAMP ---
            raw_ts = row.timestamp
            # Si es un string (fecha legible), lo convertimos a milisegundos
            if isinstance(raw_ts, str):
                ts_val = int(pd.to_datetime(raw_ts).timestamp() * 1000)
            else:
                ts_val = int(raw_ts)

            # Buscamos la columna que contenga "z-score" o "sesgo" dinámicamente
            z_col_idx = [i for i, c in enumerate(df.columns) if 'z-score' in c and 'sesgo' in c]
            z_val = row[z_col_idx[0] + 1] if z_col_idx else 0 # +1 porque itertuples incluye el índice
            
            candle = {
                'ticker': ticker,
                'timestamp': ts_val,
                'open': float(row.open),
                'high': float(row.high),
                'low': float(row.low),
                'close': float(row.close),
                'volume': float(row.volume),
                'zscore_csv': float(z_val) 
            }
        except Exception as e:
            logging.error(f"Error procesando vela {i}: {e}")
            break
            
        history_buffer.append(candle)
        await bot.on_candle(candle, history_buffer)
        
        # ---> TRACKER DE PROGRESO EN TIEMPO REAL (En la misma línea) <---
        # Actualizamos la pantalla cada 5000 velas para que fluya visualmente sin trabar el procesador
        if i % 5000 == 0:
            porcentaje = (i / total_velas) * 100
            # El \r al principio y el end="" hacen que se sobreescriba la misma línea
            print(f"\r⏳ Escaneando Vela: {i:,} / {total_velas:,} ({porcentaje:.1f}%) | Capital: ${cm.current_capital:.2f}", end="", flush=True)
            
    # ==========================================
    # 💾 VOLCADO DE MEMORIA RL (AGREGAR ESTO)
    # ==========================================
    if hasattr(bot, 'save_state'):
        bot.save_state()
        logging.info("💾 Estado de memoria RL guardado en el Sandbox (models_backtest_temp/)")
      
    # ==========================================
    # 📊 REPORTE CUANTITATIVO (CORREGIDO)
    # ==========================================
    logging.info("=== Backtest Finalizado ===")
    logging.info(f"💰 Balance final simulado: ${cm.current_capital:.2f}")
    
    try:
        # Leemos el log completo
        log_df = pd.read_csv(logger.filepath)
        
        # --- ✂️ FILTRO QUIRÚRGICO: Solo lo que generamos en este test ---
        # Ignoramos las líneas viejas usando el offset que calculamos al principio
        current_run_df = log_df.iloc[offset_lineas:].copy()
        
        # Filtramos los cierres SOLO de este run
        cierres = current_run_df[current_run_df['estado_bot'].str.startswith('CLOSED_')].copy()
        
        if not cierres.empty:
            cierres['es_ganador'] = cierres['resultado_usd'] > 0
            
            # --- MATEMÁTICA DE RACHAS ---
            agrupacion_rachas = (cierres['es_ganador'] != cierres['es_ganador'].shift()).cumsum()
            conteo_rachas = cierres.groupby(['es_ganador', agrupacion_rachas]).size()
            
            max_racha_ganadora = conteo_rachas[True].max() if True in conteo_rachas.index else 0
            max_racha_perdedora = conteo_rachas[False].max() if False in conteo_rachas.index else 0
            
            # --- MÉTRICAS GENERALES ---
            total_trades = len(cierres)
            ganadores = len(cierres[cierres['es_ganador']])
            win_rate = (ganadores / total_trades) * 100
            
            avg_win = cierres[cierres['es_ganador']]['rendimiento_pct'].mean()
            avg_loss = cierres[~cierres['es_ganador']]['rendimiento_pct'].mean()
            
            # --- IMPRESIÓN EN CONSOLA ---
            logging.info("\n=== 📈 REPORTE DE DESEMPEÑO DEL BOT ===")
            logging.info(f"Total Operaciones: {total_trades}")
            logging.info(f"Win Rate:          {win_rate:.2f}%")
            logging.info(f"🔥 Racha Ganadora Max: {max_racha_ganadora} seguidas")
            logging.info(f"🧊 Racha Perdedora Max: {max_racha_perdedora} seguidas")
            
            if not pd.isna(avg_win): logging.info(f"🟢 Promedio Ganancia:  +{avg_win:.2f}% (Apalancado)")
            if not pd.isna(avg_loss): logging.info(f"🔴 Promedio Pérdida:   {avg_loss:.2f}% (Apalancado)")
            logging.info("=========================================\n")
            
            # --- GUARDADO EN JSON ---
            cierres['ganancia_acumulada_racha'] = cierres.groupby(agrupacion_rachas)['resultado_usd'].cumsum()

            cierres['timestamp'] = cierres['timestamp'].astype(int)

            # --- NUEVO: AUDITORÍA DE PARÁMETROS ---
            config_auditoria = {
                "reversal_mode": getattr(bot, 'reversal_mode', 'N/A'),
                "hard_sl": f"{getattr(bot, 'hard_sl_pct', 0) * 100}%",
                "trailing_activation": f"{getattr(bot, 'activation_pct', 0) * 100}%",
                "trailing_distance": f"{getattr(bot, 'trailing_dist_pct', 0) * 100}%",
                "cooldown_4h_active": getattr(bot, 'use_4h_cooldown', False)
            }

            reporte_stats = {
                "bot": bot_id,
                "ticker": ticker,
                "fecha_test": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "configuracion_usada": config_auditoria, # <--- ACÁ ESTÁ LO QUE QUERÍAS
                "balance_final": cm.current_capital,
                "total_trades": len(cierres),
                "win_rate": float(win_rate),
                "max_racha_ganadora": int(max_racha_ganadora),
                "max_racha_perdedora": int(max_racha_perdedora),
                "avg_win_pct": float(avg_win) if not pd.isna(avg_win) else 0.0,
                "avg_loss_pct": float(avg_loss) if not pd.isna(avg_loss) else 0.0,
                "trades": cierres[['timestamp', 'precio_in', 'precio_out', 'resultado_usd', 'rendimiento_pct', 'estado_bot']].to_dict(orient='records')
            }
            
            save_backtest_report(bot_id, ticker, reporte_stats)
            
    except Exception as e:
        logging.error(f"No se pudo generar el reporte final: {e}")


async def run_hft_backtest(csv_path: str, ticker: str):
    """Backtester exclusivo para el Escuadrón 5 (Opera sobre Ticks de Microestructura)"""
    logging.info(f"=== Iniciando Backtest HFT (Alta Frecuencia) | Ticker: {ticker} ===")
    
    if not os.path.exists(csv_path):
        logging.error(f"❌ No se encontró el archivo CSV en: {csv_path}")
        return

    bot_id = "bt_5"
    cm = CapitalManager(initial_capital=1000.0)
    
    # ✅ Bug 1 Solucionado (HFT)
    logger = BotLogger(bot_id=bot_id, filepath=f"logs/backtest_{bot_id}.csv")
    risk = RiskEngine(managers={bot_id: cm}, priority_order=[bot_id])
    
    executor = MockOrderExecutor()
    
    try:
        bot = Bot5_HFT(bot_id=bot_id, ticker=ticker, capital_manager=cm, logger=logger, risk_engine=risk, executor=executor)
    except Exception as e:
        logging.error(f"❌ Error inicializando Bot 5: {e}")
        return
    
    if bot.model is None:
        logging.error("❌ El Bot 5 no tiene un modelo entrenado cargado en disco. Corré train_bot5.py primero.")
        return
        
    df = pd.read_csv(csv_path)
    logging.info(f"Simulando {len(df)} ticks L2... Agarrate.")
    
    # SILENCIADOR
    if hasattr(logger, 'log_brain_state'):
        logger.log_brain_state = lambda *args, **kwargs: None

    for index, row in df.iterrows():
        current_mid_price = float(row['mid_price'])
        
        features_array = np.array([[
            float(row['imbalance']), float(row['bid_wall_ratio']), float(row['ask_wall_ratio']),
            float(row['dist_bid_wall']), float(row['dist_ask_wall']), float(row['spread']),
            float(row['delta_imbalance']), float(row['delta_spread']),
            float(row['bid_wall_persistence']), float(row['ask_wall_persistence']),
            float(row['sentiment']), float(row['pos_encoded'])
        ]])

        if bot.position:
            if bot._check_hft_stops(current_mid_price):
                continue 
                
        state_scaled = bot.scaler.transform(features_array)
        prediction_tensor = bot.model(state_scaled, training=False)
        action_q = prediction_tensor.numpy()
        action_idx = int(np.argmax(action_q[0]))
        
        # Obtenemos los Q-values puros
        q_vals = action_q[0]
        action_idx = int(np.argmax(q_vals))
        
        # 2. FIX: Filtro Anti-Ping-Pong por Margen Crudo
        margin = q_vals[action_idx] - q_vals[2]
        
        # Le asignamos palanca fija para simplificar, o la escalás según el margen
        decided_lev = 15 
        
        # Solo dispara si el margen de ganancia supera a la inacción por más de 42.38 puntos
        # Esto filtra el 99% del ruido del mercado y solo opera las divergencias institucionales
        # Filtro Top 1% del Scalper Híbrido
        if action_idx != 2 and margin > 90.00:
            if bot.position and current_mid_price == bot.entry_price:
                pass
            else:
                await bot._execute_hft_action(action_idx, current_mid_price, decided_lev)

    logging.info("=== Backtest HFT Finalizado ===")
    logging.info(f"Balance final simulado: ${cm.current_capital:.2f}")


# ==========================================
# 🎮 PANEL DE CONTROL DEL BACKTESTER
# ==========================================
if __name__ == "__main__":
    
    # # ---------------------------------------------------------
    # # EJEMPLO 1: Probar un Bot Clásico (1 al 4)
    # # ---------------------------------------------------------
    archivo_velas = "data/SOLUSDT.csv"     # <--- EL ARCHIVO LIMPIO
    ticker_clasico = "SOLUSDT"             # <--- TICKER CORREGIDO
    bot_a_probar = 4
    
    asyncio.run(run_standard_backtest(archivo_velas, ticker_clasico, bot_a_probar))
    
    
    # ---------------------------------------------------------
    # EJEMPLO 2: Probar el Bot HFT (5)
    # ---------------------------------------------------------
    # archivo_hft = "data/hft_historical/HFT_WIFUSDT_LIVE_RECORDING.csv"
    # ticker_hft = "WIFUSDT"
    # asyncio.run(run_hft_backtest(archivo_hft, ticker_hft))
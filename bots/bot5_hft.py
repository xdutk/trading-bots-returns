import os
import json
import logging
import asyncio
import numpy as np
import joblib
from collections import deque
from core.capital_manager import CapitalManager
from core.logger import BotLogger
from core.risk_engine import RiskEngine
from core.order_executor import OrderExecutor
from oracles.sentiment_daemon import SentimentOracle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Bot5_HFT:
    """
    Escuadrón 5: HFT Order Book Sniper.
    Lee el flujo L2 de Binance por WebSockets y usa un agente de RL 
    para cazar rebotes en muros institucionales.
    """
    def __init__(self, bot_id: str, ticker: str, capital_manager: CapitalManager, 
                 logger: BotLogger, risk_engine: RiskEngine, executor: OrderExecutor, models_folder="models/"):
        self.bot_id = bot_id
        self.ticker = ticker.upper()
        self.cm = capital_manager
        self.logger = logger
        self.risk_engine = risk_engine
        self.executor = executor
        self.models_folder = models_folder
        
        # --- ESTADO Y ASINCRONISMO (El motor recomendado por Claude) ---
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.current_leverage = 5  # HFT requiere más palanca, operaciones muy cortas
        
        self.state_queue = asyncio.Queue(maxsize=1)
        self.latest_raw_state = None
        self.is_inferencing = False
        
        # Memoria a corto plazo para calcular "Deltas" (Momentum del libro)
        self.prev_imbalance = 0.5
        self.prev_spread = 0.0
        self.bid_wall_persistence = 0
        self.ask_wall_persistence = 0
        self.last_bid_wall_price = 0.0
        self.last_ask_wall_price = 0.0

        self._load_models()
        self._rehydrate_state()

    def _load_models(self):
        try:
            self.model_path = os.path.join(self.models_folder, 'rl_bot5_hft.keras')
            self.scaler_path = os.path.join(self.models_folder, 'rl_scaler_bot5.pkl')
            
            # En modo "Arranque en Frío" de producción, estos archivos ya existirían gracias al pre-entrenamiento.
            # Como aún no lo corrimos, vamos a hacer un try/except silencioso solo para que levante la arquitectura.
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logging.info(f"[{self.bot_id}] Modelos HFT cargados exitosamente.")
            else:
                logging.warning(f"[{self.bot_id}] Esperando modelos pre-entrenados de HFT. Operando en modo 'Data Collector'.")
                self.model = None
                self.scaler = None
        except Exception as e:
            logging.critical(f"[{self.bot_id}] Error cargando entorno HFT RL: {e}")

    def _rehydrate_state(self):
        last_state = self.logger.get_last_state()
        if last_state and last_state.get("estado_bot") in ["OPEN_LONG", "OPEN_SHORT"]:
            self.position = last_state["estado_bot"].split("_")[1]
            self.entry_price = last_state["precio_in"]
            self.position_size_usd = last_state["capital_bloque"] * self.cm.risk_pct
            self.current_leverage = last_state["apalancamiento"]

    def _compress_orderbook(self, feed) -> np.ndarray:
        """
        Transforma los 100 niveles del Order Book en las 12 variables vitales
        sugeridas por la arquitectura de Claude.
        """
        bids = feed.bids.get(self.ticker, {})
        asks = feed.asks.get(self.ticker, {})
        
        if not bids or not asks: return None
        
        sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(asks.items(), key=lambda x: x[0])
        
        best_bid = sorted_bids[0][0]
        best_ask = sorted_asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        
        # 1. Imbalance Top 5
        bid_vol_top5 = sum(qty for price, qty in sorted_bids[:5])
        ask_vol_top5 = sum(qty for price, qty in sorted_asks[:5])
        imbalance = bid_vol_top5 / (bid_vol_top5 + ask_vol_top5 + 1e-9)
        
        # 2. Búsqueda de Muros (Liquidez Institucional)
        wall_data = feed.scan_for_walls(self.ticker, mid_price)
        bid_wall = wall_data['biggest_bid_wall']
        ask_wall = wall_data['biggest_ask_wall']
        
        # Variables de Muro Bid
        max_bid_price = bid_wall['price'] if bid_wall else best_bid
        max_bid_size_usd = bid_wall['size_usd'] if bid_wall else 0
        dist_bid_wall = (mid_price - max_bid_price) / mid_price
        avg_bid_size = np.mean([price * qty for price, qty in sorted_bids[:20]]) + 1e-9
        bid_wall_ratio = max_bid_size_usd / avg_bid_size
        
        # Variables de Muro Ask
        max_ask_price = ask_wall['price'] if ask_wall else best_ask
        max_ask_size_usd = ask_wall['size_usd'] if ask_wall else 0
        dist_ask_wall = (max_ask_price - mid_price) / mid_price
        avg_ask_size = np.mean([price * qty for price, qty in sorted_asks[:20]]) + 1e-9
        ask_wall_ratio = max_ask_size_usd / avg_ask_size
        
        # 3. Spread y Deltas
        spread = (best_ask - best_bid) / mid_price
        delta_imbalance = imbalance - self.prev_imbalance
        delta_spread = spread - self.prev_spread
        
        # Actualizamos persistencia
        if max_bid_price == self.last_bid_wall_price and max_bid_price > 0: self.bid_wall_persistence += 1
        else: self.bid_wall_persistence = 0; self.last_bid_wall_price = max_bid_price
            
        if max_ask_price == self.last_ask_wall_price and max_ask_price > 0: self.ask_wall_persistence += 1
        else: self.ask_wall_persistence = 0; self.last_ask_wall_price = max_ask_price
        
        # 4. Oráculo y Posición
        sentiment = SentimentOracle.read_cached_score() / 10.0
        pos_encoded = 1 if self.position == 'LONG' else -1 if self.position == 'SHORT' else 0
        
        # Actualizamos memoria corta
        self.prev_imbalance = imbalance
        self.prev_spread = spread
        
        # Retornamos el vector exacto de 12 features
        return np.array([[
            imbalance, bid_wall_ratio, ask_wall_ratio,
            dist_bid_wall, dist_ask_wall, spread,
            delta_imbalance, delta_spread,
            self.bid_wall_persistence, self.ask_wall_persistence,
            sentiment, pos_encoded
        ]]), mid_price

    async def on_depth_update(self, feed):
        """
        Se ejecuta cada vez que el WebSocket manda un tick (milisegundos).
        NUNCA debe bloquear. Solo comprime el estado y avisa a la Queue.
        """
        state_data = self._compress_orderbook(feed)
        if state_data is None: return
        
        self.latest_raw_state = state_data
        
        # --- MODO RECOLECTOR ---
        if self.model is None:
            self._record_training_data(state_data)
            # ---> NUEVO: MANDAR TELEMETRÍA CRUDA AL DASHBOARD <---
            self._dump_data_collector_brain(state_data[0][0])
        # -----------------------
        
        if self.state_queue.empty() and not self.is_inferencing:
            await self.state_queue.put(True)

    def _record_training_data(self, state_data):
        """Graba la compresión de 12 features directo en el disco para entrenar después."""
        import csv
        features_array, mid_price = state_data
        
        output_folder = "data/hft_historical/"
        os.makedirs(output_folder, exist_ok=True)
        record_path = os.path.join(output_folder, f"HFT_{self.ticker}_LIVE_RECORDING.csv")
        
        file_exists = os.path.exists(record_path)
        with open(record_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Escribimos los encabezados la primera vez
                headers = [
                    'mid_price', 'imbalance', 'bid_wall_ratio', 'ask_wall_ratio',
                    'dist_bid_wall', 'dist_ask_wall', 'spread',
                    'delta_imbalance', 'delta_spread',
                    'bid_wall_persistence', 'ask_wall_persistence',
                    'sentiment', 'pos_encoded'
                ]
                writer.writerow(headers)
            
            # Guardamos la fila: mid_price seguido de los 12 features de Claude
            row = [round(mid_price, 4)] + [round(float(x), 6) for x in features_array[0]]
            writer.writerow(row)

    async def inference_loop(self):
        """Loop asíncrono que consume el estado más fresco posible."""
        logging.info(f"[{self.bot_id}] Cerebro HFT asíncrono iniciado.")
        
        while True:
            # Esperamos a que on_depth_update nos despierte
            await self.state_queue.get()
            self.is_inferencing = True
            
            # Agarramos siempre el frame más nuevo (saltando el lag)
            state_data = self.latest_raw_state
            if state_data is None or self.model is None:
                self.is_inferencing = False
                continue
                
            features_array, current_mid_price = state_data
            
            # --- 1. SL y TP ESTRICTO DE MICROESTRUCTURA ---
            # HFT no espera velas, chequea el precio cada tick
            if self.position:
                sl_hit = self._check_hft_stops(current_mid_price)
                if sl_hit:
                    self.is_inferencing = False
                    continue
            
            # --- 2. INFERENCIA EN SEGUNDO PLANO ---
            try:
                state_scaled = self.scaler.transform(features_array)
                
                # ---> CORRECCIÓN: Inferencia HFT directa y sin warnings <---
                prediction_tensor = self.model(state_scaled, training=False)
                action_q = prediction_tensor.numpy()
                # -----------------------------------------------------------
                
                action_idx = np.argmax(action_q[0])
                
                # --- NUEVO: CÁLCULO DE CONFIANZA Y APALANCAMIENTO ---
                # Aplicamos Softmax para convertir Q-Values a Probabilidades (0 a 1)
                q_vals = action_q[0]
                exp_q = np.exp(q_vals - np.max(q_vals)) # Estabilizador numérico
                probs = exp_q / exp_q.sum()
                confidence = probs[action_idx]
                
                # Mapeamos la confianza (ej: 0.4 a 1.0) a un apalancamiento de x10 a x50
                # Si está 100% seguro, tira x50. Si está dudando, tira x10.
                raw_lev = confidence * 50
                decided_lev = int(np.clip(round(raw_lev), 10, 50))
                # ----------------------------------------------------
                
                # --- VISIÓN DE RAYOS X ---
                self._dump_brain(action_q[0], features_array[0], action_idx)
                
                # --- EJECUCIÓN ---
                if action_idx != 2: # NO ES HOLD
                    await self._execute_hft_action(action_idx, current_mid_price, decided_lev)
                    
            except Exception as e:
                logging.error(f"[{self.bot_id}] Error en inferencia asíncrona: {e}")
                
            self.is_inferencing = False

    def _check_hft_stops(self, current_price: float) -> bool:
        """Scalper Híbrido: Entra por microestructura, sale por macro-movimiento."""
        sl_pct = 0.003 # Stop Loss de 0.30%
        tp_pct = 0.008 # Take Profit de 0.80%
        
        if self.position == 'LONG':
            if current_price <= self.entry_price * (1 - sl_pct):
                self._close_position(current_price, "STOP_LOSS")
                return True
            elif current_price >= self.entry_price * (1 + tp_pct):
                self._close_position(current_price, "TAKE_PROFIT")
                return True
                
        elif self.position == 'SHORT':
            if current_price >= self.entry_price * (1 + sl_pct):
                self._close_position(current_price, "STOP_LOSS")
                return True
            elif current_price <= self.entry_price * (1 - tp_pct):
                self._close_position(current_price, "TAKE_PROFIT")
                return True
        return False

    def _dump_brain(self, q_values, raw_features, action_idx):
        """Telemetría para el Dashboard."""
        brain_dump = {
            "estrategia": "HFT Order Book RL",
            "red_neuronal": {
                "q_values": {
                    "0_LONG": round(float(q_values[0]), 4),
                    "1_SHORT": round(float(q_values[1]), 4),
                    "2_HOLD": round(float(q_values[2]), 4)
                },
                "decision_tomada": ["LONG", "SHORT", "HOLD"][action_idx]
            },
            "inputs_clave_observados": {
                "imbalance_top5": round(float(raw_features[0]), 4),
                "muro_bid_ratio": round(float(raw_features[1]), 2),
                "muro_ask_ratio": round(float(raw_features[2]), 2),
                "persistencia_bid": int(raw_features[8]),
                "persistencia_ask": int(raw_features[9])
            }
        }
        self.logger.log_brain_state(brain_dump)

    async def _execute_hft_action(self, action_idx: int, current_price: float, lev_decided: int):
        # Esta lógica es síncrona internamente, pero envuelta en la corutina
        import time
        timestamp = int(time.time() * 1000)
        
        if action_idx == 0: # LONG
            if self.position == 'SHORT':
                self._close_position(current_price, "REVERSAL")
            if not self.position:
                self._open_position(timestamp, 'LONG', current_price, lev_decided)
                
        elif action_idx == 1: # SHORT
            if self.position == 'LONG':
                self._close_position(current_price, "REVERSAL")
            if not self.position:
                self._open_position(timestamp, 'SHORT', current_price, lev_decided)

    def _close_position(self, exit_price: float, reason: str):
        # Mismo código de cierre que los otros bots...
        import time
        timestamp = int(time.time() * 1000)
        quantity = round(self.position_size_usd / self.entry_price, 3)
        
        result = self.executor.close_position(
            ticker=self.ticker, direction=self.position, quantity=quantity, current_price=exit_price
        )
        if result and result['fill_price'] > 0: exit_price = result['fill_price']

        pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100 if self.position == 'LONG' else ((self.entry_price - exit_price) / self.entry_price) * 100
        pnl_pct_lev = pnl_pct * self.current_leverage
        pnl_usd = (self.position_size_usd * (pnl_pct_lev / 100))
        
        self.cm.register_trade_result(pnl_usd, pnl_pct_lev, atr_safe=False)
        
        state_log = self.cm.get_state()
        state_log.update({
            "timestamp": timestamp, "ticker": self.ticker, "precio_in": self.entry_price, "precio_out": exit_price,
            "resultado_usd": pnl_usd, "rendimiento_pct": pnl_pct_lev, "estado_bot": f"CLOSED_{reason}", "apalancamiento": self.current_leverage
        })
        self.logger.log_trade(state_log)
        logging.info(f"[{self.bot_id}] HFT CIERRE {self.position} ({reason}) | PnL: {pnl_pct_lev:.2f}%")
        self.position = None; self.entry_price = 0.0; self.position_size_usd = 0.0

    def _open_position(self, timestamp: int, direction: str, price: float, lev_decided: int):
        signals = {self.bot_id: direction} 
        size = self.risk_engine.get_final_position_size(self.bot_id, direction, signals)
        if size <= 0: return
            
        result = self.executor.open_position(
            ticker=self.ticker, direction=direction, size_usd=size, 
            leverage=lev_decided, current_price=price
        )
        if result is None: return
            
        self.position = direction
        self.entry_price = result['fill_price']
        self.position_size_usd = size
        self.current_leverage = lev_decided # <--- Guardamos la palanca decidida
        
        state_log = self.cm.get_state()
        state_log.update({
            "timestamp": timestamp, "ticker": self.ticker, "precio_in": self.entry_price,
            "estado_bot": f"OPEN_{direction}",
            "apalancamiento": self.current_leverage
        })
        self.logger.log_trade(state_log)
        logging.info(f"[{self.bot_id}] HFT ABIERTO {direction} en {self.entry_price} | Apalancamiento IA: x{lev_decided} | Size: ${size:.2f}")

    def _dump_data_collector_brain(self, raw_features):
        """Telemetría exclusiva para cuando el bot está juntando datos sin IA."""
        brain_dump = {
            "estrategia": "HFT Order Book - MODO ASPIRADORA",
            "estado_interno": {
                "status": "🔴 GRABANDO DATOS PARA ENTRENAMIENTO",
                "ia_cargada": False,
                "decision_latente": "HOLD (Solo Lectura)"
            },
            "inputs_clave_observados": {
                "imbalance_top5": round(float(raw_features[0]), 4),
                "muro_bid_ratio": round(float(raw_features[1]), 2),
                "muro_ask_ratio": round(float(raw_features[2]), 2),
                "persistencia_bid": int(raw_features[8]),
                "persistencia_ask": int(raw_features[9])
            }
        }
        self.logger.log_brain_state(brain_dump)
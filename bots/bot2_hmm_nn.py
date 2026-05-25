import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, List
from core.order_executor import OrderExecutor

# Silenciar los warnings de TensorFlow en la consola
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

from core.capital_manager import CapitalManager
from core.logger import BotLogger
from core.risk_engine import RiskEngine
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Bot2_Sequential:
    """
    Instancia ejecutora del Bot 2 (HMM + LSTM).
    Consume los modelos offline pre-entrenados para filtrar señales técnicas.
    """
    
    def __init__(self, bot_id: str, ticker: str, capital_manager: CapitalManager, 
                 logger: BotLogger, risk_engine: RiskEngine, executor: OrderExecutor, models_folder="models/"):
        self.bot_id = bot_id
        self.ticker = ticker.upper()
        self.cm = capital_manager
        self.logger = logger
        self.risk_engine = risk_engine
        self.executor = executor
        
        self.seq_length = 60
        self.confidence_threshold = 0.54 # El modelo debe tener 70%+ de seguridad para operar
        
        # Estado de Paper Trading
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.current_leverage = 3
        self.peak_price = 0.0

        # --- CONFIGURACIÓN DE DEFENSA Y REVERSAL ---
        self.reversal_mode = 3              # 1: Flip | 2: Solo Salir | 3: Ignorar
        self.use_4h_cooldown = True         # Toggle para prender/apagar el Muro de 4hs
        self.sl_cooldown_start_ts = 0       # Timer del Muro de 4hs
        self.cooldown_signals_remaining = 0 # Contador de señales bloqueadas (Estilo Bot 3)

        # --- PARÁMETROS DINÁMICOS ---
        self.hard_sl_pct = 0.015       # -1.5% SL inicial
        self.activation_pct = 0.020    # A partir del +2% de ganancia, se activa el modo persecución
        self.trailing_dist_pct = 0.015 # Le da 1.5% de respiro desde el pico máximo alcanzado

        self._load_models(models_folder)
        self._rehydrate_state()

    def _load_models(self, models_folder: str):
        """Carga los modelos pre-entrenados a la memoria RAM."""
        hmm_path = os.path.join(models_folder, 'hmm_bot2.pkl')
        scaler_path = os.path.join(models_folder, 'scaler_bot2.pkl')
        lstm_path = os.path.join(models_folder, 'lstm_bot2.keras')
        
        try:
            logging.info(f"[{self.bot_id}] Cargando cerebros de IA desde {models_folder}...")
            self.hmm_model = joblib.load(hmm_path)
            self.scaler = joblib.load(scaler_path)
            self.lstm_model = load_model(lstm_path)
            logging.info(f"[{self.bot_id}] Modelos HMM y LSTM cargados exitosamente.")
        except Exception as e:
            logging.critical(f"[{self.bot_id}] ERROR FATAL cargando modelos. ¿Corriste el script de entrenamiento? Detalle: {e}")
            raise SystemExit(1)

    def _rehydrate_state(self):
        """Recupera el estado tras un reinicio."""
        last_state = self.logger.get_last_state()
        if last_state and last_state.get("estado_bot") in ["OPEN_LONG", "OPEN_SHORT"]:
            self.position = last_state["estado_bot"].split("_")[1]
            self.entry_price = last_state["precio_in"]
            self.position_size_usd = last_state["capital_bloque"] * self.cm.risk_pct
            self.current_leverage = last_state["apalancamiento"]
            logging.info(f"[{self.bot_id}] Rehidratado: {self.position} en {self.entry_price}")

    def _check_trailing_stops(self, candle: Dict[str, Any], current_state=None) -> bool:
        """
        Stop Loss inicial duro, que se transforma en Trailing Stop al superar el umbral.
        """
        if not self.position:
            return False
            
        # --- PARÁMETROS DINÁMICOS ---
        hard_sl_pct = self.hard_sl_pct       # -1.5% SL inicial
        activation_pct = self.activation_pct    # A partir del +2% de ganancia, se activa el modo persecución
        trailing_dist_pct = self.trailing_dist_pct # Le da 1.5% de respiro desde el pico máximo alcanzado
        
        if self.position == 'LONG':
            if candle['high'] > self.peak_price:
                self.peak_price = candle['high']
                
            peak_profit_pct = (self.peak_price - self.entry_price) / self.entry_price
            
            if peak_profit_pct >= activation_pct:
                stop_price = self.peak_price * (1 - trailing_dist_pct)
                reason = "TRAILING_STOP"
            else:
                stop_price = self.entry_price * (1 - hard_sl_pct)
                reason = "STOP_LOSS"
                
            if candle['low'] <= stop_price:
                if current_state is not None:
                    self._close_position(candle['timestamp'], stop_price, reason, current_state)
                else:
                    self._close_position(candle['timestamp'], stop_price, reason)
                return True
                
        elif self.position == 'SHORT':
            if self.peak_price == 0.0 or candle['low'] < self.peak_price:
                self.peak_price = candle['low']
                
            peak_profit_pct = (self.entry_price - self.peak_price) / self.entry_price
            
            if peak_profit_pct >= activation_pct:
                stop_price = self.peak_price * (1 + trailing_dist_pct)
                reason = "TRAILING_STOP"
            else:
                stop_price = self.entry_price * (1 + hard_sl_pct)
                reason = "STOP_LOSS"
                
            if candle['high'] >= stop_price:
                if current_state is not None:
                    self._close_position(candle['timestamp'], stop_price, reason, current_state)
                else:
                    self._close_position(candle['timestamp'], stop_price, reason)
                return True
                
        return False

    def _close_position(self, timestamp: int, exit_price: float, reason: str):
        # 1. Llamada al OrderExecutor
        quantity = round(self.position_size_usd / self.entry_price, 3)
        result = self.executor.close_position(
            ticker=self.ticker,
            direction=self.position,
            quantity=quantity,
            current_price=exit_price
        )
        
        if result is None:
            logging.error(f"[{self.bot_id}] Cierre rechazado por el executor. Registrando SL de seguridad interno.")
        elif result and result['fill_price'] > 0:
            exit_price = result['fill_price']

        # 2. Cálculo de PnL
        if self.position == 'LONG':
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100
            
        pnl_pct_leveraged = pnl_pct * self.current_leverage
        pnl_usd = (self.position_size_usd * (pnl_pct_leveraged / 100))
        
        self.cm.register_trade_result(pnl_usd, pnl_pct_leveraged, atr_safe=True)
        
        state_log = self.cm.get_state()
        state_log.update({
            "timestamp": timestamp,
            "ticker": self.ticker,
            "precio_in": self.entry_price,
            "precio_out": exit_price,
            "resultado_usd": pnl_usd,
            "rendimiento_pct": pnl_pct_leveraged,
            "estado_bot": f"CLOSED_{reason}",
            "apalancamiento": self.current_leverage
        })
        self.logger.log_trade(state_log)
        logging.info(f"[{self.bot_id}] Cierre {self.position} ({reason}) | PnL: {pnl_pct_leveraged:.2f}%")

        # --- NUEVO: ACTIVAR COOLDOWN POR SL ---
        if reason == "STOP_LOSS":
            self.sl_cooldown_start_ts = timestamp
            logging.warning(f"[{self.bot_id}] 🧊 Cooldown de 4h iniciado por SL en {exit_price}")
        # --------------------------------------
        
        # Limpieza total de variables de estado
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.peak_price = 0.0

    def _open_position(self, timestamp: int, direction: str, price: float, ai_prob: float):
        signals = {self.bot_id: direction} 
        size = self.risk_engine.get_final_position_size(self.bot_id, direction, signals)
        
        if size <= 0: return
            
        # 1. Llamada al OrderExecutor
        result = self.executor.open_position(
            ticker=self.ticker,
            direction=direction,
            size_usd=size,
            leverage=self.cm.leverage,
            current_price=price
        )
        
        if result is None:
            logging.error(f"[{self.bot_id}] Orden rechazada por el executor. Abortando entrada.")
            return
            
        # 2. Asignación del precio real de la operación
        actual_entry_price = result['fill_price']
            
        self.position = direction
        self.entry_price = actual_entry_price
        self.peak_price = actual_entry_price
        self.position_size_usd = size
        self.current_leverage = self.cm.leverage
        
        state_log = self.cm.get_state()
        state_log.update({
            "timestamp": timestamp,
            "ticker": self.ticker,
            "precio_in": self.entry_price,
            "estado_bot": f"OPEN_{direction}",
            "apalancamiento": self.current_leverage,
            "ai_confidence": round(float(ai_prob), 4)
        })
        self.logger.log_trade(state_log)
        logging.info(f"[{self.bot_id}] ABIERTO {direction} en {actual_entry_price} | Confianza IA: {ai_prob:.2%} | Size: ${size:.2f} | Lev: x{self.current_leverage}")

    async def on_candle(self, candle: Dict[str, Any], history: List[Dict[str, Any]]):
        if candle['ticker'] != self.ticker: return

        if self.position:
            if self._check_trailing_stops(candle): return

        df = pd.DataFrame(history)
        
        if len(df) < 110: 
            logging.warning(f"[{self.bot_id}] Historial incompleto: {len(df)}/110. Esperando...")
            return
        
        # 1. Feature Engineering IDÉNTICO al entrenamiento
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=20).std()
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
        df['tech_signal'] = (df['ema_fast'] > df['ema_slow']).astype(int)
        
        # Extraemos solo las filas útiles y cortamos la longitud exacta para el LSTM (60 velas)
        df_clean = df.dropna().copy()
        if len(df_clean) < self.seq_length: return
        
        recent_df = df_clean.iloc[-self.seq_length:].copy()
        
        # 2. Inferencia del HMM (Contexto de mercado)
        hmm_features = recent_df[['log_return', 'volatility', 'hl_spread']].values
        hmm_states = self.hmm_model.predict(hmm_features)
        recent_df['hmm_state'] = hmm_states
        
        # 3. Preparar variables para el LSTM
        scaled_features = self.scaler.transform(recent_df[['log_return', 'volatility', 'hl_spread', 'tech_signal']])
        final_features = np.column_stack((scaled_features, recent_df['hmm_state'].values))
        
        # Reshape a formato batch 3D: (1 sample, 60 timesteps, N features)
        X_live = final_features.reshape(1, self.seq_length, final_features.shape[1])
        
        # 4. Inferencia del LSTM (Optimizada / Sin Warnings)
        prediction_tensor = self.lstm_model(X_live, training=False)
        lstm_prob = float(prediction_tensor[0][0])
        current_signal = recent_df['tech_signal'].iloc[-1]

        # --- NUEVO: TELEMETRÍA CEREBRAL (BRAIN DUMP) ---
        brain_dump = {
            "timestamp_local": candle['timestamp'],
            "estrategia": "HMM + Neural Network",
            "estado_interno": {
                "confianza_ia_cruda": round(float(lstm_prob), 4),
                "decision_latente": "LONG" if current_signal == 1 else "SHORT" if current_signal == 0 else "HOLD",
                "cruza_umbral_confianza": bool(lstm_prob >= self.confidence_threshold)
            },
            "inputs_clave_observados": {
                "regimen_hmm": int(recent_df['hmm_state'].iloc[-1]),
                "tech_signal": int(current_signal),
                "precio_actual": float(candle['close'])
            }
        }
        self.logger.log_brain_state(brain_dump)
        # ----------------------------------------------
        
        # =========================================================
        # 🛡️ DEFENSA 1: COOLDOWN 4H POR SL (TOGGLEABLE)
        # =========================================================
        ms_4h = 4 * 60 * 60 * 1000
        en_cooldown_4h = False
        
        if self.use_4h_cooldown and self.sl_cooldown_start_ts > 0:
            if (candle['timestamp'] - self.sl_cooldown_start_ts) < ms_4h:
                en_cooldown_4h = True
            else:
                self.sl_cooldown_start_ts = 0
                logging.info(f"[{self.bot_id}] ✅ Cooldown temporal de 4h finalizado.")

        # =========================================================
        # 🔄 LÓGICA DE REVERSAL (MODOS 1, 2, 3)
        # =========================================================
        # El reversal se gatilla solo por la señal técnica (EMA cross), sin esperar a la IA
        if self.position:
            reversal_trigger = (self.position == 'LONG' and current_signal == 0) or \
                               (self.position == 'SHORT' and current_signal == 1)

            if reversal_trigger:
                if self.reversal_mode == 1:
                    self._close_position(candle['timestamp'], candle['close'], "REVERSAL_FLIP")
                elif self.reversal_mode == 2:
                    self._close_position(candle['timestamp'], candle['close'], "REVERSAL_EXIT_ONLY")
                # Modo 3 ignora y sigue de largo.

        # =========================================================
        # 🛡️ DEFENSA 2: BLOQUEO POR CONTEO DE SEÑALES (ESTILO BOT 3)
        # =========================================================
        # 1. Le preguntamos a la IA si quiere entrar
        entry_signal = None
        if lstm_prob >= self.confidence_threshold:
            entry_signal = 'LONG' if current_signal == 1 else 'SHORT'

        # 2. Aplicamos tu lógica de conteo de señales
        if entry_signal is not None:
            # A. Gatillo: 4 pérdidas consecutivas
            if self.cm.consecutive_sl >= 4 and self.cooldown_signals_remaining == 0:
                logging.warning(f"[{self.bot_id}] 🧊 MODO COOLDOWN: 4 SL detectados. Bloqueando 5 señales de la IA.")
                self.cooldown_signals_remaining = 5
                
                # RESET ESTRATÉGICO: Bajamos a 3 para el "Trade Sonda"
                self.cm.consecutive_sl = 3
                entry_signal = None 
                
            # B. Descontamos el bloqueo
            elif self.cooldown_signals_remaining > 0:
                self.cooldown_signals_remaining -= 1
                logging.info(f"[{self.bot_id}] 🛑 Señal IA bloqueada por historial. Restan: {self.cooldown_signals_remaining}")
                entry_signal = None

        # =========================================================
        # 🚀 APERTURA DE POSICIONES
        # =========================================================
        if not self.position and entry_signal and not en_cooldown_4h:
            self._open_position(candle['timestamp'], entry_signal, candle['close'], lstm_prob)
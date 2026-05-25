import os
import shutil
import numpy as np
import pandas as pd
import joblib
import logging
import pandas_ta as ta
from typing import Dict, Any, List
import random
from collections import deque
from core.order_executor import OrderExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

from core.capital_manager import CapitalManager
from core.logger import BotLogger
from core.risk_engine import RiskEngine
from oracles.sentiment_daemon import SentimentOracle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ReplayBuffer:
    def __init__(self, maxlen=50000):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class Bot4_RL:
    """
    Instancia ejecutora del Bot 4.
    Agente de Reinforcement Learning con control de Apalancamiento, 
    Online Learning en tiempo real y Rollback de seguridad.
    """
    def __init__(self, bot_id: str, ticker: str, capital_manager: CapitalManager, 
                 logger: BotLogger, risk_engine: RiskEngine, executor: OrderExecutor, models_folder="models/"):
        self.bot_id = bot_id
        self.ticker = ticker.upper()
        self.symbol = ticker.upper()
        self.cm = capital_manager
        self.logger = logger
        self.risk_engine = risk_engine
        self.models_folder = models_folder
        self.executor = executor
        
        # Estado de la Posición
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.current_leverage = 1
        
        # Estado RL para Online Learning
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        
        # Rollback Tracker
        self.recent_rewards = deque(maxlen=20)
        self.trades_since_backup = 0
        
        self._load_models()
        self._rehydrate_state()

    def _load_models(self):
        try:
            self.model_path = os.path.join(self.models_folder, 'rl_bot4.keras')
            self.backup_path = os.path.join(self.models_folder, 'rl_bot4_BACKUP.keras')
            
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(os.path.join(self.models_folder, 'rl_scaler_bot4.pkl'))
            
            # --- SISTEMA DE MEMORIA PERSISTENTE (MULTI-TICKER) ---
            self.live_buffer_path = os.path.join(self.models_folder, f'rl_buffer_LIVE_{self.symbol}.pkl')
            base_buffer_path = os.path.join(self.models_folder, 'replay_buffer.pkl')
            
            if os.path.exists(self.live_buffer_path):
                self.replay_buffer = joblib.load(self.live_buffer_path)
                logging.info(f"[{self.bot_id}] Memoria viva recuperada para {self.symbol}: {len(self.replay_buffer)} recuerdos.")
            elif os.path.exists(base_buffer_path):
                self.replay_buffer = joblib.load(base_buffer_path)
                logging.info(f"[{self.bot_id}] Memoria BASE recuperada: {len(self.replay_buffer)} recuerdos listos para aprender.")
            else:
                self.replay_buffer = ReplayBuffer(maxlen=50000)
                logging.info(f"[{self.bot_id}] Buffer en RAM iniciado en blanco para {self.symbol}.")
                
            logging.info(f"[{self.bot_id}] Modelos RL cargados exitosamente.")
        except Exception as e:
            logging.critical(f"[{self.bot_id}] Error cargando entorno RL: {e}")
            raise SystemExit(1)

    def _rehydrate_state(self):
        last_state = self.logger.get_last_state()
        if last_state and last_state.get("estado_bot") in ["OPEN_LONG", "OPEN_SHORT"]:
            self.position = last_state["estado_bot"].split("_")[1]
            self.entry_price = last_state["precio_in"]
            self.position_size_usd = last_state["capital_bloque"] * self.cm.risk_pct
            self.current_leverage = last_state["apalancamiento"]
            self.last_reward = last_state.get("rl_reward_last", 0.0)

    def _check_survival_stop(self, candle: Dict[str, Any], current_state: np.ndarray) -> bool:
        """
        Escudo de Supervivencia (Hard Stop).
        No hay Take Profit ni Trailing Stop. La IA decide cuándo salir con un Reversal.
        """
        if not self.position: return False
            
        hard_sl_pct = 0.020 # Un 2% de aire para que la IA pueda surfear volatilidad sin que el sistema la corte
        
        if self.position == 'LONG':
            stop_price = self.entry_price * (1 - hard_sl_pct)
            if candle['low'] <= stop_price:
                self._close_position(candle['timestamp'], stop_price, "SURVIVAL_STOP", current_state)
                return True
                
        elif self.position == 'SHORT':
            stop_price = self.entry_price * (1 + hard_sl_pct)
            if candle['high'] >= stop_price:
                self._close_position(candle['timestamp'], stop_price, "SURVIVAL_STOP", current_state)
                return True
                
        return False

    def _on_trade_closed(self, reward: float, current_state: np.ndarray):
        """Bloque de Online Learning e Histéresis de Modelos."""
        final_reward = -2.0 if reward <= -1.5 else reward
        self.last_reward = final_reward
        self.recent_rewards.append(final_reward)
        self.trades_since_backup += 1
        
        if self.last_state is not None and self.last_action is not None and current_state is not None:
            self.replay_buffer.add(self.last_state, self.last_action, final_reward, current_state, done=True)
            
        if len(self.replay_buffer) >= 500:
            batch = self.replay_buffer.sample(64)
            states = np.array([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            next_states = np.array([x[3] for x in batch])
            
            next_q, _ = self.model.predict(next_states, verbose=0)
            target_q = rewards + 0.95 * np.max(next_q, axis=1)
            
            current_q, current_lev = self.model.predict(states, verbose=0)
            lev_learning_rate = 1.5
            
            for i, action in enumerate(actions):
                if action != 2:
                    current_q[i][action] = target_q[i]
                    current_lev[i][0] = current_lev[i][0] + (rewards[i] * lev_learning_rate)
                    
            current_lev = np.clip(current_lev, 1, 25)
            
            self.model.fit(states, [current_q, current_lev], epochs=1, verbose=0)
            self.model.save(self.model_path)
            
            if self.trades_since_backup % 10 == 0:
                joblib.dump(self.replay_buffer, os.path.join(self.models_folder, 'replay_buffer.pkl'))

        if len(self.recent_rewards) == 20:
            avg_reward = sum(self.recent_rewards) / 20
            if avg_reward < -1.0:
                logging.warning(f"[{self.bot_id}] 🚨 PERFORMANCE DEGRADADA (Avg PnL: {avg_reward:.2f}). EJECUTANDO ROLLBACK.")
                shutil.copy(self.backup_path, self.model_path)
                self.model = load_model(self.model_path)
                self.recent_rewards.clear()
                self.trades_since_backup = 0
                
        if self.trades_since_backup >= 100:
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            if avg_reward > 0:
                logging.info(f"[{self.bot_id}] ✅ Checkpoint: 100 Trades rentables. Actualizando BACKUP.")
                shutil.copy(self.model_path, self.backup_path)
            self.trades_since_backup = 0
            self.recent_rewards.clear()

    def _close_position(self, timestamp: int, exit_price: float, reason: str, current_state: np.ndarray):
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

        if self.position == 'LONG':
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100
            
        pnl_pct_leveraged = pnl_pct * self.current_leverage
        pnl_usd = (self.position_size_usd * (pnl_pct_leveraged / 100))
        
        self.cm.register_trade_result(pnl_usd, pnl_pct_leveraged, atr_safe=True)
        self._on_trade_closed(pnl_pct_leveraged, current_state)
        
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
        logging.info(f"[{self.bot_id}] Cierre {self.position} ({reason}) | PnL: {pnl_pct_leveraged:.2f}% | Lev: x{self.current_leverage}")
        
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.current_leverage = 1

    def _open_position(self, timestamp: int, direction: str, price: float, lev_decided: int, current_state: np.ndarray, action_idx: int):
        signals = {self.bot_id: direction} 
        size = self.risk_engine.get_final_position_size(self.bot_id, direction, signals)
        
        if size <= 0: 
            logging.warning(f"[{self.bot_id}] 🛑 Trade Abortado: Risk Engine bloqueó la orden (Size <= 0).")
            return
            
        result = self.executor.open_position(
            ticker=self.ticker,
            direction=direction,
            size_usd=size,
            leverage=lev_decided,
            current_price=price
        )
        
        if result is None:
            logging.error(f"[{self.bot_id}] Orden rechazada por el executor. Abortando entrada.")
            return
            
        actual_entry_price = result['fill_price']
            
        self.position = direction
        self.entry_price = actual_entry_price
        self.position_size_usd = size
        self.current_leverage = lev_decided
        
        self.last_state = current_state
        self.last_action = action_idx
        
        state_log = self.cm.get_state()
        state_log.update({
            "timestamp": timestamp,
            "ticker": self.ticker,
            "precio_in": self.entry_price,
            "estado_bot": f"OPEN_{direction}",
            "apalancamiento": self.current_leverage,
            "rl_action": action_idx,
            "rl_leverage_decided": lev_decided,
            "rl_reward_last": round(float(self.last_reward), 4)
        })
        self.logger.log_trade(state_log)
        logging.info(f"[{self.bot_id}] ABIERTO {direction} en {actual_entry_price} | Apalancamiento IA: x{lev_decided} | Size: ${size:.2f}")

    async def on_candle(self, candle: Dict[str, Any], history: List[Dict[str, Any]]):
        if candle['ticker'] != self.ticker: return

        df = pd.DataFrame(history)
        if len(df) < 210: return
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        ema_9 = ta.ema(df['close'], length=9)
        ema_21 = ta.ema(df['close'], length=21)
        df['ema_9_dist'] = (df['close'] - ema_9) / df['close']
        df['ema_21_dist'] = (df['close'] - ema_21) / df['close']
        df['ema_cross_signal'] = (ema_9 > ema_21).astype(int)
        
        sma_50 = ta.sma(df['close'], length=50)
        sma_200 = ta.sma(df['close'], length=200)
        df['sma_50_dist'] = (df['close'] - sma_50) / df['close']
        df['sma_200_dist'] = (df['close'] - sma_200) / df['close']
        
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        upper_bb = sma20 + (std20 * 2)
        lower_bb = sma20 - (std20 * 2)
        df['bb_width'] = (upper_bb - lower_bb) / sma20
        df['bb_p'] = (df['close'] - lower_bb) / (upper_bb - lower_bb + 1e-9)
        
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['atr_14_pct'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']
        
        vol_mean = df['volume'].rolling(50).mean()
        vol_std = df['volume'].rolling(50).std()
        df['volume_normalized'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)
        
        last_row = df.iloc[-1].copy()
        if last_row.isna().any(): return
        
        sentiment = SentimentOracle.read_cached_score()
        sentiment = float(sentiment) if sentiment is not None else 5.0 # <-- Escudo anti-vacíos
        
        features_array = np.array([[
            last_row['log_return'], last_row['volatility_20'], last_row['hl_spread'],
            last_row['ema_9_dist'], last_row['ema_21_dist'], last_row['ema_cross_signal'],
            last_row['sma_50_dist'], last_row['sma_200_dist'],
            last_row['bb_width'], last_row['bb_p'],
            last_row['rsi_14'], last_row['atr_14_pct'], last_row['volume_normalized'],
            sentiment
        ]])
        
        state_scaled = self.scaler.transform(features_array)
        current_state_flat = state_scaled[0]

        if self.position:
            if self._check_survival_stop(candle, current_state_flat): return

        prediccion = self.model(state_scaled, training=False)
        action_q = prediccion[0].numpy()
        lev_pred = prediccion[1].numpy()
        
        # 1. Primero leemos la decisión real de la IA
        action_idx = int(np.argmax(action_q[0]))
        
        # --- NUEVO: CURIOSIDAD INTELIGENTE ---
        tasa_curiosidad = 0.005  # 0.5% de curiosidad
        
        # 2. SOLO tiramos los dados si la IA decidió hacer HOLD (2) y NO estamos adentro de un trade
        if action_idx == 2 and not self.position:
            if random.random() < tasa_curiosidad:
                action_idx = random.choice([0, 1])
                logging.info(f"[{self.bot_id}] 🎲 MODO CURIOSIDAD: La IA quería hacer HOLD por miedo, pero la obligamos a explorar.")
        # -------------------------------------
        
        raw_lev = lev_pred[0][0]
        decided_lev = int(np.clip(round(raw_lev), 1, 25))

        # --- TELEMETRÍA CEREBRAL (BRAIN DUMP) ---
        brain_dump = {
            "timestamp_local": candle['timestamp'],
            "red_neuronal": {
                "q_values": {
                    "0_LONG": round(float(action_q[0][0]), 4),
                    "1_SHORT": round(float(action_q[0][1]), 4),
                    "2_HOLD": round(float(action_q[0][2]), 4)
                },
                "apalancamiento_crudo": round(float(raw_lev), 4),
                "decision_tomada": ["LONG", "SHORT", "HOLD"][action_idx]
            },
            "inputs_clave_observados": {
                "rsi_14": round(float(last_row['rsi_14']), 2),
                "atr_normalizado": round(float(last_row['atr_14_pct']), 5),
                "oraculo_sentiment": float(sentiment)
            }
        }
        self.logger.log_brain_state(brain_dump)
        # ----------------------------------------

        if action_idx == 2: # HOLD
            return
            
        elif action_idx == 0: # LONG
            if self.position == 'SHORT':
                self._close_position(candle['timestamp'], candle['close'], "REVERSAL", current_state_flat)
            if not self.position:
                self._open_position(candle['timestamp'], 'LONG', candle['close'], decided_lev, current_state_flat, action_idx)
                
        elif action_idx == 1: # SHORT
            if self.position == 'LONG':
                self._close_position(candle['timestamp'], candle['close'], "REVERSAL", current_state_flat)
            if not self.position:
                self._open_position(candle['timestamp'], 'SHORT', candle['close'], decided_lev, current_state_flat, action_idx)
    
    def save_state(self):
        """Vuelca la libreta de la RAM al disco duro (Auto-Guardado)."""
        try:
            if len(self.replay_buffer) > 0:
                joblib.dump(self.replay_buffer, self.live_buffer_path)
                logging.info(f"[{self.bot_id}] Auto-Guardado EXITOSO: {len(self.replay_buffer)} recuerdos a salvo en disco.")
        except Exception as e:
            logging.error(f"[{self.bot_id}] Error al guardar la memoria: {e}")
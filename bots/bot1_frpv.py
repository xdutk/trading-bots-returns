import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
import datetime
import logging
from typing import Dict, Any, List
from core.order_executor import OrderExecutor

# Silenciador de advertencias de Pandas
pd.set_option('future.no_silent_downcasting', True)

# Dependencias internas de QuantProtocol
from core.capital_manager import CapitalManager
from core.logger import BotLogger
from core.risk_engine import RiskEngine
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Public_MathEngine:
    """Motor matemático genérico (Bandas de Bollinger + RSI) para versión Open Source."""
    def __init__(self, bb_len=20, bb_std=2.0, rsi_len=14):
        self.bb_len = bb_len
        self.bb_std = bb_std
        self.rsi_len = rsi_len

    def generate_signals(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        
        # --- 1. INDICADORES CLÁSICOS ---
        bbands = ta.bbands(df['close'], length=self.bb_len, std=self.bb_std)
        if bbands is not None:
            df = df.join(bbands)
            
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_len)
        
        # Nombres de columnas dinámicos de pandas_ta
        bbl_col = f"BBL_{self.bb_len}_{self.bb_std}"
        bbu_col = f"BBU_{self.bb_len}_{self.bb_std}"
        
        # --- 2. LÓGICA DE SEÑALES ---
        if bbl_col in df.columns:
            df['buy_signal'] = (df['close'] < df[bbl_col]) & (df['rsi'] < 30)
            df['sell_signal'] = (df['close'] > df[bbu_col]) & (df['rsi'] > 70)
        else:
            df['buy_signal'] = False
            df['sell_signal'] = False

        # --- 3. VARIABLES DE COMPATIBILIDAD ---
        df['converging_buy_signal'] = df['buy_signal']
        df['converging_sell_signal'] = df['sell_signal']
        
        df['overext_above_1h'] = df['rsi'] > 80
        df['overext_below_1h'] = df['rsi'] < 20
        
        return df

class Public_ContextFilter:
    """Filtro de contexto público genérico (Tendencia simple SMA 50)."""
    def __init__(self):
        pass

    def calculate_zscore(self, target_ticker: str, all_buffers: dict) -> float:
        """
        Devuelve un Z-Score simulado para que la lógica de main.py siga funcionando.
        > 0 permite LONGs, < 0 permite SHORTs.
        """
        if target_ticker not in all_buffers or len(all_buffers[target_ticker]) < 50:
            return 0.0
            
        df = pd.DataFrame(all_buffers[target_ticker])
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        precio_actual = df['close'].iloc[-1]
        
        return 1.0 if precio_actual > sma_50 else -1.0


class Bot1_FRPV:
    """
    Instancia ejecutora del Bot 1.
    Maneja el estado de la posición, ejecuta las órdenes, aplica el Stop Loss 
    y reporta resultados al CapitalManager y Logger.
    """
    
    def __init__(self, bot_id: str, ticker: str, capital_manager: CapitalManager, 
                 logger: BotLogger, risk_engine: RiskEngine, data_feed, executor: OrderExecutor):
        self.bot_id = bot_id
        self.ticker = ticker.upper()
        self.cm = capital_manager
        self.logger = logger
        self.risk_engine = risk_engine
        self.data_feed = data_feed
        self.executor = executor
        
        self.reversal_mode = 3        
        self.sl_cooldown_start_ts = 0  
        self.cooldown_start_ts = 0  
        
        # Instanciación de motores genéricos Open Source
        self.math_engine = Public_MathEngine()
        self.context_filter = Public_ContextFilter()
        
        # Estado de la operación
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.current_leverage = 3
        self.peak_price = 0.0

        self.hard_sl_pct = 0.0125       
        self.activation_pct = 0.060     
        self.trailing_dist_pct = 0.030   
        
        self._rehydrate_state()

    def _rehydrate_state(self):
        """Intenta recuperar una operación abierta tras un crash."""
        last_state = self.logger.get_last_state()
        if last_state and last_state.get("estado_bot") in ["OPEN_LONG", "OPEN_SHORT"]:
            self.position = last_state["estado_bot"].split("_")[1]
            self.entry_price = last_state["precio_in"]
            self.position_size_usd = last_state["capital_bloque"] * self.cm.risk_pct
            self.current_leverage = last_state["apalancamiento"]
            logging.info(f"[{self.bot_id}] Rehidratado con posición {self.position} abierta en {self.entry_price} | Size reconstruido: ${self.position_size_usd:.2f}")

    def _check_trailing_stops(self, candle: Dict[str, Any], current_state=None) -> bool:
        """Stop Loss inicial duro, que se transforma en Trailing Stop al superar el umbral."""
        if not self.position:
            return False
            
        hard_sl_pct = self.hard_sl_pct       
        activation_pct = self.activation_pct    
        trailing_dist_pct = self.trailing_dist_pct 
        
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
        """Ejecuta el cierre, calcula el PnL, actualiza el CapitalManager y loguea el trade."""
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

        if reason == "STOP_LOSS":
            self.sl_cooldown_start_ts = timestamp
            logging.warning(f"[{self.bot_id}] 🧊 Cooldown de 4h iniciado por SL en {exit_price}")
        
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
        
        logging.info(f"[{self.bot_id}] Posición {self.position} cerrada. Razón: {reason} | PnL: {pnl_pct_leveraged:.2f}% (${pnl_usd:.2f})")
        
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.peak_price = 0.0

    def _open_position(self, timestamp: int, direction: str, price: float, overext_above: bool = False, overext_below: bool = False):
        """Abre una nueva orden de mercado consultando al RiskEngine y OrderExecutor."""
        signals = {self.bot_id: direction} 
        size = self.risk_engine.get_final_position_size(self.bot_id, direction, signals)
        
        if size <= 0:
            logging.warning(f"[{self.bot_id}] Señal {direction} bloqueada por RiskEngine (Size 0).")
            return
            
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
            "overext_above": overext_above,
            "overext_below": overext_below
        })
        self.logger.log_trade(state_log)
        logging.info(f"[{self.bot_id}] ABIERTO {direction} en {actual_entry_price} | Size: ${size:.2f} | Lev: x{self.current_leverage}")

    async def on_candle(self, candle: Dict[str, Any], history: List[Dict[str, Any]]):
        """Callback principal. Se ejecuta cada vez que cierra una vela de 5m."""
        if candle['ticker'] != self.ticker:
            return

        if self.position:
            sl_hit = self._check_trailing_stops(candle)
            if sl_hit:
                return
        
        df = pd.DataFrame(history)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        if len(df) < 2410:
            return
            
        df_signals = self.math_engine.generate_signals(df)
        last_row = df_signals.iloc[-1]
        
        buy_signal = last_row['converging_buy_signal']
        sell_signal = last_row['converging_sell_signal']
        
        overext_above = last_row['overext_above_1h']
        overext_below = last_row['overext_below_1h']

        # Telemetría de Caja Blanca para el Dashboard
        brain_dump = {
            "timestamp_local": candle['timestamp'],
            "estrategia": "FRPV (Reversión a la Media - Public)",
            "estado_interno": {
                "converging_buy": bool(buy_signal),
                "converging_sell": bool(sell_signal),
                "decision_latente": "LONG" if overext_below else "SHORT" if overext_above else "HOLD"
            },
            "inputs_clave_observados": {
                "overext_above": bool(overext_above),
                "overext_below": bool(overext_below),
                "precio_actual": float(candle['close'])
            }
        }
        self.logger.log_brain_state(brain_dump)

        # Escudos defensivos
        ms_4h = 4 * 60 * 60 * 1000
        ms_24h = 24 * 60 * 60 * 1000
        en_penitencia = False

        if self.sl_cooldown_start_ts > 0:
            if (candle['timestamp'] - self.sl_cooldown_start_ts) < ms_4h:
                en_penitencia = True
            else:
                self.sl_cooldown_start_ts = 0
                logging.info(f"[{self.bot_id}] ✅ Cooldown de 4h finalizado.")

        if self.cm.consecutive_sl >= 4:
            if self.cooldown_start_ts == 0:
                self.cooldown_start_ts = candle['timestamp']
                logging.warning(f"[{self.bot_id}] 🧊 MODO COOLDOWN 24H: Bloqueo total.")

            if (candle['timestamp'] - self.cooldown_start_ts) < ms_24h:
                en_penitencia = True
            else:
                self.cm.consecutive_sl = 0
                self.cooldown_start_ts = 0
                logging.info(f"[{self.bot_id}] 🛡️ Muro de 24h finalizado.")

        # Lógica de Reversal
        if self.position:
            reversal_trigger = (self.position == 'LONG' and sell_signal) or \
                               (self.position == 'SHORT' and buy_signal)

            if reversal_trigger:
                if self.reversal_mode == 1:
                    self._close_position(candle['timestamp'], candle['close'], "REVERSAL_FLIP")
                elif self.reversal_mode == 2:
                    self._close_position(candle['timestamp'], candle['close'], "REVERSAL_EXIT_ONLY")

        # Apertura controlada
        if not self.position and not en_penitencia:
            if buy_signal or sell_signal:
                z_score = candle.get('zscore_csv')
                
                if z_score is None:
                    z_score = self.context_filter.calculate_zscore(self.ticker, self.data_feed.buffers)
                
                if buy_signal and z_score > 0:
                    self._open_position(candle['timestamp'], 'LONG', candle['close'], overext_above, overext_below)
                elif sell_signal and z_score < 0:
                    self._open_position(candle['timestamp'], 'SHORT', candle['close'], overext_above, overext_below)

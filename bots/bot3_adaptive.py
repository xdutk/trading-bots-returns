import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List
from core.order_executor import OrderExecutor

# Dependencias internas de QuantProtocol
from core.capital_manager import CapitalManager
from core.logger import BotLogger
from core.risk_engine import RiskEngine
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AdaptiveCrossEngine:
    """
    Motor matemático del Bot 3.
    Ajusta dinámicamente el período de las medias móviles según la varianza 
    del mercado y aplica un mecanismo de "Freeze" en mercados laterales.
    """
    def __init__(self, var_window=50, freeze_lookback=20, norm_window=200):
        self.var_window = var_window
        self.freeze_lookback = freeze_lookback
        self.norm_window = norm_window

    def _calc_dynamic_ema(self, series: pd.Series, periods: pd.Series) -> pd.Series:
        """
        Calcula una EMA donde el período cambia en cada vela.
        No se puede usar ewm() de pandas directo, por lo que iteramos la fórmula de suavizado.
        """
        emas = np.zeros(len(series))
        emas[:] = np.nan
        
        first_valid = series.first_valid_index()
        if first_valid is None: 
            return pd.Series(emas, index=series.index)
            
        start_idx = series.index.get_loc(first_valid)
        emas[start_idx] = series.iloc[start_idx]
        
        vals = series.values
        p_vals = periods.values
        
        for i in range(start_idx + 1, len(vals)):
            if np.isnan(p_vals[i]):
                emas[i] = vals[i]
            else:
                # Fórmula de suavizado exponencial: Alpha = 2 / (Periodo + 1)
                alpha = 2.0 / (p_vals[i] + 1.0)
                emas[i] = vals[i] * alpha + emas[i-1] * (1.0 - alpha)
                
        return pd.Series(emas, index=series.index)

    def analyze(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        
        # 1. Retornos Logarítmicos y Varianza
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['var'] = df['log_ret'].rolling(self.var_window).var()
        
        # 2. Normalización de Varianza (0 a 1) respecto a las últimas 200 velas
        rolling_min = df['var'].rolling(self.norm_window).min()
        rolling_max = df['var'].rolling(self.norm_window).max()
        
        denom = rolling_max - rolling_min
        denom = denom.replace(0, 1e-9) # Evitar división por cero
        
        df['var_norm'] = (df['var'] - rolling_min) / denom
        df['var_norm'] = df['var_norm'].clip(0, 1)
        
        # =========================================================
        # 3. Mapeo del Período Rápido con AMORTIGUADOR (Anti-Jitter)
        # =========================================================
        # Calculamos el período crudo y nervioso (Alta Varianza = 5, Baja Varianza = 30)
        raw_fast_period = 30 - (df['var_norm'] * 25)
        
        # MAGIA QUANT: Le aplicamos una EMA de 10 períodos al propio período.
        # Esto mata los latigazos. El bot reacciona a la volatilidad, pero sin histeria.
        smoothed_period = raw_fast_period.ewm(span=10, adjust=False).mean()
        
        df['fast_period'] = smoothed_period.fillna(30).clip(5, 30).round().astype(int)
        df['slow_period'] = df['fast_period'] * 3
        # =========================================================
        
        # 4. Cálculo de EMAs Dinámicas
        df['ema_fast'] = self._calc_dynamic_ema(df['close'], df['fast_period'])
        df['ema_slow'] = self._calc_dynamic_ema(df['close'], df['slow_period'])
        
        # 5. Detección de Cruces
        df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        df['cross'] = (df['trend'].diff() != 0) & (df['trend'].shift(1).notna())
        
        # 6. Mecanismo de Freeze (Chop Filter)
        df['cross_count'] = df['cross'].rolling(self.freeze_lookback).sum()
        
        # Máquina de estados para is_frozen:
        # >= 3 entra en Freeze. <= 1 sale de Freeze. (El ffill mantiene el estado)
        conditions = [df['cross_count'] >= 3, df['cross_count'] <= 1]
        choices = [1, 0]
        df['freeze_signal'] = np.select(conditions, choices, default=np.nan)
        df['is_frozen'] = df['freeze_signal'].ffill().fillna(0).astype(bool)
        
        # 6.5 Cálculo de RSI (Filtro de Pullbacks ganador del laboratorio)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # 7. Extracción Final (Filtro RSI más ajustado)
        last_row = df.iloc[-1]
        raw_signal = None
        entry_signal = None
        
        if last_row['cross']:
            raw_signal = 'LONG' if last_row['trend'] == 1 else 'SHORT'
            
            # Ajustamos de 55/45 a 50/50 para ser más selectivos
            if raw_signal == 'LONG' and last_row['rsi'] < 50:
                entry_signal = 'LONG'
            elif raw_signal == 'SHORT' and last_row['rsi'] > 50:
                entry_signal = 'SHORT'
                
        return {
            'raw_signal': raw_signal,
            'entry_signal': entry_signal,
            'fast_period': int(last_row['fast_period']),
            'slow_period': int(last_row['slow_period']),
            'cross_count': int(last_row['cross_count']) if pd.notna(last_row['cross_count']) else 0,
            'is_frozen': bool(last_row['is_frozen']),
            'rsi_actual': float(last_row['rsi'])
        }

class Bot3_Adaptive:
    """
    Instancia ejecutora del Bot 3.
    Opera cruces de EMAs con períodos dinámicos basados en varianza.
    """
    def __init__(self, bot_id: str, ticker: str, capital_manager: CapitalManager, 
                 logger: BotLogger, risk_engine: RiskEngine, executor: OrderExecutor):
        self.bot_id = bot_id
        self.ticker = ticker.upper()
        self.cm = capital_manager
        self.logger = logger
        self.risk_engine = risk_engine
        self.executor = executor
        
        self.engine = AdaptiveCrossEngine()
        
        # Estado de Paper Trading
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.current_leverage = 3
        self.peak_price = 0.0
        self.entry_timestamp = 0

        # --- NUEVO: ESTADO DE COOLDOWN ---
        self.cooldown_signals_remaining = 0

        # --- PARÁMETROS DINÁMICOS ---
        self.hard_sl_pct = 0.0125       # -1.25% SL inicial
        self.activation_pct = 0.040    # A partir del +4.0% de ganancia, se activa el modo persecución
        self.trailing_dist_pct = 0.015 # Le da 1.5% de respiro desde el pico máximo alcanzado

        # --- CONFIGURACIÓN DE REVERSAL ---
        # 1: Reversal estándar (Cierra y permite abrir el opuesto)
        # 2: Reversal solo cierre (Cierra pero no abre el opuesto en la misma vela)
        # 3: Reversal bloqueado (Ignora señales opuestas si ya hay una posición)
        self.reversal_mode = 2 
        
        self._rehydrate_state()

    def _rehydrate_state(self):
        last_state = self.logger.get_last_state()
        
        if last_state and last_state.get("estado_bot") in ["OPEN_LONG", "OPEN_SHORT"]:
            # 1. Recuperar la dirección
            self.position = last_state["estado_bot"].split("_")[1]
            
            # 2. Casteo explícito a numéricos (Evita el TypeError: int - str)
            self.entry_price = float(last_state["precio_in"])
            self.position_size_usd = float(last_state["capital_bloque"]) * self.cm.risk_pct
            self.current_leverage = int(last_state["apalancamiento"])
            
            # 3. EL FIX: Recuperar el timestamp de entrada para el cálculo de madurez
            # Usamos .get() por seguridad y casteamos a int para poder restar
            self.entry_timestamp = int(last_state.get("timestamp", 0))
            
            logging.info(f"[{self.bot_id}] Rehidratado: {self.position} en {self.entry_price} | T: {self.entry_timestamp}")

    def _check_trailing_stops(self, candle: Dict[str, Any], current_state=None) -> bool:
        """
        Stop Loss inicial duro, que se transforma en Trailing Stop al superar el umbral.
        """
        if not self.position:
            return False
            
        # --- PARÁMETROS DINÁMICOS ---
        hard_sl_pct = self.hard_sl_pct      # -1.25% SL inicial
        activation_pct = self.activation_pct    # A partir del +4.0% de ganancia, se activa el modo persecución
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
        # El PnL porcentual se calcula como: $$PnL_{\%} = \frac{Price_{exit} - Price_{entry}}{Price_{entry}} \times 100$$
        if self.position == 'LONG':
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100
            
        pnl_pct_leveraged = pnl_pct * self.current_leverage
        pnl_usd = (self.position_size_usd * (pnl_pct_leveraged / 100))
        
        self.cm.register_trade_result(pnl_usd, pnl_pct_leveraged, atr_safe=True)
        
        state_log = self.cm.get_state()
        state_log.update({
            "timestamp": timestamp,            # Momento del cierre (para el historial)
            "entry_timestamp": self.entry_timestamp, # Momento de apertura (sin colisión)
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
        
        # --- RESET DE ESTADO POST-CIERRE ---
        self.position = None
        self.entry_price = 0.0
        self.entry_timestamp = 0 # Limpiamos el reloj
        self.position_size_usd = 0.0
        
    def _open_position(self, timestamp: int, direction: str, price: float, analysis: dict):
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
            
        # 2. Asignación del precio real y el timestamp de entrada
        actual_entry_price = result['fill_price']
            
        self.position = direction
        self.entry_price = actual_entry_price
        self.peak_price = actual_entry_price
        self.entry_timestamp = timestamp # Guardamos el inicio del trade
        self.position_size_usd = size
        self.current_leverage = self.cm.leverage
        
        state_log = self.cm.get_state()
        state_log.update({
            "timestamp": timestamp,
            "ticker": self.ticker,
            "precio_in": self.entry_price,
            "estado_bot": f"OPEN_{direction}",
            "apalancamiento": self.current_leverage,
            "fast_period": analysis['fast_period'],
            "slow_period": analysis['slow_period'],
            "cross_count": analysis['cross_count'],
            "is_frozen": analysis['is_frozen']
        })
        self.logger.log_trade(state_log)
        logging.info(f"[{self.bot_id}] ABIERTO {direction} en {actual_entry_price} | Fast: {analysis['fast_period']}, Slow: {analysis['slow_period']} | Lev: x{self.current_leverage}")
    
    async def on_candle(self, candle: Dict[str, Any], history: List[Dict[str, Any]]):
        if candle['ticker'] != self.ticker: return

        if self.position:
            if self._check_trailing_stops(candle): return

        df = pd.DataFrame(history)
        
        # Necesitamos un buffer largo para calcular la varianza (50) y normalizarla (200)
        if len(df) < self.engine.norm_window + 10: return 
        
        analysis = self.engine.analyze(df)
        raw_signal = analysis['raw_signal']
        entry_signal = analysis['entry_signal']
        is_frozen = analysis['is_frozen']

        # --- TELEMETRÍA CEREBRAL (BRAIN DUMP) ---
        brain_dump = {
            "timestamp_local": candle['timestamp'],
            "estrategia": "Medias Móviles Adaptativas",
            "estado_interno": {
                "fast_period_actual": analysis['fast_period'],
                "slow_period_actual": analysis['slow_period'],
                "cruces_detectados": analysis['cross_count'],
                "bot_congelado": is_frozen,
                "decision_latente": entry_signal if entry_signal else "HOLD"
            },
            "inputs_clave_observados": {
                "precio_actual": float(candle['close'])
            }
        }
        self.logger.log_brain_state(brain_dump)
        # ----------------------------------------------

        # 1. GESTIÓN DE CIERRE / REVERSAL
        skip_entry_this_candle = False 

        if self.position:
            # Calculamos la duración y el PnL latente (sin apalancar)
            velas_desde_inicio = (candle['timestamp'] - self.entry_timestamp) / 300000
            
            if self.position == 'LONG':
                pnl_latente = (candle['close'] - self.entry_price) / self.entry_price * 100
            else:
                pnl_latente = (self.entry_price - candle['close']) / self.entry_price * 100

            # 🛡️ REGLA DE ORO: Si subió más de 15%, bloqueamos el Reversal
            # Solo evaluamos reversal si PnL < 15%
            if velas_desde_inicio >= 3 and pnl_latente < 15.0 and self.reversal_mode != 3:
                
                hay_reversal = (self.position == 'LONG' and raw_signal == 'SHORT') or \
                               (self.position == 'SHORT' and raw_signal == 'LONG')

                if hay_reversal:
                    if self.reversal_mode == 1:
                        self._close_position(candle['timestamp'], candle['close'], "REVERSAL")
                    elif self.reversal_mode == 2:
                        self._close_position(candle['timestamp'], candle['close'], "REVERSAL_ONLY")
                        skip_entry_this_candle = True
            
        # =========================================================
        # 🛡️ 2. SISTEMA DE DEFENSA: ESCUDO DE COOLDOWN
        # =========================================================
        if entry_signal is not None:
            # A. Gatillo: 4 pérdidas consecutivas acumuladas en el CapitalManager
            if self.cm.consecutive_sl >= 4 and self.cooldown_signals_remaining == 0:
                logging.warning(f"[{self.bot_id}] 🧊 MODO COOLDOWN: 4 SL detectados. Bloqueando 5 señales.")
                
                # Seteamos el bloqueo de 5 señales (5 probadas en BT, probar 3)
                self.cooldown_signals_remaining = 5
                
                # RESET ESTRATÉGICO: Bajamos a 3 para el "Trade Sonda"
                # Así, si la señal que bloqueamos era la mala, la próxima entra normal.
                self.cm.consecutive_sl = 3
                
                entry_signal = None # Anulamos la entrada actual
                
            # B. Descontamos el bloqueo si estamos en medio de uno
            elif self.cooldown_signals_remaining > 0:
                self.cooldown_signals_remaining -= 1
                logging.info(f"[{self.bot_id}] 🛑 Señal bloqueada por historial de pérdidas.")
                entry_signal = None
        
        # 3. EVALUACIÓN DE NUEVAS ENTRADAS
        if not self.position and entry_signal and not skip_entry_this_candle:
            if is_frozen:
                logging.debug(f"[{self.bot_id}] Señal {entry_signal} ignorada (Mercado Congelado)")
                return
                
            self._open_position(candle['timestamp'], entry_signal, candle['close'], analysis)
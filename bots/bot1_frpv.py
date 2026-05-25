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

class FRPV_MathEngine:
    """Motor matemático vectorizado de la estrategia FRPV."""
    def __init__(self, kama_len=20, sma_len=200, lrc_len=40, dist_period=100):
        self.kama_len = kama_len
        self.sma_len = sma_len
        self.lrc_len = lrc_len
        self.dist_period = dist_period

    @staticmethod
    def calc_kama_custom(series: pd.Series, length: int) -> pd.Series:
        change = series.diff(length).abs()
        volatility = series.diff().abs().rolling(window=length).sum()
        er = change / volatility
        er = er.replace([np.inf, -np.inf], 0).fillna(0)
        
        fast_sc = 0.666
        slow_sc = 0.0645
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama = np.zeros_like(series.values)
        kama[:] = np.nan
        
        first_valid = series.first_valid_index()
        if first_valid is None:
            return pd.Series(kama, index=series.index)
            
        start_idx = series.index.get_loc(first_valid) + length
        if start_idx < len(series):
            kama[start_idx-1] = series.iloc[start_idx-1]
            series_vals = series.values
            sc_vals = sc.values
            
            for i in range(start_idx, len(series_vals)):
                prev_kama = kama[i-1]
                if pd.isna(prev_kama):
                    kama[i] = series_vals[i]
                else:
                    kama[i] = prev_kama + sc_vals[i] * (series_vals[i] - prev_kama)
                    
        return pd.Series(kama, index=series.index)

    def generate_signals(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        
        # 5 MINUTOS
        df['kama_5m'] = self.calc_kama_custom(df['close'], self.kama_len)
        df['sma_5m'] = ta.sma(df['close'], length=self.sma_len)
        df['lrc_5m'] = ta.linreg(df['close'], length=self.lrc_len, offset=0)
        
        df['buy_signal_5m'] = (df['close'] > df['lrc_5m']) & (df['close'] < df['kama_5m'])
        df['sell_signal_5m'] = (df['close'] < df['lrc_5m']) & (df['close'] > df['kama_5m'])

        # --- 2. LÓGICA 1 HORA ---
        df_1h = df.resample('1h', closed='right', label='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        df_1h['kama_1h'] = self.calc_kama_custom(df_1h['close'], self.kama_len)
        df_1h['sma_1h'] = ta.sma(df_1h['close'], length=self.sma_len)
        df_1h['lrc_1h'] = ta.linreg(df_1h['close'], length=self.lrc_len, offset=0)
        
        df_1h['buy_signal_1h'] = (df_1h['close'] < df_1h['sma_1h']) & (df_1h['close'] > df_1h['lrc_1h']) & (df_1h['close'] < df_1h['kama_1h'])
        df_1h['sell_signal_1h'] = (df_1h['close'] > df_1h['sma_1h']) & (df_1h['close'] < df_1h['lrc_1h']) & (df_1h['close'] > df_1h['kama_1h'])
        
        # ---> NUEVO: Lógica de Sobreextensión 1H
        df_1h['distance_to_sma_1h'] = df_1h['close'] - df_1h['sma_1h']
        df_1h['mean_distance_1h'] = ta.sma(df_1h['distance_to_sma_1h'].abs(), length=self.dist_period)
        
        df_1h['overext_above_1h'] = df_1h['close'] > (df_1h['sma_1h'] + df_1h['mean_distance_1h'])
        df_1h['overext_below_1h'] = df_1h['close'] < (df_1h['sma_1h'] - df_1h['mean_distance_1h'])
        
        # --- 3. MERGE MULTI-TIMEFRAME ---
        # 1. Alineamos el DF de 1H al índice de 5m y arrastramos el último valor conocido (ffill)
        df_1h_aligned = df_1h[['buy_signal_1h', 'sell_signal_1h', 'overext_above_1h', 'overext_below_1h']].reindex(df.index).ffill()
        
        # 2. Rellenamos los NaNs iniciales y forzamos el tipo booleano explícitamente
        # Esto silencia el warning porque ya no hay "downcasting silencioso"
        df_1h_aligned = df_1h_aligned.infer_objects(copy=False).fillna(False).astype(bool)
        
        # 3. Unimos las columnas al DF original
        df = df.join(df_1h_aligned)
        
        # 4. Señales convergentes limpias (ambos DFs ya son puramente booleanos)
        df['converging_buy_signal'] = df['buy_signal_5m'] & df['buy_signal_1h']
        df['converging_sell_signal'] = df['sell_signal_5m'] & df['sell_signal_1h']
        
        return df

class Top20ContextFilter:
    """Filtro de contexto Z-Score contra Basket Return (200 Días Reales)."""
    def __init__(self, smooth_len=5, lookback=50, bias_len=200, bias_weight_start=100):
        self.smooth_len = smooth_len
        self.lookback = lookback
        self.bias_len = bias_len
        self.bias_weight_start = bias_weight_start
        
        # Conexión directa para velas diarias
        self.exchange = ccxt.binanceusdm({'enableRateLimit': True})
        self.daily_cache = {}
        self.last_update_day = None

    def _update_daily_cache(self, symbols: list):
        """Descarga y actualiza el caché de velas diarias solo una vez por día."""
        current_day = datetime.datetime.now().date()
        
        # Si ya es el mismo día y tenemos datos, usamos la memoria RAM
        if self.last_update_day == current_day and self.daily_cache:
            return
            
        for sym in symbols:
            try:
                # Descargamos 250 días reales de historia directamente
                bars = self.exchange.fetch_ohlcv(sym, timeframe='1d', limit=250)
                df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                self.daily_cache[sym] = df
            except Exception:
                continue
                
        self.last_update_day = current_day

    def calculate_zscore(self, target_ticker: str, all_buffers: dict) -> float:
        # Extraemos la lista de monedas del DataFeed
        symbols_list = list(all_buffers.keys())
        
        # Actualiza la historia diaria en segundo plano (solo si cambió el día)
        self._update_daily_cache(symbols_list)
        
        rets, vols = [], []
        
        for sym in symbols_list:
            if sym not in self.daily_cache: 
                continue
                
            df = self.daily_cache[sym].copy() # <--- IMPORTANTE: .copy()
            
            # ---> LA MAGIA DE LA SINCRONIZACIÓN <---
            if len(all_buffers[sym]) > 0:
                ultimo_precio_vivo = all_buffers[sym][-1]['close'] 
                df.iloc[-1, df.columns.get_loc('close')] = ultimo_precio_vivo
            # ----------------------------------------
            
            rets.append(df['close'].pct_change())
            vols.append(df['volume'])
            
        if not rets: return 0.0
            
        df_rets = pd.concat(rets, axis=1)
        df_vols = pd.concat(vols, axis=1)
        
        # --- Cálculo idéntico a tu Pine Script ---
        basket_return = (df_rets * df_vols).sum(axis=1) / df_vols.sum(axis=1)
        basket_return = basket_return.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if target_ticker not in self.daily_cache:
            return 0.0
            
        target_df = self.daily_cache[target_ticker].copy() # <--- IMPORTANTE: .copy()
        
        # ---> SINCRONIZACIÓN DEL OBJETIVO <---
        if len(all_buffers[target_ticker]) > 0:
            ultimo_precio_target = all_buffers[target_ticker][-1]['close']
            target_df.iloc[-1, target_df.columns.get_loc('close')] = ultimo_precio_target
        # -------------------------------------
        
        asset_ret = target_df['close'].pct_change().fillna(0)
        
        basket_return_smooth = basket_return.rolling(window=self.smooth_len).mean()
        asset_ret_smooth = asset_ret.rolling(window=self.smooth_len).mean()
        
        perf_diff = asset_ret_smooth - basket_return_smooth
        avg_perf_diff = perf_diff.rolling(window=self.lookback).mean()
        
        if len(avg_perf_diff) < self.bias_len:
            return 0.0
            
        recent_diffs = avg_perf_diff.iloc[-self.bias_len:].values
        weights = np.ones(self.bias_len)
        weights[-self.bias_weight_start:] = 2.0
        
        sum_weight = np.sum(weights)
        weighted_mean = np.sum(recent_diffs * weights) / sum_weight
        
        weighted_var_sum = np.sum(weights * ((recent_diffs - weighted_mean) ** 2))
        weighted_stdev = max(np.sqrt(weighted_var_sum / sum_weight), 1e-12)
        
        zscore_bias = (recent_diffs[-1] - weighted_mean) / weighted_stdev
        return zscore_bias


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
        # En Bot1_FRPV.__init__
        self.reversal_mode = 3        # 1: Salir y Entrar | 2: Solo Salir | 3: Ignorar Reversal
        self.sl_cooldown_start_ts = 0  # Timestamp para el cooldown de 4h por cualquier SL
        self.cooldown_start_ts = 0  # Timestamp de cuando empezó el bloqueo
        
        self.math_engine = FRPV_MathEngine()
        self.context_filter = Top20ContextFilter()
        
        # Estado de la operación
        self.position = None
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.current_leverage = 3
        self.peak_price = 0.0

        self.hard_sl_pct = 0.0125       # 1.25%
        self.activation_pct = 0.060     # 6.0%[cite: 5]
        self.trailing_dist_pct = 0.030   # 3.0%[cite: 5]
        
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
        """
        Stop Loss inicial duro, que se transforma en Trailing Stop al superar el umbral.
        """
        if not self.position:
            return False
            
        # --- PARÁMETROS DINÁMICOS ---
        hard_sl_pct = self.hard_sl_pct       # -1.5% SL inicial | Probar 1.25%
        activation_pct = self.activation_pct    # A partir del +2% de ganancia, se activa el modo persecución | Probar 6.0%
        trailing_dist_pct = self.trailing_dist_pct # Le da 1.5% de respiro desde el pico máximo alcanzado | Probar 3.0%
        
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

        # --- NUEVO: ACTIVAR COOLDOWN POR SL ---
        if reason == "STOP_LOSS":
            self.sl_cooldown_start_ts = timestamp
            logging.warning(f"[{self.bot_id}] 🧊 Cooldown de 4h iniciado por SL en {exit_price}")
        # --------------------------------------
        
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
            
        # 2. Asignación del precio real
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
        
        # Necesitamos 200 horas de historia (aprox 2400 velas de 5m) para la SMA 200 de 1H
        if len(df) < 2410:
            return
            
        df_signals = self.math_engine.generate_signals(df)
        last_row = df_signals.iloc[-1]
        
        buy_signal = last_row['converging_buy_signal']
        sell_signal = last_row['converging_sell_signal']
        
        overext_above = last_row['overext_above_1h']
        overext_below = last_row['overext_below_1h']

        # --- NUEVO: TELEMETRÍA CEREBRAL (BRAIN DUMP) ---
        brain_dump = {
            "timestamp_local": candle['timestamp'],
            "estrategia": "FRPV (Reversión a la Media)",
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
        # ----------------------------------------------

        # =========================================================
        # 🛡️ SISTEMA DE DEFENSA: DOBLE ESCUDO (4H y 24H)
        # =========================================================
        ms_4h = 4 * 60 * 60 * 1000
        ms_24h = 24 * 60 * 60 * 1000
        en_penitencia = False

        # A. Escudo Pequeño: 4 Horas por cualquier SL
        if self.sl_cooldown_start_ts > 0:
            if (candle['timestamp'] - self.sl_cooldown_start_ts) < ms_4h:
                en_penitencia = True
            else:
                self.sl_cooldown_start_ts = 0
                logging.info(f"[{self.bot_id}] ✅ Cooldown de 4h finalizado.")

        # B. Escudo Grande: 24 Horas por racha de 4 SL
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

        # =========================================================
        # 🔄 LÓGICA DE REVERSAL (MODOS 1, 2, 3)
        # =========================================================
        if self.position:
            reversal_trigger = (self.position == 'LONG' and sell_signal) or \
                               (self.position == 'SHORT' and buy_signal)

            if reversal_trigger:
                if self.reversal_mode == 1:
                    # Modo 1: Cierra y el bloque de apertura abrirá la contraria
                    self._close_position(candle['timestamp'], candle['close'], "REVERSAL_FLIP")
                elif self.reversal_mode == 2:
                    # Modo 2: Solo sale y espera nueva señal
                    self._close_position(candle['timestamp'], candle['close'], "REVERSAL_EXIT_ONLY")
                # Modo 3: No entra acá, por ende, NO hace nada.

        # if buy_signal or sell_signal:
        #     z_score = candle.get('zscore_csv', 0)
        #     print(f"\r🎯 [DEBUG] Técnica: {'BUY' if buy_signal else 'SELL'} | Z-Score: {z_score:.4f}", end="")
            
        # =========================================================
        # 🚀 APERTURA (CON FILTRO DE PENITENCIA)
        # =========================================================
        if not self.position and not en_penitencia:
            if buy_signal or sell_signal:
                # --- LÓGICA HÍBRIDA DE CONTEXTO ---
                # Si el candle ya trae el zscore (Backtest), lo usamos. 
                # Si no, lo calculamos (Live).
                z_score = candle.get('zscore_csv')
                
                if z_score is None:
                    z_score = self.context_filter.calculate_zscore(self.ticker, self.data_feed.buffers)
                
                # Ejecución basada en el Z-Score del CSV
                if buy_signal and z_score > 0:
                    self._open_position(candle['timestamp'], 'LONG', candle['close'], overext_above, overext_below)
                elif sell_signal and z_score < 0:
                    self._open_position(candle['timestamp'], 'SHORT', candle['close'], overext_above, overext_below)
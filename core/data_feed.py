import asyncio
import json
import logging
import websockets
import time
import ccxt
from collections import deque
from typing import Callable, List, Dict, Any

class DataFeed:
    """
    Conexión WebSocket asíncrona a Binance.
    Mantiene un buffer histórico centralizado (OHLCV) precargado por REST API, 
    y despacha eventos seguros a los motores de QuantProtocol.
    """
    
    WS_URL = "wss://stream.binance.com:9443/stream?streams="

    def __init__(self, symbols: List[str], interval: str = "5m", buffer_size: int = 250):
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        # Buffer centralizado por ticker (Guardamos en MAYÚSCULAS para consistencia)
        self.buffers: Dict[str, deque] = {s.upper(): deque(maxlen=buffer_size) for s in symbols}
        
        # Firma del callback: recibe la vela actual y una copia del historial completo
        self.callbacks: List[Callable[[Dict[str, Any], List[Dict[str, Any]]], None]] = []
        
        self._running = False
        self._ws = None  # Referencia al WebSocket activo para el cierre forzado

    def register_callback(self, callback: Callable[[Dict[str, Any], List[Dict[str, Any]]], None]):
        """Registra un motor para recibir la nueva vela y el historial."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def _build_stream_url(self) -> str:
        if not self.symbols:
            raise ValueError("No se definieron símbolos para el DataFeed.")
        streams = [f"{sym}@kline_{self.interval}" for sym in self.symbols]
        return self.WS_URL + "/".join(streams)

    def preload_history(self):
        """Descarga el historial profundo por REST API evadiendo el límite de 1500 de Binance."""
        logging.info(f"[DATAFEED] Descargando historial profundo ({self.buffers[self.symbols[0].upper()].maxlen} velas)... esto va a tomar un momento.")
        
        exchange = ccxt.binanceusdm({'enableRateLimit': True}) 
        
        for sym in self.symbols: 
            ticker = sym.upper()
            try:
                limit = 1500 # Límite máximo de Binance por request
                
                # 1. Traemos las 1500 velas más recientes
                bars_recent = exchange.fetch_ohlcv(ticker, timeframe=self.interval, limit=limit)
                
                # 2. Calculamos el tiempo exacto para buscar las 1500 anteriores
                first_timestamp = bars_recent[0][0]
                # 1500 velas * 5 minutos * 60 seg * 1000 ms
                since = first_timestamp - (limit * 5 * 60 * 1000) 
                
                # 3. Traemos el bloque antiguo de velas
                bars_older = exchange.fetch_ohlcv(ticker, timeframe=self.interval, since=since, limit=limit)
                
                # 4. Unimos los dos bloques (3000 velas) y recortamos lo que pide el buffer (2500)
                all_bars = (bars_older + bars_recent)[-self.buffers[ticker].maxlen:]
                
                for bar in all_bars:
                    candle = {
                        'timestamp': bar[0],
                        'open': bar[1],
                        'high': bar[2],
                        'low': bar[3],
                        'close': bar[4],
                        'volume': bar[5],
                        'ticker': ticker
                    }
                    self.buffers[ticker].append(candle)
                    
                logging.info(f"[DATAFEED] ✅ {ticker}: Historial profundo cargado ({len(self.buffers[ticker])} velas).")
                time.sleep(0.1) # Pausa para no saturar la API
                
            except Exception as e:
                logging.error(f"[DATAFEED] ❌ Error descargando historial de {ticker}: {e}")

    async def _safe_callback(self, cb: Callable, candle: Dict[str, Any], history: List[Dict[str, Any]]):
        """Wrapper defensivo para evitar que excepciones en los bots mueran en silencio."""
        try:
            await cb(candle, history)
        except Exception as e:
            logging.error(f"DataFeed: Error crítico en bot/callback '{cb.__name__}': {e}", exc_info=True)

    async def _handle_message(self, message: str):
        """Parsea el payload, actualiza el buffer y despacha el evento."""
        try:
            data = json.loads(message)
            
            if 'data' in data and 'k' in data['data']:
                kline = data['data']['k']
                
                # Solo procesamos la vela consolidada (is_closed == True)
                if kline['x']:
                    ticker = data['data']['s']
                    candle_data = {
                        "ticker": ticker,
                        "timestamp": int(kline['t']),
                        "open": float(kline['o']),
                        "high": float(kline['h']),
                        "low": float(kline['l']),
                        "close": float(kline['c']),
                        "volume": float(kline['v'])
                    }
                    
                    # 1. Actualizamos el buffer centralizado
                    self.buffers[ticker].append(candle_data)
                    
                    # 2. Extraemos una copia estática (snapshot) para pasarle a los bots
                    history_snapshot = list(self.buffers[ticker])
                    
                    # 3. Despachamos a todos los bots subscritos de forma segura
                    for callback in self.callbacks:
                        asyncio.create_task(self._safe_callback(callback, candle_data, history_snapshot))
                        
        except json.JSONDecodeError:
            logging.error("DataFeed: Error decodificando JSON de Binance.")
        except KeyError as e:
            logging.error(f"DataFeed: Payload inválido, falta key {e}")

    async def start(self):
        """Inicia el bucle de conexión y maneja reconexiones automáticas."""
        self._running = True
        url = self._build_stream_url()
        
        logging.info(f"Iniciando DataFeed | Activos: {len(self.symbols)} | Buffer: {self.buffers[self.symbols[0].upper()].maxlen} velas")

        # ---> NUEVO: Precargar historia ANTES de conectarse al vivo <---
        # Usamos run_in_executor para que ccxt (que es síncrono) no bloquee el event loop de asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.preload_history)
        # ---------------------------------------------------------------

        logging.info("DataFeed: Conectando a WebSocket para flujo en vivo...")

        while self._running:
            try:
                async for websocket in websockets.connect(url, ping_interval=20, ping_timeout=20):
                    self._ws = websocket
                    logging.info("DataFeed: WebSocket conectado.")
                    try:
                        async for message in websocket:
                            if not self._running:
                                break
                            await self._handle_message(message)
                    except websockets.ConnectionClosed as e:
                        logging.warning(f"DataFeed: Desconexión WS ({e}). Reconectando en 3s...")
                        await asyncio.sleep(3)
                        break
            except Exception as e:
                if self._running:
                    logging.error(f"DataFeed: Falla de red: {e}. Reintentando en 5s...")
                    await asyncio.sleep(5)

    async def stop(self):
        """Detiene el bucle y cierra el socket activamente. Debe ser llamado con await."""
        logging.info("Deteniendo DataFeed y cerrando sockets...")
        self._running = False
        if self._ws:
            await self._ws.close()
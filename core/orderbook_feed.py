import asyncio
import json
import logging
import websockets
from collections import defaultdict
from typing import Dict, List, Tuple

class OrderBookFeed:
    """
    Se conecta al WebSocket de Binance Futures L2 Order Book.
    Mantiene una copia local en RAM del libro de órdenes y busca 'Muros' (Order Blocks).
    """
    def __init__(self, tickers: List[str]):
        # Binance necesita los tickers en minúscula para WebSockets (ej: btcusdt)
        self.tickers = [t.lower() for t in tickers] 
        self.wsts_url = "wss://fstream.binance.com/stream?streams="
        
        # Diccionarios para mantener el L2 local (RAM)
        self.bids: Dict[str, Dict[float, float]] = defaultdict(dict)
        self.asks: Dict[str, Dict[float, float]] = defaultdict(dict)
        
        # Configuración del escáner de Muros
        self.wall_threshold_usd = 2_000_000 # Un muro debe tener al menos 2 Millones de USD

    def _build_url(self) -> str:
        """Arma la URL con todos los streams a suscribir (@depth@100ms)"""
        streams = [f"{t}@depth@100ms" for t in self.tickers]
        return self.wsts_url + "/".join(streams)

    async def connect_and_stream(self):
        """Loop principal de conexión WebSocket."""
        url = self._build_url()
        logging.info(f"[ORDER BOOK] Conectando a {url} ...")
        
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logging.info("[ORDER BOOK] ✅ Conectado al stream de profundidad L2.")
                    
                    async for message in ws:
                        data = json.loads(message)
                        if 'data' in data:
                            self._process_update(data['data'])
                            
            except websockets.exceptions.ConnectionClosed:
                logging.warning("[ORDER BOOK] ⚠️ WebSocket desconectado. Reconectando en 3s...")
                await asyncio.sleep(3)
            except Exception as e:
                logging.error(f"[ORDER BOOK] Error crítico: {e}. Reconectando en 5s...")
                await asyncio.sleep(5)

    def _process_update(self, data: dict):
        """Actualiza el libro local con los cambios incrementales."""
        symbol = data['s'].upper()
        
        # Actualizamos Bids (Compradores)
        for price_str, qty_str in data.get('b', []):
            price, qty = float(price_str), float(qty_str)
            if qty == 0:
                self.bids[symbol].pop(price, None)
            else:
                self.bids[symbol][price] = qty
                
        # Actualizamos Asks (Vendedores)
        for price_str, qty_str in data.get('a', []):
            price, qty = float(price_str), float(qty_str)
            if qty == 0:
                self.asks[symbol].pop(price, None)
            else:
                self.asks[symbol][price] = qty

    def scan_for_walls(self, ticker: str, current_price: float) -> dict:
        """
        Escanea el libro local en busca de Muros gigantes cerca del precio.
        Retorna la pared de compra (Bid) y venta (Ask) más grande encontrada.
        """
        result = {"biggest_bid_wall": None, "biggest_ask_wall": None}
        
        # 1. Escanear Bids (Soporte / Suelo)
        bids = self.bids.get(ticker, {})
        max_bid_usd = 0
        max_bid_price = 0
        for price, qty in bids.items():
            # Filtramos para no mirar muy lejos del precio actual (ej: a más del 5%)
            if price < current_price * 0.95: continue
            
            usd_value = price * qty
            if usd_value > self.wall_threshold_usd and usd_value > max_bid_usd:
                max_bid_usd = usd_value
                max_bid_price = price
                
        if max_bid_price > 0:
            result["biggest_bid_wall"] = {"price": max_bid_price, "size_usd": max_bid_usd}

        # 2. Escanear Asks (Resistencia / Techo)
        asks = self.asks.get(ticker, {})
        max_ask_usd = 0
        max_ask_price = 0
        for price, qty in asks.items():
            if price > current_price * 1.05: continue
            
            usd_value = price * qty
            if usd_value > self.wall_threshold_usd and usd_value > max_ask_usd:
                max_ask_usd = usd_value
                max_ask_price = price
                
        if max_ask_price > 0:
            result["biggest_ask_wall"] = {"price": max_ask_price, "size_usd": max_ask_usd}

        return result
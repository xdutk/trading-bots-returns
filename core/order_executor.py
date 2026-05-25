import time
import logging
from typing import Optional, Dict, Any

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    Client = None
    BinanceAPIException = Exception

class OrderExecutor:
    """
    Puente de ejecución entre los algoritmos y el mercado.
    Abstrae la complejidad de la API de Binance y maneja el modo simulación (Paper Trading).
    """
    def __init__(self, modo_paper: bool, api_key: str = "", api_secret: str = ""):
        self.modo_paper = modo_paper
        self.client = None

        if not self.modo_paper:
            if not api_key or not api_secret:
                logging.critical("[EXECUTOR] ERROR FATAL: Faltan API Keys para MODO LIVE.")
                raise ValueError("Se requieren BINANCE_API_KEY y BINANCE_API_SECRET para MODO LIVE.")
            
            try:
                self.client = Client(api_key, api_secret)
                # Test de conexión rápido al arrancar
                self.client.futures_ping()
                logging.info("[EXECUTOR] Conectado exitosamente a Binance Futures API (MODO LIVE).")
            except BinanceAPIException as e:
                logging.critical(f"[EXECUTOR] Fallo al conectar con Binance API: {e}")
                raise
            except Exception as e:
                logging.critical(f"[EXECUTOR] Fallo de red genérico al conectar: {e}")
                raise
        else:
            logging.info("[EXECUTOR] Iniciado en MODO PAPER (Simulación). El capital está a salvo.")

    def set_leverage(self, ticker: str, leverage: int) -> None:
        """Ajusta el apalancamiento en Binance Futures."""
        if self.modo_paper:
            logging.debug(f"[EXECUTOR-PAPER] Apalancamiento simulado a x{leverage} para {ticker}")
            return

        try:
            self.client.futures_change_leverage(symbol=ticker, leverage=leverage)
            logging.info(f"[EXECUTOR-LIVE] Apalancamiento ajustado a x{leverage} en {ticker}")
        except BinanceAPIException as e:
            logging.error(f"[EXECUTOR-LIVE] Error API ajustando apalancamiento en {ticker}: {e}")
        except Exception as e:
            logging.error(f"[EXECUTOR-LIVE] Error de red ajustando apalancamiento en {ticker}: {e}")

    def open_position(self, ticker: str, direction: str, size_usd: float, 
                      leverage: int, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Abre una posición al mercado. 
        Retorna el diccionario de la orden si es exitoso, o None si ocurre un error en modo Live.
        """
        side = 'BUY' if direction.upper() == 'LONG' else 'SELL'
        
        # Tamaño nocional convertido a cantidad de cripto, redondeado a 3 decimales por seguridad
        quantity = round(size_usd / current_price, 3)

        if self.modo_paper:
            logging.info(f"[EXECUTOR-PAPER] ABRIENDO: {side} {quantity} {ticker} @ ${current_price:.2f} (Lev: x{leverage})")
            return {
                "status": "SIMULATED",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "fill_price": current_price,
                "timestamp": int(time.time() * 1000)
            }

        # --- MODO LIVE ---
        try:
            # 1. Asegurar el apalancamiento antes de disparar la orden
            self.set_leverage(ticker, leverage)
            
            # 2. Ejecutar la orden de mercado
            order = self.client.futures_create_order(
                symbol=ticker,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            # Extraemos el precio promedio de llenado (Binance suele devolverlo en avgPrice)
            fill_price = float(order.get('avgPrice', 0.0))
            if fill_price == 0.0:
                fill_price = current_price  # Fallback en caso de latencia de la API en el update

            logging.info(f"[EXECUTOR-LIVE] ✅ ORDEN LLENADA: {side} {quantity} {ticker} @ ${fill_price:.2f}")

            return {
                "status": "FILLED",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "fill_price": fill_price,
                "timestamp": int(order.get('updateTime', time.time() * 1000))
            }

        except BinanceAPIException as e:
            logging.critical(f"[EXECUTOR-LIVE] ❌ ERROR API Binance abriendo {side} en {ticker}: {e}")
            return None
        except Exception as e:
            logging.critical(f"[EXECUTOR-LIVE] ❌ ERROR CRÍTICO abriendo {side} en {ticker}: {e}")
            return None

    def close_position(self, ticker: str, direction: str, quantity: float, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Cierra una posición abierta invirtiendo el side con reduceOnly=True.
        """
        # Invertimos la dirección de la posición original para cerrarla
        side = 'SELL' if direction.upper() == 'LONG' else 'BUY'
        quantity = round(quantity, 3)

        if self.modo_paper:
            logging.info(f"[EXECUTOR-PAPER] CERRANDO: {side} {quantity} {ticker} @ ${current_price:.2f}")
            return {
                "status": "SIMULATED",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "fill_price": current_price,
                "timestamp": int(time.time() * 1000)
            }

        # --- MODO LIVE ---
        try:
            order = self.client.futures_create_order(
                symbol=ticker,
                side=side,
                type='MARKET',
                quantity=quantity,
                reduceOnly=True  # Protección: Garantiza que solo achique/cierre la posición
            )

            fill_price = float(order.get('avgPrice', 0.0))
            if fill_price == 0.0:
                fill_price = current_price

            logging.info(f"[EXECUTOR-LIVE] ✅ CIERRE LLENADO: {side} {quantity} {ticker} @ ${fill_price:.2f}")

            return {
                "status": "FILLED",
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "fill_price": fill_price,
                "timestamp": int(order.get('updateTime', time.time() * 1000))
            }

        except BinanceAPIException as e:
            logging.critical(f"[EXECUTOR-LIVE] ❌ ERROR API Binance cerrando {ticker}: {e}")
            return None
        except Exception as e:
            logging.critical(f"[EXECUTOR-LIVE] ❌ ERROR CRÍTICO cerrando {ticker}: {e}")
            return None
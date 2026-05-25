import os
import io
import json
import logging
import requests
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DATA PIPELINE] - %(message)s')

class HFTDataDownloader:
    """
    Descarga datos crudos del L2 Order Book de Binance Vision,
    los procesa y los convierte en el formato de 12 Features para el Bot 5.
    """
    def __init__(self, output_folder="data/hft_historical/"):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.base_url = "https://data.binance.vision/data/futures/um/daily/depthSnapshots"

    def download_and_process(self, symbol: str, start_date: str, required_days: int = 1):
        """
        Descarga archivos ZIP diarios. Si hay un error 404 (Binance no subió el dato),
        retrocede un día automáticamente hasta conseguir la cantidad de días requeridos.
        """
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        days_downloaded = 0
        attempts = 0
        max_attempts = 30 # Límite de seguridad para no quedar en un loop infinito
        
        while days_downloaded < required_days and attempts < max_attempts:
            date_str = current_date.strftime("%Y-%m-%d")
            filename = f"{symbol}-depthSnapshots-{date_str}.zip"
            url = f"{self.base_url}/{symbol}/{filename}"
            
            logging.info(f"Buscando {symbol} para el {date_str}...")
            
            try:
                # Usamos stream=True para no cargar todo en RAM de golpe si el archivo es gigante
                response = requests.get(url, timeout=15, stream=True)
                
                if response.status_code == 200:
                    logging.info(f"✅ Archivo encontrado. Descomprimiendo en memoria...")
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        csv_filename = z.namelist()[0]
                        with z.open(csv_filename) as f:
                            raw_df = pd.read_csv(f)
                            
                    logging.info(f"Datos descargados crudos: {len(raw_df)} ticks. Iniciando compresión de IA...")
                    processed_df = self._compress_to_features(raw_df)
                    
                    if processed_df is not None and not processed_df.empty:
                        out_name = os.path.join(self.output_folder, f"HFT_{symbol}_{date_str}.csv")
                        processed_df.to_csv(out_name, index=False)
                        logging.info(f"💾 Guardado exitoso: {out_name} ({len(processed_df)} filas procesadas)")
                        days_downloaded += 1 # Contamos un día de éxito
                else:
                    logging.warning(f"⚠️ HTTP 404: Binance no subió la data del {date_str}. Buscando el día anterior...")
                    
            except Exception as e:
                logging.error(f"Error crítico procesando {date_str}: {e}")
                
            # Restamos un día para buscar en el pasado de forma automática
            current_date -= timedelta(days=1)
            attempts += 1
            
        if days_downloaded < required_days:
            logging.warning(f"Terminó la búsqueda. Se consiguieron {days_downloaded}/{required_days} días.")

    def _compress_to_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma el Order Book crudo de Binance a las 12 variables HFT.
        """
        processed_data = []
        
        # Variables de memoria para calcular los "Deltas"
        prev_imbalance = 0.5
        prev_spread = 0.0
        bid_persistence, ask_persistence = 0, 0
        last_bid_wall, last_ask_wall = 0.0, 0.0
        
        wall_threshold_usd = 2_000_000 # 2 Millones USD para ser considerado "Muro"

        # Binance guarda los bids/asks como strings con formato de lista de listas: "[[price, qty], ...]"
        for index, row in df.iterrows():
            try:
                # Parsear el string JSON a listas de Python
                bids = json.loads(row['bids'])
                asks = json.loads(row['asks'])
                
                if not bids or not asks: continue
                
                # Bids y Asks ya vienen ordenados (Mejor precio primero)
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid_price = (best_bid + best_ask) / 2.0
                
                # 1. Imbalance (Presión de compras vs ventas en los primeros 5 niveles)
                bid_vol_top5 = sum(float(qty) for price, qty in bids[:5])
                ask_vol_top5 = sum(float(qty) for price, qty in asks[:5])
                imbalance = bid_vol_top5 / (bid_vol_top5 + ask_vol_top5 + 1e-9)
                
                # 2. Búsqueda de Muros (Francotirador)
                max_bid_price, max_bid_usd = best_bid, 0.0
                for price_str, qty_str in bids:
                    p, q = float(price_str), float(qty_str)
                    usd_val = p * q
                    if usd_val > wall_threshold_usd and usd_val > max_bid_usd:
                        max_bid_usd, max_bid_price = usd_val, p
                        
                max_ask_price, max_ask_usd = best_ask, 0.0
                for price_str, qty_str in asks:
                    p, q = float(price_str), float(qty_str)
                    usd_val = p * q
                    if usd_val > wall_threshold_usd and usd_val > max_ask_usd:
                        max_ask_usd, max_ask_price = usd_val, p
                
                # Ratios de Muros vs Promedio
                avg_bid_size = np.mean([float(p)*float(q) for p,q in bids]) + 1e-9
                avg_ask_size = np.mean([float(p)*float(q) for p,q in asks]) + 1e-9
                bid_wall_ratio = max_bid_usd / avg_bid_size
                ask_wall_ratio = max_ask_usd / avg_ask_size
                
                # Distancias relativas
                dist_bid_wall = (mid_price - max_bid_price) / mid_price
                dist_ask_wall = (max_ask_price - mid_price) / mid_price
                spread = (best_ask - best_bid) / mid_price
                
                # 3. Deltas y Persistencia
                delta_imbalance = imbalance - prev_imbalance
                delta_spread = spread - prev_spread
                
                if max_bid_price == last_bid_wall and max_bid_price > 0: bid_persistence += 1
                else: bid_persistence = 0; last_bid_wall = max_bid_price
                    
                if max_ask_price == last_ask_wall and max_ask_price > 0: ask_persistence += 1
                else: ask_persistence = 0; last_ask_wall = max_ask_price
                
                # Oráculo Neutral (En simulación histórica no tenemos el oráculo exacto del pasado, asumimos neutral)
                sentiment = 0.5
                pos_encoded = 0 # El entrenamiento es sin posición previa
                
                # Guardar fila
                processed_data.append([
                    mid_price, imbalance, bid_wall_ratio, ask_wall_ratio,
                    dist_bid_wall, dist_ask_wall, spread,
                    delta_imbalance, delta_spread,
                    bid_persistence, ask_persistence,
                    sentiment, pos_encoded
                ])
                
                # Actualizar memoria
                prev_imbalance, prev_spread = imbalance, spread
                
            except Exception as e:
                # Si una fila está corrupta, la saltamos
                continue
                
        cols = ['mid_price', 'imbalance', 'bid_wall_ratio', 'ask_wall_ratio',
                'dist_bid_wall', 'dist_ask_wall', 'spread',
                'delta_imbalance', 'delta_spread',
                'bid_wall_persistence', 'ask_wall_persistence',
                'sentiment', 'pos_encoded']
                
        return pd.DataFrame(processed_data, columns=cols)

if __name__ == "__main__":
    downloader = HFTDataDownloader()
    
    print("Iniciando Pipeline de Datos HFT (Modo Cazador)...")
    
    # Binance tarda un par de días en consolidar el zip actual
    # Empezamos a buscar desde hace 3 días y le pedimos que nos consiga 3 archivos válidos
    start_search_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    
    downloader.download_and_process(symbol="BTCUSDT", start_date=start_search_date, required_days=3)
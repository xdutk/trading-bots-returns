import time
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ORACLE DAEMON] - %(levelname)s - %(message)s'
)

class SentimentOracle:
    @staticmethod
    def read_cached_score() -> int:
        """
        Función helper para que los bots (como Bot 4) puedan leer 
        el score guardado en el disco sin romper nada.
        """
        try:
            with open("sentiment_score.txt", "r") as f:
                score = int(f.read().strip())
                return max(1, min(10, score))
        except Exception:
            return 5 # Si el archivo no existe o hay error, devuelve 5 (Neutral)

def get_binance_ls_ratio(symbol="BTCUSDT", period="5m"):
    """
    Obtiene el Global Long/Short Ratio de Binance Futures.
    No requiere API Key (es un endpoint público).
    """
    url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period={period}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # El endpoint devuelve una lista histórica, el último elemento es el actual
        if data and isinstance(data, list):
            latest_ratio = float(data[-1]['longShortRatio'])
            return latest_ratio
    except Exception as e:
        logging.error(f"Error consultando Binance L/S Ratio: {e}")
    return None

def translate_ratio_to_score(ratio: float) -> int:
    """
    Traduce el ratio matemático de Binance a la escala 1-10 que esperan los Bots.
    - Ratio 1.0 (Empate) -> Score 5 (Neutral)
    - Ratio > 1.0 (Más Longs) -> Score > 5 (Euforia)
    - Ratio < 1.0 (Más Shorts) -> Score < 5 (Pánico)
    """
    if ratio >= 1.0:
        # Mapeamos ratios de 1.0 a 3.0+ hacia scores de 5 a 10
        score = 5 + ((ratio - 1.0) / 1.5) * 5
    else:
        # Mapeamos ratios de 0.3 a 1.0 hacia scores de 1 a 5
        score = 5 - ((1.0 - ratio) / 0.7) * 4
        
    return max(1, min(10, int(round(score))))

def main():
    logging.info("Iniciando Sentiment Oracle (Modo: Binance Long/Short Ratio)")
    logging.info("El Oráculo correrá cada 5 minutos sincronizado con los bots.")
    
    while True:
        try:
            # Consultamos a Binance
            ratio = get_binance_ls_ratio()
            
            if ratio is not None:
                # Traducimos a la escala de la IA
                score = translate_ratio_to_score(ratio)
                
                # Guardamos para que el Orquestador lo lea
                with open("sentiment_score.txt", "w") as f:
                    f.write(str(score))
                
                logging.info(f"Ratio crudo: {ratio:.2f} -> Score traducido IA: {score}/10")
            else:
                logging.warning("No se pudo obtener ratio. Manteniendo score anterior por seguridad.")

        except Exception as e:
            logging.error(f"Error crítico en el loop del Oráculo: {e}")

        # Dormimos exactamente 5 minutos (300 segundos) para no saturar la API
        time.sleep(300)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Apagado manual detectado. Cerrando Sentiment Oracle.")
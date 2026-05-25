import csv
import os
import json # <--- AGREGADO
import logging
from pathlib import Path
from typing import Optional

class BotLogger:
    """
    Maneja la persistencia de estado y logs de operaciones en CSV.
    Garantiza que los datos se guarden en disco instantáneamente para 
    soportar rehidratación tras cortes abruptos del proceso.
    """
    
    # Definimos las columnas exactamente como en el Blueprint
    FIELDS = [
        "timestamp", 
        "ticker", 
        "capital_bloque", 
        "apalancamiento", 
        "precio_in", 
        "precio_out", 
        "resultado_usd", 
        "rendimiento_pct", 
        "estado_bot", 
        "balance", 
        "consecutive_sl",
        "overext_above",  # <--- Bot 1
        "overext_below",   # <--- Bot 1
        "ai_confidence",   # <--- Bot 2
        "fast_period",   # <--- Bot 3
        "slow_period",   # <--- Bot 3
        "cross_count",   # <--- Bot 3
        "is_frozen"      # <--- Bot 3
    ]

    def __init__(self, bot_id: str, filepath: str):
        self.bot_id = bot_id
        self.filepath = Path(filepath)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Crea el archivo y los encabezados si no existe."""
        if not self.filepath.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    def log_trade(self, trade_data: dict):
        """
        Escribe una fila en el CSV de forma atómica y segura.
        Filtra automáticamente cualquier dato sobrante del diccionario.
        """
        row = {field: trade_data.get(field, "") for field in self.FIELDS}
        
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)
            
            # Forzamos el vaciado del buffer de Python
            f.flush()
            # Obligamos al Sistema Operativo a escribir físicamente en el disco
            os.fsync(f.fileno())

    def get_last_state(self) -> Optional[dict]:
        """
        Lee la última fila válida del CSV para rehidratar el estado.
        Convierte los strings del CSV a sus tipos correctos.
        Retorna None si el archivo está vacío (solo tiene headers).
        """
        if not self.filepath.exists():
            return None
            
        last_row = None
        try:
            with open(self.filepath, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # NOTA PARA EL FUTURO: Si el CSV crece a >100k filas, 
                # optimizar leyendo el archivo desde el final en reversa.
                for row in reader:
                    last_row = row
        except Exception:
            return None
            
        if not last_row:
            return None

        # Bloque de Rehidratación: Cast estricto de tipos
        try:
            return {
                "timestamp": last_row["timestamp"],
                "ticker": last_row["ticker"],
                "estado_bot": last_row["estado_bot"],
                "capital_bloque": float(last_row["capital_bloque"]) if last_row["capital_bloque"] else 0.0,
                "apalancamiento": int(last_row["apalancamiento"]) if last_row["apalancamiento"] else 3,
                "precio_in": float(last_row["precio_in"]) if last_row["precio_in"] else 0.0,
                "precio_out": float(last_row["precio_out"]) if last_row["precio_out"] else 0.0,
                "resultado_usd": float(last_row["resultado_usd"]) if last_row["resultado_usd"] else 0.0,
                "rendimiento_pct": float(last_row["rendimiento_pct"]) if last_row["rendimiento_pct"] else 0.0,
                "balance": float(last_row["balance"]) if last_row["balance"] else 1250.0,
                "consecutive_sl": int(last_row["consecutive_sl"]) if last_row["consecutive_sl"] else 0,
                "overext_above": last_row.get("overext_above") == "True",
                "overext_below": last_row.get("overext_below") == "True",
                "ai_confidence": float(last_row["ai_confidence"]) if last_row.get("ai_confidence") else 0.0,
                "fast_period": int(last_row["fast_period"]) if last_row.get("fast_period") else 0,
                "slow_period": int(last_row["slow_period"]) if last_row.get("slow_period") else 0,
                "cross_count": int(last_row["cross_count"]) if last_row.get("cross_count") else 0,
                "is_frozen": last_row.get("is_frozen") == "True"
            }
        except ValueError as e:
            # Reemplazamos el print por el logger para producción
            logging.warning(f"[{self.bot_id}] Fila corrompida en rehidratación: {e}")
            return None

    def log_brain_state(self, brain_data: dict): # <--- MÉTODO AGREGADO ACÁ
        """
        Vuelca la telemetría de Alta Frecuencia (Caja Blanca) a un archivo JSON.
        Se sobreescribe constantemente para no saturar el disco.
        """
        # Genera un archivo tipo: logs/bot4_FTMUSDT_brain.json
        filepath = os.path.join(os.path.dirname(self.filepath), f"{self.bot_id}_brain.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(brain_data, f, indent=4)
        except Exception as e:
            # Falla silenciosa para que un error de I/O no crashee al bot
            pass
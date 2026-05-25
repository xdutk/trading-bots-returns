import asyncio
import websockets
import json
import time

async def main():
    # URL con formato combined stream de Binance
    url = "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_5m/btcusdt@depth5"
    
    print("="*60)
    print(" 📡 TEST DE CONEXIÓN WEBSOCKET Y LATENCIA NTP 📡 ")
    print("="*60)
    print(f"[ INFO ] Conectando a: {url}...")

    try:
        # El bloque async with garantiza el cierre limpio del socket al salir
        async with websockets.connect(url) as ws:
            print("[ OK ] Conexión establecida. Escuchando el torrente de datos...\n")
            
            for i in range(1, 6):
                raw_msg = await ws.recv()
                # Capturamos el tiempo local en milisegundos en el instante exacto de recepción
                local_time_ms = int(time.time() * 1000)
                
                msg_data = json.loads(raw_msg)
                
                # En streams combinados, la info real viaja dentro del nodo 'data'
                stream_name = msg_data.get('stream', 'Desconocido')
                payload = msg_data.get('data', {})
                event_time = payload.get('E')  # 'E' es el Event Time estándar de Binance
                
                print(f"--- Mensaje {i}/5 | Stream: {stream_name} ---")
                
                if event_time:
                    # Calculamos el delta. Usamos abs() por si el reloj local está atrasado
                    latency = local_time_ms - event_time
                    abs_latency = abs(latency)
                    
                    print(f"Tiempo Local  : {local_time_ms} ms")
                    print(f"Tiempo Binance: {event_time} ms")
                    print(f"Delta         : {latency} ms")
                    
                    if abs_latency > 500:
                        print(f"⚠️ ADVERTENCIA: LATENCIA ALTA ({abs_latency}ms) — Ajustar NTP antes de iniciar main.py")
                    else:
                        print(f"⚡ Latencia óptima ({abs_latency}ms)")
                else:
                    print("⚠️ Mensaje recibido sin campo 'E' (Event Time).")
                    
                # Formateo legible del JSON. Truncamos si es muy largo (ej. order book depth5)
                formatted_json = json.dumps(payload, indent=2)
                lines = formatted_json.split('\n')
                if len(lines) > 12:
                    short_json = '\n'.join(lines[:12]) + "\n  ... [contenido truncado para legibilidad]"
                else:
                    short_json = formatted_json
                    
                print(f"Payload:\n{short_json}\n")
                
            print("="*60)
            print("✅ WebSocket OK — Sistema listo para arrancar")
            print("="*60)
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ ERROR: La conexión se cerró inesperadamente. Código: {e.code}")
    except Exception as e:
        print(f"\n❌ ERROR FATAL al conectar: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ INFO ] Test interrumpido por el usuario.")
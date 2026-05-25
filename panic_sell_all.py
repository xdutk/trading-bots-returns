import os
import sys
import psutil
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv  # <-- 1. Importar la librería

# Cargar variables del archivo .env al entorno
load_dotenv()

def kill_main_process():
    """Busca procesos Python ejecutando main.py y los liquida."""
    print("\n[ SISTEMA ] Buscando proceso principal main.py...")
    killed = False
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and 'python' in proc.info['name'].lower():
                if any('main.py' in cmd_part for cmd_part in cmdline):
                    print(f"    -> Encontrado PID {proc.info['pid']}: {' '.join(cmdline)}")
                    proc.kill()  # SIGKILL fulminante
                    killed = True
                    print(f"    ✅ Proceso {proc.info['pid']} aniquilado.")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
            
    if not killed:
        print("    ⚪ No se encontró ningún main.py corriendo.")

def main():
    print("="*50)
    print(" 🚨 MODO PÁNICO: LIQUIDACIÓN TOTAL DE QUANTPROTOCOL 🚨 ")
    print("="*50)
    print("Este script CERRARÁ TODAS las posiciones abiertas a precio de mercado")
    print("y CANCELARÁ TODAS las órdenes pendientes en Binance Futures.")
    print("="*50)
    
    confirm = input("\nEscribí CONFIRMAR para ejecutar Panic Sell: ")
    if confirm != "CONFIRMAR":
        print("\nAbortado. No se ejecutó ninguna acción.")
        sys.exit(0)
        
    print("\n[ CREDENCIALES ] Leyendo variables de entorno...")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ ERROR CRÍTICO: Faltan BINANCE_API_KEY o BINANCE_API_SECRET en el entorno.")
        sys.exit(1)
        
    try:
        client = Client(api_key, api_secret)
        # Test rápido de conexión
        client.futures_ping()
        print("✅ Conectado a Binance Futures exitosamente.")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: Fallo al conectar con Binance. Detalle: {e}")
        sys.exit(1)

    # --- 1. CERRAR POSICIONES ---
    print("\n[ POSICIONES ] Obteniendo estado del portafolio...")
    try:
        positions = client.futures_position_information()
        open_positions = [p for p in positions if float(p['positionAmt']) != 0.0]
        
        if not open_positions:
            print("    ⚪ No hay posiciones abiertas.")
        else:
            for pos in open_positions:
                symbol = pos['symbol']
                amt = float(pos['positionAmt'])
                side = 'SELL' if amt > 0 else 'BUY'
                abs_amt = abs(amt)
                
                print(f"    -> Liquidando {symbol} | Cantidad: {amt} | Dirección de cierre: {side}")
                try:
                    client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type='MARKET',
                        quantity=abs_amt,
                        reduceOnly=True  # Protección: Asegura que solo cierre, no abra
                    )
                    print(f"    ✅ Posición de {symbol} cerrada.")
                except BinanceAPIException as e:
                    print(f"    ❌ ERROR cerrando {symbol}: {e}")
                except Exception as e:
                    print(f"    ❌ Falla de red cerrando {symbol}: {e}")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO obteniendo posiciones: {e}")

    # --- 2. LIMPIAR EL LIBRO DE ÓRDENES ---
    print("\n[ ÓRDENES ] Buscando órdenes pendientes (Limit, Stop Loss, Take Profit)...")
    try:
        open_orders = client.futures_get_open_orders()
        symbols_with_orders = set([order['symbol'] for order in open_orders])
        
        if not symbols_with_orders:
            print("    ⚪ No hay órdenes pendientes.")
        else:
            for symbol in symbols_with_orders:
                try:
                    client.futures_cancel_all_open_orders(symbol=symbol)
                    print(f"    ✅ Órdenes limpiadas para {symbol}.")
                except BinanceAPIException as e:
                    print(f"    ❌ ERROR cancelando órdenes de {symbol}: {e}")
                except Exception as e:
                    print(f"    ❌ Falla de red cancelando órdenes de {symbol}: {e}")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO obteniendo órdenes pendientes: {e}")

    # --- 3. LIQUIDAR EL MOTOR LOCAL ---
    kill_main_process()
    
    print("\n" + "="*50)
    print(" 🛡️ SECUENCIA DE PÁNICO COMPLETADA 🛡️ ")
    print("="*50)

if __name__ == "__main__":
    main()
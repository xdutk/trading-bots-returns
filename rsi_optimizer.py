import pandas as pd
import pandas_ta as ta
import numpy as np

def optimizar_rsi(csv_path, sl=1.00, tp=4.0):
    print(f"🚀 Analizando calidad de señales en: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    # 1. Indicadores Base (Bot 3 Adaptive)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema_f'] = ta.ema(df['close'], length=10) # Usamos promedios estándar para la prueba
    df['ema_s'] = ta.ema(df['close'], length=30)
    
    # 2. Identificar Cruces (Signals)
    df['trend'] = np.where(df['ema_f'] > df['ema_s'], 1, -1)
    df['cross'] = df['trend'].diff().fillna(0) != 0
    
    signals = df[df['cross'] == True].copy()
    resultados = []

    # 3. Simulación Rápida de cada Señal
    for i in signals.index:
        rsi_val = df.loc[i, 'rsi']
        tipo = "LONG" if df.loc[i, 'trend'] == 1 else "SHORT"
        entry_p = df.loc[i, 'close']
        
        # Miramos el futuro (próximas 100 velas) para ver qué toca primero
        future = df.loc[i+1 : i+100]
        win = False
        
        for p in future.itertuples():
            price = p.close
            change = ((price - entry_p) / entry_p) * 100 if tipo == "LONG" else ((entry_p - price) / entry_p) * 100
            
            if change >= tp: # Toca el objetivo
                win = True
                break
            if change <= -sl: # Toca el Stop Loss
                win = False
                break
        
        resultados.append({'rsi': rsi_val, 'tipo': tipo, 'win': win})

    # 4. Tabla de Probabilidades por Rangos de RSI
    res_df = pd.DataFrame(resultados)
    
    print("\n📊 RESULTADOS POR UMBRAL DE RSI (LONG):")
    print("-" * 40)
    for r in range(40, 65, 2):
        filtro = res_df[(res_df['tipo'] == 'LONG') & (res_df['rsi'] < r)]
        wr = (filtro['win'].mean() * 100) if len(filtro) > 0 else 0
        print(f"RSI < {r} | Trades: {len(filtro):<4} | WinRate: {wr:.2f}%")

    print("\n📊 RESULTADOS POR UMBRAL DE RSI (SHORT):")
    print("-" * 40)
    for r in range(35, 60, 2):
        filtro = res_df[(res_df['tipo'] == 'SHORT') & (res_df['rsi'] > r)]
        wr = (filtro['win'].mean() * 100) if len(filtro) > 0 else 0
        print(f"RSI > {r} | Trades: {len(filtro):<4} | WinRate: {wr:.2f}%")

if __name__ == "__main__":
    optimizar_rsi("data/SOLUSDT.csv") # Asegurate de que la ruta sea correcta
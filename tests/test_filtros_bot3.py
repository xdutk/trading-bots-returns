import pandas as pd
import numpy as np
import time

def calcular_adx(df, period=14):
    """Calcula el ADX usando Pandas"""
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff(-1) * -1
    
    plus_dm[plus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0
    minus_dm[minus_dm < plus_dm] = 0
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def calcular_macd(df, fast=12, slow=26, signal=9):
    """Calcula el MACD Clásico"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calcular_rsi(df, period=14):
    """Calcula el RSI Clásico"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def evaluar_winrate(df, columna_señal, sl_pct=0.015, tp_pct=0.025):
    """
    Simulador ultrarrápido: Mira hacia el futuro desde la señal para ver si toca TP o SL.
    """
    trades = 0
    wins = 0
    
    señales = df[df[columna_señal].notna()]
    
    for idx, row in señales.iterrows():
        direccion = row[columna_señal]
        precio_entrada = row['close']
        
        # Recortamos el dataframe desde la vela siguiente hasta 100 velas en el futuro
        futuro = df.loc[idx+1 : idx+100] 
        
        for _, vela_futura in futuro.iterrows():
            if direccion == 'LONG':
                if vela_futura['low'] <= precio_entrada * (1 - sl_pct):
                    trades += 1
                    break # Perdió
                elif vela_futura['high'] >= precio_entrada * (1 + tp_pct):
                    trades += 1
                    wins += 1
                    break # Ganó
                    
            elif direccion == 'SHORT':
                if vela_futura['high'] >= precio_entrada * (1 + sl_pct):
                    trades += 1
                    break # Perdió
                elif vela_futura['low'] <= precio_entrada * (1 - tp_pct):
                    trades += 1
                    wins += 1
                    break # Ganó

    win_rate = (wins / trades * 100) if trades > 0 else 0
    return trades, win_rate

def run_laboratorio():
    print("🧪 Iniciando Laboratorio Quant: Pruebas de Filtro Bot 3")
    start_time = time.time()
    
    # 1. CARGA DE DATOS
    df = pd.read_csv("data/SOLUSDT.csv")
    df.columns = [col.lower() for col in df.columns]
    df = df.tail(200000).reset_index(drop=True)
    
    # 2. SEÑAL BASE (Cruces rápidos simulados del Bot 3)
    # Para hacerlo en 2 segundos, usamos una EMA 15 y EMA 45 (el promedio que suele usar tu motor adaptativo)
    df['ema_fast'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=45, adjust=False).mean()
    df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    df['cross'] = df['trend'].diff()
    
    df['signal_base'] = np.nan
    df.loc[df['cross'] == 2, 'signal_base'] = 'LONG'
    df.loc[df['cross'] == -2, 'signal_base'] = 'SHORT'
    
    # 3. CÁLCULO DE INDICADORES
    df['adx'] = calcular_adx(df)
    df['macd'], df['macd_signal'] = calcular_macd(df)
    df['rsi'] = calcular_rsi(df)
    
    # 4. APLICACIÓN DE FILTROS
    
    # Filtro A: ADX > 25 (Solo opera si hay fuerza de tendencia)
    df['signal_adx'] = df['signal_base']
    df.loc[(df['signal_base'].notna()) & (df['adx'] < 25), 'signal_adx'] = np.nan
    
    # Filtro B: MACD (MACD Clásico a favor de la tendencia)
    df['signal_macd'] = df['signal_base']
    df.loc[(df['signal_base'] == 'LONG') & (df['macd'] < df['macd_signal']), 'signal_macd'] = np.nan
    df.loc[(df['signal_base'] == 'SHORT') & (df['macd'] > df['macd_signal']), 'signal_macd'] = np.nan
    
    # Filtro C: RSI (Comprar en sobreventa 45, Vender en sobrecompra 55)
    df['signal_rsi'] = df['signal_base']
    df.loc[(df['signal_base'] == 'LONG') & (df['rsi'] > 45), 'signal_rsi'] = np.nan
    df.loc[(df['signal_base'] == 'SHORT') & (df['rsi'] < 55), 'signal_rsi'] = np.nan

    # 5. EVALUACIÓN DE RESULTADOS
    print("\n📊 RESULTADOS DEL WIN RATE (Riesgo Fijo: 1.5% SL / 2.0% TP)")
    print("-" * 50)
    
    trades_base, wr_base = evaluar_winrate(df, 'signal_base')
    print(f"📉 BOT 3 BASE (Sin Filtro) : {trades_base} trades | Win Rate: {wr_base:.2f}%")
    
    trades_adx, wr_adx = evaluar_winrate(df, 'signal_adx')
    print(f"🔥 + FILTRO ADX (>25)      : {trades_adx} trades | Win Rate: {wr_adx:.2f}%")
    
    trades_macd, wr_macd = evaluar_winrate(df, 'signal_macd')
    print(f"🌊 + FILTRO MACD (Momentum): {trades_macd} trades | Win Rate: {wr_macd:.2f}%")
    
    trades_rsi, wr_rsi = evaluar_winrate(df, 'signal_rsi')
    print(f"🎯 + FILTRO RSI (Pullbacks): {trades_rsi} trades | Win Rate: {wr_rsi:.2f}%")
    
    print("-" * 50)
    print(f"⏱️ Laboratorio finalizado en {time.time() - start_time:.2f} segundos.")

if __name__ == "__main__":
    run_laboratorio()
import pandas as pd
import numpy as np

def audit_hft_csv(filepath):
    print(f"\n🔍 INICIANDO AUDITORÍA FORENSE: {filepath}\n" + "="*50)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Error leyendo el archivo: {e}")
        return

    # 1. Chequeo de Tamaño
    print(f"📊 Total de Ticks (Filas): {len(df):,}")
    
    # 2. Análisis del Precio (El más importante)
    if 'mid_price' in df.columns:
        precio_max = df['mid_price'].max()
        precio_min = df['mid_price'].min()
        rango_dolares = precio_max - precio_min
        rango_pct = (rango_dolares / precio_min) * 100
        
        print(f"\n💰 RADIOGRAFÍA DEL PRECIO (mid_price):")
        print(f" • Precio Mínimo: ${precio_min:.4f}")
        print(f" • Precio Máximo: ${precio_max:.4f}")
        print(f" • Rango de Movimiento Total: ${rango_dolares:.4f} ({rango_pct:.2f}%)")
        
        if rango_pct < 0.5:
            print(" ⚠️ ALERTA: El precio se movió menos de un 0.5% en toda la historia.")
            print("     Es físicamente imposible que toque un Take Profit de 0.5%.")
    else:
        print("❌ NO EXISTE LA COLUMNA 'mid_price'")

    # 3. Análisis de Variabilidad de las Features (Para ver si están "congeladas")
    print("\n🧬 RADIOGRAFÍA DE FEATURES (¿Están vivos los datos?):")
    features = [
        'imbalance', 'bid_wall_ratio', 'ask_wall_ratio',
        'dist_bid_wall', 'dist_ask_wall', 'spread',
        'sentiment'
    ]
    
    for feat in features:
        if feat in df.columns:
            desvio = df[feat].std()
            if desvio == 0:
                print(f" 💀 {feat}: MUERTO (Variación 0. Todos los valores son idénticos)")
            elif pd.isna(desvio):
                print(f" 💀 {feat}: LLENO DE NaNs (Datos corruptos)")
            else:
                print(f" ✅ {feat}: VIVO (Desvío Estándar: {desvio:.6f})")
        else:
            print(f" ❌ {feat}: NO EXISTE EN EL CSV")

    # 4. Auditoría de Muros
    if 'bid_wall_ratio' in df.columns and 'ask_wall_ratio' in df.columns:
        print("\n🧱 ANÁLISIS DE MUROS (Institucionales):")
        print(f" • Bid Wall Promedio: {df['bid_wall_ratio'].mean():.2f}x")
        print(f" • Ask Wall Promedio: {df['ask_wall_ratio'].mean():.2f}x")
        if df['bid_wall_ratio'].max() == 0 and df['ask_wall_ratio'].max() == 0:
            print(" ⚠️ ALERTA: Nunca se detectó un muro de liquidez en todo el archivo.")

    print("\n" + "="*50)

if __name__ == "__main__":
    archivo = "data/hft_historical/HFT_SOLUSDT_LIVE_RECORDING.csv"
    audit_hft_csv(archivo)
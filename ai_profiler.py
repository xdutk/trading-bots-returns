import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from tensorflow.keras.models import load_model
import warnings

# Silenciamos warnings de TF y Pandas para consola limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

class AIModelProfiler:
    """
    Herramienta institucional para auditar la distribución de confianza 
    de los modelos de IA (LSTM, RL) y calibrar los umbrales (Thresholds).
    """
    def __init__(self, models_folder="models/", data_folder="data/"):
        self.models_folder = models_folder
        self.data_folder = data_folder

    def _menu_seleccion(self):
        print("\n" + "="*60)
        print(" 🧠 AI CONFIDENCE PROFILER | QuantProtocol")
        print("="*60)
        print(" [2] Bot 2: HMM + LSTM (Velas 5m)")
        print(" [4] Bot 4: RL Swing (Velas 5m)")
        print(" [5] Bot 5: HFT RL (Ticks L2)")
        print("="*60)
        
        opcion = input("🔢 Elegí el modelo a perfilar (2, 4, 5): ")
        return int(opcion) if opcion.isdigit() else 2

    def _cargar_datos(self, filepath, n_samples=50000):
        print(f"\n📂 Cargando {n_samples} muestras de {filepath}...")
        df = pd.read_csv(filepath)
        df.columns = [col.lower() for col in df.columns]
        return df.tail(n_samples).reset_index(drop=True)

    def _generar_reporte_distribucion(self, probabilidades, bot_name):
        probs = np.array(probabilidades)
        
        print("\n" + "█"*60)
        print(f" 📊 RADIOGRAFÍA DEL CEREBRO: {bot_name}")
        print("█"*60)
        
        print(f"\n📈 ESTADÍSTICAS GLOBALES:")
        print(f" • Promedio de Confianza: {np.mean(probs):.2%}")
        print(f" • Mediana (Percentil 50): {np.median(probs):.2%}")
        print(f" • Confianza Máxima Vista: {np.max(probs):.2%}")
        print(f" • Confianza Mínima Vista: {np.min(probs):.2%}")
        
        print(f"\n🎯 DISTRIBUCIÓN DE UMBRALES (THRESHOLDS):")
        total_signals = len(probs)
        
        umbrales = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        for u in umbrales:
            cantidad = np.sum(probs >= u)
            porcentaje = (cantidad / total_signals) * 100
            
            # Formateo visual para decisiones rápidas
            if porcentaje > 20:   estado = "⚠️ Demasiadas señales (Ruido)"
            elif porcentaje > 5:  estado = "✅ Rango Saludable"
            elif porcentaje > 0.5: estado = "🎯 Francotirador (Pocas señales)"
            else:                 estado = "🧊 Mercado Congelado (Casi nunca opera)"
            
            print(f" > Umbral {u:.2f} : {cantidad:5} señales ({porcentaje:05.2f}%) -> {estado}")
            
        print("\n💡 RECOMENDACIÓN QD:")
        p95 = np.percentile(probs, 95)
        p99 = np.percentile(probs, 99)
        print(f" Para un bot activo (Top 5% señales): Usar Threshold = {p95:.4f}")
        print(f" Para un bot conservador (Top 1% señales): Usar Threshold = {p99:.4f}")
        print("█"*60 + "\n")

    def perfilar_bot2(self):
        df = self._cargar_datos(f"{self.data_folder}SOLUSDT.csv")
        
        # Carga de modelos
        hmm = joblib.load(f"{self.models_folder}hmm_bot2.pkl")
        scaler = joblib.load(f"{self.models_folder}scaler_bot2.pkl")
        lstm = load_model(f"{self.models_folder}lstm_bot2.keras")
        
        # 1. Feature Engineering (Idéntico a bot2_hmm_nn.py)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=20).std()
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
        df['tech_signal'] = (df['ema_fast'] > df['ema_slow']).astype(int)
        
        df_clean = df.dropna().copy()
        
        # 2. Inferencia HMM
        hmm_features = df_clean[['log_return', 'volatility', 'hl_spread']].values
        df_clean['hmm_state'] = hmm.predict(hmm_features)
        
        # 3. Preparar secuencias LSTM (Ventana de 60)
        seq_length = 60
        scaled_features = scaler.transform(df_clean[['log_return', 'volatility', 'hl_spread', 'tech_signal']])
        final_features = np.column_stack((scaled_features, df_clean['hmm_state'].values))
        
        # Creamos batches de 60 velas para inferencia rápida
        print("🧠 Procesando matriz tensorial (esto puede tardar unos segundos)...")
        X = []
        for i in range(len(final_features) - seq_length):
            X.append(final_features[i : i + seq_length])
        X = np.array(X)
        
        # 4. Predicción masiva
        predicciones = lstm.predict(X, batch_size=256, verbose=0)
        
        # El LSTM tira una probabilidad [0, 1]. La extraemos toda.
        confianzas = predicciones.flatten()
        
        self._generar_reporte_distribucion(confianzas, "Bot 2 (HMM + LSTM)")

    def perfilar_bot4(self):
        # 1. Cargar datos
        df = self._cargar_datos(f"{self.data_folder}SOLUSDT.csv") 
        
        # --- 🛡️ BLINDAJE DE DATOS (El Fix) ---
        print("🛡️ Forzando limpieza de tipos numéricos...")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        print("🧠 Cargando modelo RL y Scaler...")
        scaler = joblib.load(f"{self.models_folder}rl_scaler_bot4.pkl")
        model = load_model(f"{self.models_folder}rl_bot4.keras")
        
        print("⚙️ Aplicando Feature Engineering del Bot 4...")
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20'] = df['log_return'].rolling(20).std()
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        ema_9 = ta.ema(df['close'], length=9)
        ema_21 = ta.ema(df['close'], length=21)
        df['ema_9_dist'] = (df['close'] - ema_9) / df['close']
        df['ema_21_dist'] = (df['close'] - ema_21) / df['close']
        df['ema_cross_signal'] = (ema_9 > ema_21).astype(int)
        
        sma_50 = ta.sma(df['close'], length=50)
        sma_200 = ta.sma(df['close'], length=200)
        df['sma_50_dist'] = (df['close'] - sma_50) / df['close']
        df['sma_200_dist'] = (df['close'] - sma_200) / df['close']
        
        bbands = ta.bbands(df['close'], length=20, std=2)
        bbb_col = [col for col in bbands.columns if 'BBB' in col][0]
        df['bb_width'] = bbands[bbb_col]
        bbp_col = [col for col in bbands.columns if 'BBP' in col][0]
        df['bb_p'] = bbands[bbp_col]      
        
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['atr_14_pct'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']
        
        vol_mean = df['volume'].rolling(50).mean()
        vol_std = df['volume'].rolling(50).std()
        df['volume_normalized'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)
        
        # Respetando el nombre de variable que usaste en el train_bot4.py
        df['sentiment_score'] = 5.0 
        
        # --- 🚨 RADAR DE NANS MASIVOS ---
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean = df.dropna().copy()
        
        if df_clean.empty:
            print("\n🚨 ERROR CRÍTICO: Todas las velas se volvieron NaN (Vacías).")
            print("Radiografía del desastre (Cantidad de NaNs por columna):")
            print(df.isna().sum())
            return
            
        feature_cols = [
            'log_return', 'volatility_20', 'hl_spread', 
            'ema_9_dist', 'ema_21_dist', 'ema_cross_signal',
            'sma_50_dist', 'sma_200_dist',
            'bb_width', 'bb_p', 
            'rsi_14', 'atr_14_pct', 'volume_normalized', 'sentiment_score'
        ]
        
        features_array = df_clean[feature_cols].values
        
        print(f"🧠 Escalando y procesando matriz tensorial ({len(features_array)} velas útiles)...")
        state_scaled = scaler.transform(features_array)
        
        print("⚡ Predicción masiva de Q-Values...")
        predicciones = model.predict(state_scaled, batch_size=256, verbose=0)
        
        # predicciones[0] son las decisiones, predicciones[1] es el apalancamiento
        q_values_matrix = predicciones[0] 
        
        print("🧮 Convirtiendo Q-Values a Probabilidades (Softmax)...")
        confianzas = []
        for q_vals in q_values_matrix:
            action_idx = np.argmax(q_vals)
            exp_q = np.exp(q_vals - np.max(q_vals))
            probs = exp_q / exp_q.sum()
            confidence = probs[action_idx]
            confianzas.append(confidence)
            
        self._generar_reporte_distribucion(confianzas, "Bot 4 (RL Swing)")

    def perfilar_bot5(self):
        print("\n⚡ PERFILANDO FRANCOTIRADOR HFT (Bot 5)...")
        # 1. Cargar el CSV de Microestructura
        hft_file = f"{self.data_folder}hft_historical/HFT_SOLUSDT_LIVE_RECORDING.csv"
        df = self._cargar_datos(hft_file, n_samples=100000) # Cargar 100k ticks
        
        print("🧠 Cargando modelo HFT RL y Scaler...")
        scaler = joblib.load(f"{self.models_folder}rl_scaler_bot5.pkl")
        model = load_model(f"{self.models_folder}rl_bot5_hft.keras")
        
        # 2. Las 12 features mágicas de Claude
        feature_cols = [
            'imbalance', 'bid_wall_ratio', 'ask_wall_ratio',
            'dist_bid_wall', 'dist_ask_wall', 'spread',
            'delta_imbalance', 'delta_spread',
            'bid_wall_persistence', 'ask_wall_persistence',
            'sentiment', 'pos_encoded'
        ]
        
        # OJO ACÁ: Blindaje por si faltan datos en el CSV
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean = df.dropna(subset=feature_cols).copy()
        
        if df_clean.empty:
            print("\n🚨 ERROR CRÍTICO: El CSV de HFT está vacío o lleno de NaNs.")
            return

        features_array = df_clean[feature_cols].values
        
        print(f"🧠 Escalando matriz tensorial ({len(features_array)} ticks útiles)...")
        state_scaled = scaler.transform(features_array)
        
        print("⚡ Predicción masiva de Q-Values a velocidad luz...")
        # A diferencia del Bot 4 que devuelve 2 tensores (decisión y apalancamiento),
        # tu Bot 5 devuelve 1 solo tensor con las decisiones [LONG, SHORT, HOLD]
        predicciones = model.predict(state_scaled, batch_size=1024, verbose=0)
        
        print("🧮 Analizando Margen Crudo (Q-Value vs HOLD)...")
        confianzas = []
        for q_vals in predicciones:
            action_idx = np.argmax(q_vals)
            if action_idx != 2: 
                # La verdadera confianza es qué tan mejor es disparar vs no hacer nada
                margen_crudo = q_vals[action_idx] - q_vals[2]
                confianzas.append(margen_crudo)
                
        if not confianzas:
            print("\n🧊 EL BOT ESTÁ CONGELADO: En 100,000 ticks, la IA decidió hacer HOLD el 100% de las veces.")
            print("El costo de las comisiones en el entrenamiento fue demasiado agresivo. Bajale el 'fee_penalty'.")
            return
            
        self._generar_reporte_distribucion(confianzas, "Bot 5 (HFT RL Microestructura)")

    def ejecutar(self):
        opcion = self._menu_seleccion()
        if opcion == 2:
            self.perfilar_bot2()
        elif opcion == 4:
            self.perfilar_bot4()
        elif opcion == 5:
            self.perfilar_bot5()
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    profiler = AIModelProfiler()
    profiler.ejecutar()
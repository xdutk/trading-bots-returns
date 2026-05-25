import os
import glob
import numpy as np
import pandas as pd
import joblib
import logging
import tensorflow as tf

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TRAINING] - %(message)s')

class ModelTrainer:
    """
    Pipeline offline para entrenar el HMM y el LSTM usando datos históricos.
    Implementa tf.data.Dataset (Generadores) para procesar infinitos CSVs 
    sin saturar la memoria RAM.
    """
    
    def __init__(self, data_folder="data/", models_folder="models/", seq_length=60, target_horizon=5):
        self.data_folder = data_folder
        self.models_folder = models_folder
        self.seq_length = seq_length
        self.target_horizon = target_horizon 
        
        os.makedirs(self.models_folder, exist_ok=True)
        self.scaler = StandardScaler()
        self.hmm_model = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Features matemáticas
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=20).std()
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
        df['tech_signal'] = (df['ema_fast'] > df['ema_slow']).astype(int)
        
        # --- EL NUEVO TARGET (PREDICCIÓN DIRECCIONAL PURA) ---
        # Le enseñamos a predecir si el precio va a subir (1) o bajar (0)
        # en las próximas 5 velas (target_horizon), pero ignorando la señal técnica.
        
        future_price = df['close'].shift(-self.target_horizon)
        # Retorno a futuro. Si es positivo (> 0), el precio subió.
        df['target'] = (future_price > df['close']).astype(int) 
        
        df.dropna(inplace=True)
        return df

    def _sequence_generator(self, file_list):
        """Generador Yield: Lee un CSV, procesa, emite secuencias y lo borra de la RAM."""
        for file in file_list:
            try:
                df = pd.read_csv(file)
                df = self._engineer_features(df)
                
                # 1. HMM
                hmm_feat = df[['log_return', 'volatility', 'hl_spread']].values
                df['hmm_state'] = self.hmm_model.predict(hmm_feat)
                
                # 2. Scaler
                scaled = self.scaler.transform(df[['log_return', 'volatility', 'hl_spread', 'tech_signal']])
                final_features = np.column_stack((scaled, df['hmm_state'].values))
                targets = df['target'].values
                
                # 3. Emitir secuencias una por una a la GPU
                for i in range(len(final_features) - self.seq_length):
                    yield final_features[i : i + self.seq_length], targets[i + self.seq_length]
                    
            except Exception as e:
                logging.error(f"Error en generador leyendo {file}: {e}")
                continue

    def train_pipeline(self):
        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        if not all_files:
            logging.error("No se encontraron archivos CSV en la carpeta data/")
            return
            
        logging.info(f"Iniciando pipeline en modo STREAMING con {len(all_files)} archivos CSV...")
        
        # --- PASO 1: Ajustar Scaler y HMM con una muestra (Max 5 archivos para cuidar la RAM) ---
        sample_files = all_files[:min(5, len(all_files))]
        logging.info(f"Calibrando modelos base (HMM y Scaler) usando una muestra de {len(sample_files)} archivos...")
        
        df_list = []
        for file in sample_files:
            df = self._engineer_features(pd.read_csv(file))
            df_list.append(df)
            
        sample_data = pd.concat(df_list, ignore_index=True)
        
        # Entrenar HMM
        self.hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        self.hmm_model.fit(sample_data[['log_return', 'volatility', 'hl_spread']].values)
        joblib.dump(self.hmm_model, os.path.join(self.models_folder, 'hmm_bot2.pkl'))
        
        # Entrenar Scaler
        self.scaler.fit(sample_data[['log_return', 'volatility', 'hl_spread', 'tech_signal']])
        joblib.dump(self.scaler, os.path.join(self.models_folder, 'scaler_bot2.pkl'))
        
        # Liberamos la RAM de la muestra
        del df_list, sample_data

        # --- PASO 2: Separar archivos de Train y Validation ---
        split_idx = int(len(all_files) * 0.8)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        # --- PASO 3: Crear los Datasets de TensorFlow ---
        output_signature = (
            tf.TensorSpec(shape=(self.seq_length, 5), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
        
        train_dataset = tf.data.Dataset.from_generator(
            lambda: self._sequence_generator(train_files),
            output_signature=output_signature
        )
        val_dataset = tf.data.Dataset.from_generator(
            lambda: self._sequence_generator(val_files),
            output_signature=output_signature
        )

        # Magia de optimización: lotes de 2048, pre-cargando el siguiente en segundo plano
        train_dataset = train_dataset.batch(2048).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(2048).prefetch(tf.data.AUTOTUNE)

        # --- PASO 4: Entrenar LSTM ---
        logging.info("Entrenando LSTM en bloque... (El progreso se mostrará por lotes)")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.seq_length, 5)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid') 
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(
            train_dataset, 
            validation_data=val_dataset, 
            epochs=20, 
            steps_per_epoch=3000,
            validation_steps=500,
            callbacks=[early_stop]
        )
        
        model.save(os.path.join(self.models_folder, 'lstm_bot2.keras'))
        logging.info("LSTM entrenado y guardado exitosamente.")

if __name__ == "__main__":
    trainer = ModelTrainer(data_folder="data", models_folder="models")
    # Ya no limitamos la cantidad de archivos, el generador maneja infinitos datos.
    trainer.train_pipeline()
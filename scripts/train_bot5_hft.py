import os
import glob
import random
import logging
import numpy as np
import pandas as pd
import joblib
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [HFT TRAINER] - %(message)s')

class HFTReplayBuffer:
    def __init__(self, maxlen=100000):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class HFTRLTrainer:
    def __init__(self, data_folder="data/hft_historical/", models_folder="models/"):
        self.data_folder = data_folder
        self.models_folder = models_folder
        os.makedirs(self.models_folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.buffer = HFTReplayBuffer()
        
        self.gamma = 0.90 # Descuento menor porque el HFT vive en el presente inmediato
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999 # Decaimiento más lento por la cantidad de ticks

    def build_model(self, input_dim=12) -> Model:
        """La Red Neuronal del Sniper: Rápida y directa."""
        inputs = Input(shape=(input_dim,))
        x = Dense(256, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        
        # Salida: Q-Values para [LONG, SHORT, HOLD]
        action_out = Dense(3, activation='linear', name='action_head')(x)
        
        model = Model(inputs=inputs, outputs=action_out)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        return model

    def train_hft_agent(self):
        # 1. Buscar CSVs históricos. 
        # NOTA: Deben tener las 12 features pre-calculadas y la columna 'mid_price'
        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        if not all_files:
            logging.error(f"❌ No hay CSVs de Order Book en {self.data_folder}.")
            logging.error("Conseguí al menos 1 mes de data L2 de Binance Vision.")
            return
            
        logging.info(f"Iniciando Entrenamiento HFT con {len(all_files)} archivos...")

        feature_cols = [
            'imbalance', 'bid_wall_ratio', 'ask_wall_ratio',
            'dist_bid_wall', 'dist_ask_wall', 'spread',
            'delta_imbalance', 'delta_spread',
            'bid_wall_persistence', 'ask_wall_persistence',
            'sentiment', 'pos_encoded'
        ]

        # --- FASE 1: Calibrar el Scaler ---
        logging.info("Calibrando Scaler para microestructura...")
        sample_df = pd.read_csv(all_files[0], nrows=50000) # Leer solo una muestra para no matar la RAM
        self.scaler.fit(sample_df[feature_cols])
        joblib.dump(self.scaler, os.path.join(self.models_folder, 'rl_scaler_bot5.pkl'))

        # --- FASE 2: Entorno de Simulación Tick-by-Tick ---
        self.model = self.build_model()
        batch_size = 128
        
        for file_idx, f in enumerate(all_files):
            logging.info(f"Procesando fragmento {file_idx+1}/{len(all_files)}: {os.path.basename(f)}")
            
            # Leer en chunks para no desbordar memoria
            chunk_iterator = pd.read_csv(f, chunksize=100000)
            
            for chunk in chunk_iterator:
                df = chunk.dropna().reset_index(drop=True)
                scaled_states = self.scaler.transform(df[feature_cols])
                prices = df['mid_price'].values
                
                # Simular experiencia
                # Simular experiencia
                for i in range(len(df) - 5000): # <--- FIX 1: Ampliamos el horizonte a 5000 ticks
                    # Usamos .copy() para poder modificar el estado sin alterar el array original
                    state = scaled_states[i].copy()
                    next_state = scaled_states[i+1].copy()
                    
                    fake_pos = random.choice([-1.0, 0.0, 1.0])
                    state[11] = fake_pos
                    next_state[11] = fake_pos
                    
                    action = random.choice([0, 1, 2])
                    reward = 0.0
                    
                    if action != 2: # Si no es HOLD
                        entry_price = prices[i]
                        future_prices = prices[i+1 : i+5000] # <--- FIX 2: Visión a mediano plazo
                        
                        # --- FIX 3: NUEVOS MÁRGENES DE SCALPER ---
                        sl_pct = 0.003  # Stop Loss del 0.30% 
                        tp_pct = 0.008  # Take Profit del 0.80% (Cubre comisiones y deja ganancia fuerte)
                        
                        fee_penalty = 0.08 # Costo fijo de Binance
                        
                        if action == 0: # LONG
                            if min(future_prices) <= entry_price * (1 - sl_pct):
                                reward = -2.0 - fee_penalty 
                            elif max(future_prices) >= entry_price * (1 + tp_pct):
                                reward = 3.0 - fee_penalty  
                            else:
                                pnl = (future_prices[-1] - entry_price) / entry_price * 100
                                reward = np.clip(pnl, -2.0, 3.0) - fee_penalty
                                
                        elif action == 1: # SHORT
                            if max(future_prices) >= entry_price * (1 + sl_pct):
                                reward = -2.0 - fee_penalty
                            elif min(future_prices) <= entry_price * (1 - tp_pct):
                                reward = 3.0 - fee_penalty
                            else:
                                pnl = (entry_price - future_prices[-1]) / entry_price * 100
                                reward = np.clip(pnl, -2.0, 3.0) - fee_penalty
                                
                    elif action == 2: # HOLD
                        reward = 0.0 # HACER HOLD ES 100% GRATIS
                                
                    self.buffer.add(state, action, reward, next_state, done=False)

                # Entrenar la red neuronal
                if len(self.buffer) >= batch_size:
                    for _ in range(10): # Repasos por chunk
                        batch = self.buffer.sample(batch_size)
                        states_b = np.array([e[0] for e in batch])
                        actions_b = np.array([e[1] for e in batch])
                        rewards_b = np.array([e[2] for e in batch])
                        next_states_b = np.array([e[3] for e in batch])
                        
                        # ---> OPTIMIZACIÓN DE HARDWARE: Inferencia sin .predict() <---
                        next_q_tensor = self.model(next_states_b, training=False)
                        target_q_tensor = self.model(states_b, training=False)
                        
                        next_q = next_q_tensor.numpy()
                        target_q = target_q_tensor.numpy()
                        # --------------------------------------------------------------
                        
                        for idx, (act, rev) in enumerate(zip(actions_b, rewards_b)):
                            if act != 2:
                                # FIX: Si disparó (0 o 1), el 'rev' ya contiene todo el resultado del trade.
                                # No le sumamos el futuro para evitar que los números exploten al infinito.
                                target_q[idx][act] = rev
                            else:
                                # FIX: Si hizo HOLD (2), su premio es 0, pero conserva la esperanza de
                                # encontrar un buen trade en el futuro.
                                target_q[idx][act] = rev + self.gamma * np.max(next_q[idx])
                                
                        # Entrenamos el batch
                        self.model.fit(states_b, target_q, verbose=0, batch_size=batch_size)

        # --- FASE 3: Guardado ---
        self.model.save(os.path.join(self.models_folder, 'rl_bot5_hft.keras'))
        logging.info("✅ ¡Cerebro HFT pre-entrenado y guardado en disco! Listo para la guerra.")

if __name__ == "__main__":
    trainer = HFTRLTrainer()
    trainer.train_hft_agent()
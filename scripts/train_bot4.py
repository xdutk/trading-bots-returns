import os
import glob
import random
import logging
import numpy as np
import pandas as pd
import joblib
import pandas_ta as ta
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TRAINING RL] - %(message)s')

class ReplayBuffer:
    def __init__(self, maxlen=50000):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class RLTrainer:
    def __init__(self, data_folder="data/", models_folder="models/"):
        self.data_folder = data_folder
        self.models_folder = models_folder
        os.makedirs(self.models_folder, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.buffer = ReplayBuffer(maxlen=50000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
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
        
        df['sentiment_score'] = 5.0
        
        df.dropna(inplace=True)
        return df

    def build_model(self, input_dim: int) -> Model:
        inputs = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        
        action_out = Dense(3, activation='linear', name='action_head')(x)
        leverage_out = Dense(1, activation='linear', name='leverage_head')(x)
        
        model = Model(inputs=inputs, outputs=[action_out, leverage_out])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss={'action_head': 'mse', 'leverage_head': 'mse'})
        return model

    def _train_step(self, batch):
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        
        next_q_values, _ = self.model.predict(next_states, verbose=0)
        q_targets, lev_targets = self.model.predict(states, verbose=0)
        
        lev_learning_rate = 1.5 
        
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            if action != 2: 
                q_targets[i][action] = reward + self.gamma * np.max(next_q_values[i])
                lev_targets[i][0] = lev_targets[i][0] + (reward * lev_learning_rate)
                
        lev_targets = np.clip(lev_targets, 1, 25)
        
        self.model.fit(
            states,
            {'action_head': q_targets, 'leverage_head': lev_targets},
            verbose=0,
            batch_size=len(batch)
        )

    def populate_buffer_and_train(self):
        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        if not all_files:
            logging.error("No se encontraron CSVs en data/")
            return
            
        logging.info(f"Iniciando pipeline RL STREAMING con {len(all_files)} archivos CSV...")

        feature_cols = [
            'log_return', 'volatility_20', 'hl_spread', 
            'ema_9_dist', 'ema_21_dist', 'ema_cross_signal',
            'sma_50_dist', 'sma_200_dist',
            'bb_width', 'bb_p', 
            'rsi_14', 'atr_14_pct', 'volume_normalized', 'sentiment_score'
        ]

        # --- PASO 1: Calibrar Scaler con Muestra ---
        sample_files = all_files[:min(5, len(all_files))]
        logging.info("Calibrando Scaler con muestra inicial...")
        df_list = []
        for f in sample_files:
            df_list.append(self._engineer_features(pd.read_csv(f)))
        
        train_df = pd.concat(df_list, ignore_index=True)
        self.scaler.fit(train_df[feature_cols])
        joblib.dump(self.scaler, os.path.join(self.models_folder, 'rl_scaler_bot4.pkl'))
        del df_list, train_df # Liberamos RAM

        # --- PASO 2: Inicializar Red Neuronal ---
        self.model = self.build_model(len(feature_cols))
        batch_size = 64
        
        # --- PASO 3: Streaming y Entrenamiento Continuo ---
        logging.info("Iniciando lectura y entrenamiento archivo por archivo...")
        for idx, f in enumerate(all_files):
            try:
                df = pd.read_csv(f)
                df = self._engineer_features(df)
                scaled_features = self.scaler.transform(df[feature_cols])
                
                # Cargar experiencias de ESTE archivo al buffer
                for i in range(len(scaled_features) - 5):
                    state = scaled_features[i]
                    next_state = scaled_features[i+1]
                    action = random.choice([0, 1, 2]) 
                    future_return = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
                    
                    if action == 2:
                        reward = 0.0
                    elif action == 0:
                        reward = future_return
                        if min(df['low'].iloc[i:i+5]) <= df['close'].iloc[i] * 0.985:
                            reward = -2.0
                    else:
                        reward = -future_return
                        if max(df['high'].iloc[i:i+5]) >= df['close'].iloc[i] * 1.015:
                            reward = -2.0
                            
                    self.buffer.add(state, action, reward, next_state, done=False)
                
                # Entrenar la red si el buffer tiene suficiente tamaño
                if len(self.buffer) >= batch_size:
                    # Entrenamos 50 pasos por cada archivo que procesamos
                    for step in range(50):
                        batch = self.buffer.sample(batch_size)
                        self._train_step(batch)
                        
                        if self.epsilon > self.epsilon_min:
                            self.epsilon *= self.epsilon_decay
                
                logging.info(f"Archivo {idx+1}/{len(all_files)} procesado y entrenado. Epsilon actual: {self.epsilon:.4f}")
                
            except Exception as e:
                logging.error(f"Error procesando {f}: {e}")

        # --- PASO 4: Guardado Final ---
        joblib.dump(self.buffer, os.path.join(self.models_folder, 'replay_buffer.pkl'))
        self.model.save(os.path.join(self.models_folder, 'rl_bot4.keras'))
        self.model.save(os.path.join(self.models_folder, 'rl_bot4_BACKUP.keras'))
        logging.info("✅ Entrenamiento Finalizado. Modelos RL listos para Producción.")

if __name__ == "__main__":
    trainer = RLTrainer()
    trainer.populate_buffer_and_train()
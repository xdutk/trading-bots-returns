# 🛰️ QuantProtocol

QuantProtocol es un sistema de trading algorítmico cuantitativo, asíncrono y multi-agente diseñado para operar en Binance Futures. Implementa una arquitectura modular donde múltiples motores tácticos (Bots) operan en paralelo, gestionados por un motor de riesgo centralizado, ejecutores abstractos y alimentados por modelos de Machine Learning y Deep Reinforcement Learning continuo.

## 🏗️ Arquitectura del Sistema
El sistema está diseñado bajo el patrón Event-Driven (Orientado a Eventos). Un único hilo de conexión asíncrona (WebSocket) alimenta a todos los motores simultáneamente, garantizando latencia ultrabaja y cero bloqueos de I/O.

### 1. El Núcleo Core (`core/`)
La espina dorsal del protocolo. Separa la lógica matemática de la gestión operativa:
* **DataFeed & OrderBookFeed:** Administran el WebSocket de Binance. Mantienen buffers en memoria de velas L1 y microestructura L2 (Order Book), despachando las actualizaciones a los bots asíncronamente.
* **RiskEngine:** El Global Exposure Manager. Audita las intenciones de los bots antes de permitirles abrir una posición, ajustando tamaños según rachas y exposición direccional.
* **CapitalManager:** Gestiona el capital de forma aislada para cada bot, calculando el *position sizing* dinámico.
* **OrderExecutor:** El puente abstracto con Binance. Permite alternar entre MODO PAPER (Simulación pura) y LIVE (Ejecución real).
* **BotLogger:** Sistema de persistencia de estado atómico en CSV para recuperación (*rehidratación*) ante reinicios.

### 2. Los Motores Tácticos (`bots/`)
QuantProtocol despliega 5 escuadrones cuantitativos independientes:
* **Bot 1 (FRPV - Fractal Reversion):** Estrategia multi-timeframe (5m y 1H). Utiliza medias adaptativas (KAMA) y Regresión Lineal para ineficiencias de corto plazo.
* **Bot 2 (HMM + LSTM):** Inteligencia Artificial Secuencial. Un *Hidden Markov Model* detecta el régimen de mercado, y una red LSTM predice la probabilidad de éxito de cruces técnicos.
* **Bot 3 (Adaptive Cross):** Motor matemático dinámico que ajusta períodos de medias móviles según la varianza normalizada del mercado (filtro de histéresis).
* **Bot 4 (Reinforcement Learning - Swing):** Agente DQN que decide acciones y apalancamiento dinámico (x1-x25) aprendiendo del mercado continuo.
* **Bot 5 (HFT Hybrid Scalper):** El francotirador de Alta Frecuencia. Analiza los desbalances del *Order Book* L2 y la persistencia de muros institucionales usando un modelo RL para ejecutar *scalps* milimétricos, filtrando ruido mediante análisis de margen crudo.

## 📂 Estructura de Directorios

```text
QuantProtocol/
├── core/                   # Motores de ejecución, feeds, riesgo y capital
├── bots/                   # Lógica táctica de los 5 escuadrones
├── oracles/                # Daemon de sentimiento macro (Binance L/S Ratio)
├── dashboard/              # Mission Control (Streamlit UI)
├── scripts/                # Pipelines de entrenamiento offline (LSTM / RL)
├── data_pipelines/         # Scrapers y procesadores de data histórica
├── tests/                  # Pruebas unitarias de la arquitectura
├── data/                   # (Ignorado en git) CSVs de mercado L1 y L2
├── logs/                   # (Ignorado en git) Telemetría y estado de bots
├── models/                 # (Ignorado en git) Pesos .keras y scalers .pkl
├── backtest_results/       # (Ignorado en git) Resultados de simulaciones
│
├── main.py                 # Orquestador Central (Inicia el sistema)
├── backtester.py           # Simulador Offline de precisión para estrategias
├── ai_profiler.py          # Auditor forense de Q-Values y calibrador de umbrales
├── audit_data.py           # Herramienta de chequeo de sanidad para data L2 HFT
├── quant_analyzer.py       # Analizador cuantitativo de PnL y métricas
├── panic_sell_all.py       # KILL SWITCH: Cierra todo en Binance de emergencia
├── sentiment_score.txt     # Caché atómico de temperatura de mercado
├── .env                    # Variables de entorno y API Keys
└── .gitignore              # Filtros de exclusión para GitHub
🚀 Instalación y Setup
Clonar entorno virtual:
```
```Bash
git clone [https://github.com/xdutk/QuantProtocol.git](https://github.com/xdutk/QuantProtocol.git)
cd QuantProtocol
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```
# Instalar dependencias:

```Bash
pip install pandas numpy pandas-ta scikit-learn hmmlearn tensorflow websockets python-binance streamlit psutil google-genai requests python-dotenv feedparser
# Opcional (Recomendado): pip install tensorflow[and-cuda]
Variables de Entorno (.env):
```
```Fragmento de código
BINANCE_API_KEY="tu_api_key"
BINANCE_API_SECRET="tu_api_secret"
BOT1_LIVE="true" # false = Ejecución Real
BOT5_LIVE="true"
```
# 🧠 Flujo de Ejecución Cuantitativa (Runbook)

Entrenamiento (scripts/): Entrenar modelos de IA offline (train_bot4.py, train_bot5_hft.py).

Auditoría de IA (ai_profiler.py): Analizar los Q-Values resultantes para extraer el "Threshold" de confianza óptimo (Filtro Top 1% / 5%).

Simulación (backtester.py): Validar la estrategia, los umbrales y las comisiones en un entorno offline cerrado.

Despliegue (main.py):

Terminal 1: python -m oracles.sentiment_daemon

Terminal 2: streamlit run dashboard/app.py

Terminal 3: python main.py

#🛑 Protocolos de Seguridad

Graceful Shutdown: Al presionar Ctrl+C, el sistema detiene los websockets y vuelca la memoria de la IA de forma segura (Timeout de 10s -> SIGTERM).

Circuit Breaker: Suspensión automática del bot tras 3 Stop Loss consecutivos.

KILL SWITCH (panic_sell_all.py): Script aislado que liquida todas las posiciones a Market y asesina los procesos del orquestador.

# ⚠️ Disclaimer: Software con fines de investigación cuantitativa. El trading algorítmico apalancado conlleva riesgo de liquidación. Usar MODO PAPER para validación.

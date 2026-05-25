import os
import glob
import json
import pandas as pd
import numpy as np

# Silenciamos advertencias de Pandas para consola limpia
pd.options.mode.chained_assignment = None

class QuantAnalyzerPro:
    def __init__(self, folder="backtest_results"):
        self.folder = folder
        self.database = pd.DataFrame()

    def _cargar_base_datos(self):
        """Lee todos los JSONs y arma una base de datos maestra para comparar."""
        archivos = glob.glob(os.path.join(self.folder, "*.json"))
        data_list = []
        
        for arch in archivos:
            try:
                with open(arch, 'r') as f:
                    data = json.load(f)
                    
                # ✅ CORRECCIÓN: Usamos la clave exacta de tu JSON
                config = data.get('configuracion_usada', {})
                
                resumen = {
                    'archivo': os.path.basename(arch),
                    'fecha': data.get('fecha_test', 'N/A'),
                    'bot': data.get('bot', 'N/A'),
                    'ticker': data.get('ticker', 'N/A'),
                    'balance': data.get('balance_final', 0),
                    'trades': data.get('total_trades', 0),
                    'win_rate': data.get('win_rate', 0),
                    'modo_rev': config.get('reversal_mode', 'N/A'),
                    'sl_pct': config.get('hard_sl', 'N/A'),
                    'trail_act': config.get('trailing_activation', 'N/A'),
                    'racha_perdida': data.get('max_racha_perdedora', 0),
                    'avg_win': data.get('avg_win_pct', 0),
                    'avg_loss': data.get('avg_loss_pct', 0)
                }
                
                # Cálculos avanzados para la tabla
                df_trades = pd.DataFrame(data.get('trades', []))
                if not df_trades.empty and 'resultado_usd' in df_trades.columns:
                    df_trades['resultado_usd'] = df_trades['resultado_usd'].astype(float)
                    g_profit = df_trades[df_trades['resultado_usd'] > 0]['resultado_usd'].sum()
                    g_loss = abs(df_trades[df_trades['resultado_usd'] < 0]['resultado_usd'].sum())
                    
                    resumen['profit_factor'] = round(g_profit / g_loss, 2) if g_loss != 0 else np.inf
                    
                    # Max Drawdown
                    balance_acumulado = df_trades['resultado_usd'].cumsum()
                    resumen['max_dd'] = round((balance_acumulado.cummax() - balance_acumulado).max(), 2)
                    
                    # Esperanza Matemática
                    wr_dec = resumen['win_rate'] / 100
                    resumen['esperanza'] = round((wr_dec * resumen['avg_win']) - ((1 - wr_dec) * abs(resumen['avg_loss'])), 2)
                else:
                    resumen['profit_factor'] = 0.0
                    resumen['max_dd'] = 0.0
                    resumen['esperanza'] = 0.0

                data_list.append(resumen)
            except Exception as e:
                continue
                
        if data_list:
            self.database = pd.DataFrame(data_list)
            self.database.sort_values(by='balance', ascending=False, inplace=True)

    def seleccionar_reporte(self):
        archivos = sorted(glob.glob(os.path.join(self.folder, "*.json")), key=os.path.getmtime, reverse=True)
        if not archivos:
            print(f"❌ No hay reportes en '{self.folder}'.")
            return None
        
        print("\n📂 ÚLTIMOS 10 REPORTES GENERADOS:")
        for i, archivo in enumerate(archivos[:10]):
            nombre = os.path.basename(archivo)
            print(f" [{i}] {nombre}")
        
        try:
            opcion = input("\n🔢 Elegí el número del reporte a auditar (o Enter para el último): ")
            if opcion == "": return archivos[0]
            return archivos[int(opcion)]
        except:
            return archivos[0]

    def mostrar_comparativa(self, target_bot, target_ticker):
        if self.database.empty:
            return

        df_filtro = self.database[(self.database['bot'] == target_bot) & (self.database['ticker'] == target_ticker)]
        if df_filtro.empty:
            return

        print("\n" + "="*105)
        print(f" 🏆 LEADERBOARD CUANTITATIVO | {target_bot.upper()} - {target_ticker}")
        print("="*105)
        
        # Seleccionamos y renombramos columnas para el panel de control
        tabla = df_filtro[['archivo', 'balance', 'win_rate', 'profit_factor', 'esperanza', 'max_dd', 'racha_perdida', 'modo_rev', 'sl_pct']]
        tabla.columns = ['Archivo', 'Balance ($)', 'WinRate (%)', 'Prof.Fact', 'Esperanza (%)', 'Max DD ($)', 'Peor Racha', 'Modo', 'StopLoss']
        
        # Formateo visual
        tabla['Archivo'] = tabla['Archivo'].apply(lambda x: x[-17:-5] if len(x) > 17 else x)
        tabla['Balance ($)'] = tabla['Balance ($)'].apply(lambda x: f"{x:.2f}")
        tabla['WinRate (%)'] = tabla['WinRate (%)'].apply(lambda x: f"{x:.1f}")
        
        print(tabla.head(10).to_string(index=False))
        print("="*105 + "\n")

    def analizar(self):
        self._cargar_base_datos()
        
        filepath = self.seleccionar_reporte()
        if not filepath: return

        with open(filepath, 'r') as f:
            data = json.load(f)

        bot_id = data.get('bot', 'Desconocido')
        ticker = data.get('ticker', 'Desconocido')

        self.mostrar_comparativa(bot_id, ticker)

        df = pd.DataFrame(data.get('trades', []))
        if df.empty:
            print("⚠️ El reporte seleccionado no tiene operaciones cerradas.")
            return

        df['resultado_usd'] = df['resultado_usd'].astype(float)
        df['es_ganador'] = df['resultado_usd'] > 0
        df['balance_acumulado'] = df['resultado_usd'].cumsum()

        # Re-cálculos para el reporte individual
        gross_profit = df[df['es_ganador']]['resultado_usd'].sum()
        gross_loss = abs(df[~df['es_ganador']]['resultado_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        max_drawdown = (df['balance_acumulado'].cummax() - df['balance_acumulado']).max()
        
        win_rate = data.get('win_rate', 0)
        avg_win = data.get('avg_win_pct', 0)
        avg_loss = abs(data.get('avg_loss_pct', 0))
        
        wr_dec = win_rate / 100
        expectancy = (wr_dec * avg_win) - ((1 - wr_dec) * avg_loss)
        
        # Criterio de Kelly
        risk_reward = avg_win / avg_loss if avg_loss != 0 else 0
        kelly_pct = (wr_dec - ((1 - wr_dec) / risk_reward)) * 100 if risk_reward > 0 else 0

        # Integridad
        df['distancia_velas'] = ((df['timestamp'] - df['timestamp'].shift(1)) / 300000).fillna(0) # 300,000 ms = 5 min
        trades_flash = df[df['distancia_velas'] <= 1]
        duracion_promedio = df['distancia_velas'].mean()

        print("█"*80)
        print(f" 🔬 RADIOGRAFÍA TÉCNICA: {bot_id} | {ticker} | {os.path.basename(filepath)}")
        print("█"*80)

        config = data.get('configuracion_usada', {})
        if config:
            print(f"\n⚙️ CONFIGURACIÓN APLICADA:")
            print(f" • Modo Reversal:         {config.get('reversal_mode', 'N/A')}")
            print(f" • Hard Stop Loss:        {config.get('hard_sl', 'N/A')}")
            print(f" • Activación Trailing:   {config.get('trailing_activation', 'N/A')}")
            print(f" • Distancia Trailing:    {config.get('trailing_distance', 'N/A')}")
            print(f" • Muro 4hs por SL:       {'Activado' if config.get('cooldown_4h_active') else 'Desactivado'}")

        print(f"\n📈 METRICAS CORE DE RENTABILIDAD:")
        print(f" • Balance Final:         ${data.get('balance_final', 0):.2f}")
        print(f" • Total Operaciones:     {data.get('total_trades', 0)}")
        print(f" • Profit Factor:         {profit_factor:.2f}")
        print(f" • Win Rate:              {win_rate:.2f}%")
        print(f" • Risk/Reward Ratio:     1 : {risk_reward:.2f}")
        print(f" • Esperanza por Trade:   {expectancy:.2f}%")
        print(f" • Criterio de Kelly:     {kelly_pct:.2f}% (Riesgo sugerido)")

        print(f"\n🛡️ METRICAS DE RIESGO Y DIBUJO:")
        print(f" • Max Drawdown:          -${max_drawdown:.2f}")
        print(f" • Peor Racha Perdedora:  {data.get('max_racha_perdedora', 0)} trades seguidos")
        print(f" • Mejor Racha Ganadora:  {data.get('max_racha_ganadora', 0)} trades seguidos")
        
        duplicados = df.duplicated(subset=['timestamp']).sum()
        print(f"\n🔍 DIAGNÓSTICO DE INTEGRIDAD:")
        print(f" • Trades duplicados:     {duplicados} " + ("(⚠️ Error)" if duplicados > 0 else "(✅ OK)"))
        print(f" • Trades Flash (<=1v):   {len(trades_flash)} " + ("(⚠️ Jitter)" if len(trades_flash) > (len(df)*0.1) else "(✅ Estabilidad OK)"))
        print(f" • Frecuencia Promedio:   {duracion_promedio:.1f} velas entre operaciones")
        
        print("\n" + "█"*80 + "\n")

if __name__ == "__main__":
    analista = QuantAnalyzerPro()
    analista.analizar()
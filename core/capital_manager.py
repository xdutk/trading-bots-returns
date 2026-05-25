class CapitalManager:
    """
    Gestiona el saldo, el tamaño de posición por bloques y la máquina de estados 
    del apalancamiento dinámico para un bot individual en QuantProtocol.
    """
    
    def __init__(self, initial_capital=1250.0, block_size=250.0, 
                 risk_pct=0.05, sl_threshold_pct=-0.5):
        self.current_capital = initial_capital
        self.block_size = block_size
        self.risk_pct = risk_pct
        self.sl_threshold_pct = sl_threshold_pct
        self.leverage = 3  # Estado inicial
        self.consecutive_sl = 0  # Contador para el Circuit Breaker

    def get_operative_capital(self) -> float:
        """
        Calcula el capital base según el sistema de escalones.
        Ej: Balance de 600 -> Operativo de 500.
        """
        return (self.current_capital // self.block_size) * self.block_size

    def is_margin_halted(self) -> bool:
        """
        Verifica si el capital cayó por debajo del bloque mínimo.
        """
        return self.current_capital < self.block_size

    def get_position_size(self) -> float:
        """
        Retorna el tamaño en USD que se debe usar en la próxima orden.
        """
        if self.is_margin_halted():
            return 0.0
        return self.get_operative_capital() * self.risk_pct

    def register_trade_result(self, pnl_usd: float, pnl_pct: float, atr_safe: bool = True):
        self.current_capital += pnl_usd
        gain_threshold = 7.0  # Umbral de victoria real
        
        # 1. Actualizar Circuit Breaker
        if pnl_pct <= self.sl_threshold_pct:
            self.consecutive_sl += 1
        elif pnl_pct >= gain_threshold:
            self.consecutive_sl = 0

        # 2. 🧠 MÁQUINA DE ESTADOS CON PISO EN x10
        prev_lev = self.leverage

        # --- CASO A: GANANCIA REAL (>= 7%) ---
        if pnl_pct >= gain_threshold:
            if prev_lev == 3: self.leverage = 30
            elif prev_lev == 30: self.leverage = 25
            elif prev_lev == 10: self.leverage = 25
            elif prev_lev == 25: self.leverage = 20
            elif prev_lev == 20: self.leverage = 20

        # --- CASO B: ZONA NEUTRA (0 < PnL < 7%) ---
        elif 0 < pnl_pct < gain_threshold:
            # Si venís de arriba (30, 25, 20), bajás al piso de x10
            # Si ya estás en x10, te mantenés ahí (el "piso")
            if prev_lev in [30, 25, 20, 10]:
                self.leverage = 10
            else:
                self.leverage = 3 # Si estás en x3 y es neutra, te quedás en x3

        # --- CASO C: PÉRDIDA (PnL <= 0) ---
        else:
            if prev_lev == 30:
                self.leverage = 10 # Única excepción: pérdida en x30 baja a x10
            else:
                self.leverage = 3  # Cualquier otra pérdida resetea a x3

    def is_circuit_breaker_active(self) -> bool:
        """
        Retorna True si el bot acumuló 4 o más SL consecutivos.
        """
        return self.consecutive_sl >= 4

    def get_state(self) -> dict:
        """
        Retorna el estado actual para inyectar en el logger (CSV).
        """
        return {
            "balance": self.current_capital,
            "operative_capital": self.get_operative_capital(),
            "position_size": self.get_position_size(),
            "leverage": self.leverage,
            "consecutive_sl": self.consecutive_sl,
            "margin_halted": self.is_margin_halted(),
            "circuit_breaker": self.is_circuit_breaker_active()
        }
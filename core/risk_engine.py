from core.capital_manager import CapitalManager

class RiskEngine:
    """
    Orquesta las reglas de riesgo globales del sistema QuantProtocol.
    Opera sobre todos los CapitalManagers en paralelo.
    """
    
    def __init__(self, managers: dict[str, CapitalManager], priority_order: list[str] = None, backtest_mode: bool = False):
        self.managers = managers
        self.priority_order = priority_order or list(managers.keys())
        self.backtest_mode = backtest_mode
        
        # Validación defensiva: fallar rápido si hay inconsistencias en los IDs
        unknown = set(self.priority_order) - set(self.managers.keys())
        if unknown:
            raise ValueError(f"priority_order contiene bots no registrados: {unknown}")

    def check_circuit_breaker(self, bot_id: str) -> bool:
        """
        Retorna True si el bot debe estar en pausa.
        El script principal es responsable de gestionar el timer de 12hs.
        """
        return self.managers[bot_id].is_circuit_breaker_active()

    def _get_exposure_multipliers(self, signals: dict[str, str]) -> dict[str, float]:
        """
        Calcula los multiplicadores de posición según el Global Exposure Manager.
        Si 3+ bots van en la misma dirección, las señales de menor prioridad se reducen al 50%.
        """
        multipliers = {bot_id: 1.0 for bot_id in signals}
        
        # Agrupar bots por dirección
        direction_groups: dict[str, list[str]] = {}
        for bot_id, direction in signals.items():
            direction_groups.setdefault(direction, []).append(bot_id)

        # Aplicar penalización al excedente si hay 3+ en la misma dirección
        for direction, bots in direction_groups.items():
            if len(bots) >= 3:
                # Ordenamos los bots basándonos en la jerarquía oficial
                bots_sorted = sorted(
                    bots, 
                    key=lambda b: self.priority_order.index(b) if b in self.priority_order else 99
                )
                
                # Los primeros 2 entran full, el resto sufre el recorte de exposición
                for bot_id in bots_sorted[2:]:
                    multipliers[bot_id] = 0.5
                    
        return multipliers

    def get_final_position_size(self, bot_id: str, direction: str, 
                                all_signals: dict[str, str]) -> float:
        manager = self.managers[bot_id]
        
        # Guardianes de seguridad innegociables
        if manager.is_margin_halted():
            return 0.0
            
        # NUEVO: Si estamos en Backtest, ignoramos el Circuit Breaker temporal
        if not self.backtest_mode and self.check_circuit_breaker(bot_id):
            return 0.0
            
        base_size = manager.get_position_size()
        multipliers = self._get_exposure_multipliers(all_signals)
        
        return base_size * multipliers.get(bot_id, 1.0)
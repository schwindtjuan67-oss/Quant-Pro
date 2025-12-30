# Live/shadow_risk_manager.py
#
# Alias 1:1 del RiskManager institucional para el entorno SHADOW.
# De esta forma, cualquier cambio que hagas en Live/risk_manager.py
# se refleja automáticamente también en el ShadowEngine, sin duplicar código.


from Live.risk_manager import RiskManager as ShadowRiskManager

__all__ = ["ShadowRiskManager"]

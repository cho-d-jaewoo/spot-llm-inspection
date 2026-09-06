"""Human oracle — LLM 에스컬레이션 시 GT Tier2 값을 반환 (완벽한 인간 판단 시뮬레이션)."""

from anomaly_config import get_ground_truth


def oracle_tier2(anomaly_type: str, attrs: dict) -> str:
    """LLM이 에스컬레이션 결정 시 호출 → GT Tier2 반환."""
    return get_ground_truth(anomaly_type, attrs)["tier2"]

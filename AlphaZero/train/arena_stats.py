"""Promotion statistics independent of the neural network runtime."""
from math import sqrt


def promotion_evidence(wins: int, losses: int, draws: int,
                       score_threshold: float) -> tuple[float, float, bool]:
    """Return score, Wilson lower bound on decisive win rate, promotion.

    Draws count as half a point for match score. Only decisive games count
    toward evidence that the candidate beats the incumbent.
    """
    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total else 0.0
    decisive = wins + losses
    if not decisive:
        return score, 0.0, False
    p = wins / decisive
    z = 1.959963984540054
    denominator = 1 + z * z / decisive
    lower = (p + z * z / (2 * decisive) -
             z * sqrt((p * (1 - p) + z * z / (4 * decisive)) / decisive)) / denominator
    return score, lower, score >= score_threshold and lower > 0.5

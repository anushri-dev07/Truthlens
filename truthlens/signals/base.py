from __future__ import annotations


class SignalScore:
    def __init__(self, name: str, prob_ai: float = 0.5, evidence: list[str] | None = None):
        self.name = name
        self.prob_ai = max(0.0, min(1.0, float(prob_ai)))
        self.evidence = evidence or []

    def to_dict(self):
        return {"name": self.name, "prob_ai": self.prob_ai, "evidence": self.evidence}

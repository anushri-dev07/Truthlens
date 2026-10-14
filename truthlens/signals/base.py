from __future__ import annotations

VALID_STATUS = {"completed", "not_available", "failed"}


class SignalScore:
    def __init__(
        self,
        name: str,
        prob_ai: float = 0.5,
        evidence: list[str] | None = None,
        status: str = "completed",
        details: dict | None = None,
    ):
        self.name = name
        self.prob_ai = max(0.0, min(1.0, float(prob_ai)))
        self.evidence = evidence or []
        self.status = status if status in VALID_STATUS else "completed"
        self.details = details or {}

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prob_ai": self.prob_ai,
            "evidence": self.evidence,
            "status": self.status,
            "details": self.details,
        }

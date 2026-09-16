"""Configuration loader for Speech-act Role Attention Completion (SRAC)."""

from dataclasses import dataclass


@dataclass
class SRACConfig:
    enabled: bool = True
    mode: str = "anomaly_only"
    threshold: float = 0.72
    max_history_rounds: int = 10
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    polarity_positive_threshold: float = 0.55
    polarity_unrelated_threshold: float = 0.35
    polarity_positive_anchors: list[str] | None = None
    polarity_negative_threshold: float = 0.55
    polarity_negative_anchors: list[str] | None = None
    allowed_illocutions: list[str] | None = None
    auto_close_rounds: int | None = None
    # Dose ceilings on delivered completions.  None/0 leaves the volume
    # uncapped, which is the behaviour of every previously reported run.
    max_completions_per_round: int | None = None
    max_completions_per_task: int | None = None


def load_srac_config(raw: dict | None) -> SRACConfig:
    """Build SRACConfig from raw YAML dict."""
    if raw is None:
        return SRACConfig()
    polarity = raw.get("semantic_polarity", {}) or {}
    return SRACConfig(
        enabled=bool(raw.get("enabled", True)),
        mode=str(raw.get("mode", "anomaly_only")),
        threshold=float(raw.get("similarity_threshold", 0.72)),
        max_history_rounds=int(raw.get("max_history_rounds", 10)),
        model_name=str(raw.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
        device=str(raw.get("device", "cpu")),
        batch_size=int(raw.get("batch_size", 32)),
        polarity_positive_threshold=float(polarity.get("positive_threshold", 0.55)),
        polarity_unrelated_threshold=float(polarity.get("unrelated_threshold", 0.35)),
        polarity_positive_anchors=polarity.get("positive_anchors", None),
        polarity_negative_threshold=float(polarity.get("negative_threshold", 0.55)),
        polarity_negative_anchors=polarity.get("negative_anchors", None),
        allowed_illocutions=raw.get("allowed_illocutions", None),
        auto_close_rounds=(
            int(raw["auto_close_rounds"]) if raw.get("auto_close_rounds") is not None else None
        ),
        max_completions_per_round=(
            int(raw["max_completions_per_round"])
            if raw.get("max_completions_per_round") is not None else None
        ),
        max_completions_per_task=(
            int(raw["max_completions_per_task"])
            if raw.get("max_completions_per_task") is not None else None
        ),
    )

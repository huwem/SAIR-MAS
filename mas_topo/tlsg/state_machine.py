"""TLSG state machine — uptake-driven interaction lifecycle (v3).

A speech act reaches its illocutionary effect when uptake occurs: an
expected respondent produces a reply that is semantically related to the
act (positive, neutral, or negative — any substantive engagement counts).
Unrelated replies do not constitute uptake, and the act remains OPEN.

This separates uptake (illocutionary closure) from agreement (perlocutionary
effect), which is handled by the consensus layer.  There is no contested
state — a negative reply closes the original act and opens a new one
asserting a competing claim.

Closure rules:
  Pure informational (Assertive/Expressive) — born CLOSED on write:
        content records that open no interaction and demand no uptake.
  Command-type (Directive/Commissive/Declarative) — born OPEN; closes when
        an expected respondent produces a related substantive (non-Expressive)
        reply.  Silence is not closure; optional auto_close_rounds
        force-closes stale acts.

Edge types (response / srac_completion) are provenance metadata for audit
and memory tracing — they record HOW a link was established, not WHETHER
it counts.  Perlocutionary discharge is judged from the child node's own
properties: sender qualification (expected respondent), illocutionary
substance (non-Expressive), and semantic relatedness (polarity).
"""

import logging
from dataclasses import dataclass, field

from mas_topo.tlsg.graph import TemporalLayeredSpeechActGraph, IllocutionNode

logger = logging.getLogger(__name__)


@dataclass
class StateMachineReport:
    updated_ids: list[int] = field(default_factory=list)
    open_count: int = 0
    closed_count: int = 0


class TLSGStateMachine:
    """Speech-act-driven state inference: OPEN -> CLOSED when an expected
    respondent produces a semantically related substantive reply (uptake).
    Unrelated replies and Expressive (non-propositional) children do not
    close.  No contested state — negative is valid uptake.

    Optional ``auto_close_rounds``: OPEN nodes older than this many rounds
    are closed unconditionally (opt-in escape hatch so stale obligations
    leave the SRAC pool; None keeps the v3 "never via time" semantics).
    """

    def __init__(self, polarity_model=None, auto_close_rounds: int | None = None, **_kwargs):
        self._polarity = polarity_model  # Optional[SemanticPolarity]
        self.auto_close_rounds = auto_close_rounds

    # -- public API ---------------------------------------------------------

    def update(
        self,
        graph: TemporalLayeredSpeechActGraph,
        current_round: int,
        related_pairs: set[tuple[int, int]] | None = None,
    ) -> StateMachineReport:
        """Advance perlocutionary effects for one round.

        ``related_pairs``: optional precomputed ``{(parent_id, child_id)}``
        relatedness from the routing layer's unified embedding pass.  When
        given, the polarity model is bypassed (no re-embedding); when None,
        falls back to the polarity model (standalone/ablation use).
        """
        report = StateMachineReport()

        pairs: list[tuple[IllocutionNode, IllocutionNode]] = []
        for node in graph.all_nodes():
            if node.status != "OPEN":
                continue
            respondents = node.expected_respondents
            # 边类型不参与闭合判定（response / srac_completion 仅记录链路
            # 建立方式）；言后效果由子节点自身性质判定：预期回应者 +
            # 有命题内容（非 Expressive）+ 语义相关（polarity 门）。
            for child in graph.children_of(node.entry_id):
                if child.sender in respondents and child.illocution != "Expressive":
                    pairs.append((node, child))

        if related_pairs is not None:
            # 统一言后通道已用共享嵌入矩阵判定过相关性，直接复用。
            labels = [
                "positive" if (p.entry_id, c.entry_id) in related_pairs else "unrelated"
                for p, c in pairs
            ]
        elif pairs and self._polarity is not None:
            try:
                pair_texts = [
                    (p.to_embedding_text(), c.to_embedding_text())
                    for p, c in pairs
                ]
                labels = self._polarity.batch_polarity(pair_texts)
            except Exception:
                logger.warning("Batch polarity failed, falling back to per-pair.")
                labels = [self._label_for(p, c) for p, c in pairs]
        elif pairs:
            # No polarity model — permissive: any structural match closes.
            labels = ["positive"] * len(pairs)
        else:
            labels = []

        closed: set[int] = set()
        for (parent, _child), label in zip(pairs, labels):
            if parent.entry_id in closed:
                continue
            # Uptake: any substantive reply (positive/neutral/negative)
            # constitutes uptake and closes the speech act.
            if label != "unrelated":
                parent.status = "CLOSED"
                report.updated_ids.append(parent.entry_id)
                closed.add(parent.entry_id)

        # Auto-close stale OPEN nodes so they leave the SRAC pool (opt-in).
        if self.auto_close_rounds is not None:
            for node in graph.all_nodes():
                if (
                    node.status == "OPEN"
                    and current_round - node.round >= self.auto_close_rounds
                ):
                    node.status = "CLOSED"
                    report.updated_ids.append(node.entry_id)

        for node in graph.all_nodes():
            if node.status == "CLOSED":
                report.closed_count += 1
            elif node.status == "OPEN":
                report.open_count += 1

        return report

    # -- per-pair fallback --------------------------------------------------

    def _label_for(self, parent: IllocutionNode, child: IllocutionNode) -> str:
        """Fallback when batch encoding fails — conservative ("unrelated")."""
        if self._polarity is None:
            return "positive"
        try:
            label, _score = self._polarity.polarity(
                parent.to_embedding_text(),
                child.to_embedding_text(),
            )
            return label
        except Exception:
            logger.warning(
                "SemanticPolarity failed; treating pair as unrelated.", exc_info=True
            )
            return "unrelated"

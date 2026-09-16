"""Speech-act Role Attention Completion (SRAC) for TLSG.

Speech Act Theory (Austin 1962): for an illocutionary act to succeed, the
hearer must achieve "uptake" — understanding of the illocutionary force.
Without uptake, the speech act's perlocutionary effect cannot be realized.

SRAC is the pragmatic mechanism that assures uptake: when a new speech act
is semantically related to an unresolved (OPEN) act in the collaboration
ledger, SRAC infers that the OPEN act's obligation bearer should receive
the new act — ensuring the illocutionary force reaches the right agent.

This is pragmatic inference (cf. Grice's Maxim of Relation), not semantic
matching: we are not guessing "who might be interested in this content",
but rather "whose outstanding obligation this act engages with."
"""

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SRACCandidate:
    recipient: str
    ref_entry_id: int
    score: float
    decayed_score: float


class SRACCompletion:
    """Pragmatic inference of additional recipients to assure illocutionary uptake.

    Uses Sentence-BERT to detect semantic relatedness between a new speech act
    and unresolved (OPEN) acts.  When related, the OPEN act's sender is inferred
    as a necessary recipient — ensuring the illocutionary force of the new act
    reaches those who have a stake in the ongoing collaboration.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.72,
        device: str = "cpu",
        max_history_rounds: int = 10,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.max_history_rounds = max_history_rounds
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy-load and return the shared SentenceTransformer model."""
        self._load_model()
        return self._model

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from mas_topo.tlsg.model_loader import load_sentence_transformer

            self._model = load_sentence_transformer(self.model_name, device=self.device)
        except Exception as e:
            logger.error("Failed to load Sentence-BERT model %s: %s", self.model_name, e)
            raise

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        try:
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:
            # Fallback for models/embedders that don't accept SentenceTransformer kwargs.
            return self._model.encode(texts)

    def score_matrix(self, new_nodes: list, open_nodes: list) -> np.ndarray:
        """Embed once and return the ``|open| x |new|`` cosine similarity matrix.

        Single relatedness source shared by semantic linking, SRAC delivery,
        and state-machine closure (the unified perlocutionary pass).
        """
        self._load_model()
        if not open_nodes or not new_nodes:
            return np.zeros((len(open_nodes), len(new_nodes)))
        texts_open = [n.to_embedding_text() for n in open_nodes]
        texts_new = [n.to_embedding_text() for n in new_nodes]
        emb = self._embed(texts_open + texts_new)
        emb_open, emb_new = emb[: len(open_nodes)], emb[len(open_nodes):]
        return self._cosine_similarity(emb_open, emb_new)

    def candidates_from_scores(
        self,
        new_node,
        scored_open: list[tuple],
        already_delivered: set[str] | None = None,
    ) -> list[SRACCandidate]:
        """Turn precomputed ``(open_node, score)`` pairs into delivery candidates.

        Applies the link threshold, waiting-party derivation, and delivery
        dedup.  ``new_node.recipients`` may be mutated by the caller between
        invocations — dedup checks it live, so sequential calls stay correct.
        """
        # Unify all already-covered recipients into one set for O(1) lookup.
        covered: set[str] = set(already_delivered or ())
        covered.update(new_node.recipients)
        results: list[SRACCandidate] = []
        for node, score in scored_open:
            if score < self.threshold:
                continue
            waiting_parties = self._waiting_parties(node)
            for recipient in waiting_parties:
                if recipient == new_node.sender:
                    continue
                if recipient in covered:
                    continue
                results.append(
                    SRACCandidate(
                        recipient=recipient,
                        ref_entry_id=node.entry_id,
                        score=float(score),
                        decayed_score=float(score),
                    )
                )
        # Sort by score descending (no contested — uptake-driven closure means
        # all OPEN nodes are equally awaiting first response).
        results.sort(key=lambda c: -c.score)
        return results

    def complete(
        self,
        new_node,
        open_nodes: Iterable,
        sender_neighbors: set[str],
        current_round: int,
        allowed_illocutions: list[str] | None = None,
        already_delivered: set[str] | None = None,
    ) -> list[SRACCandidate]:
        """Infer additional recipients via pragmatic relatedness to unresolved speech acts.

        v3 improvements:
        - Illocution gate: only Assertive/Expressive are completed (Directive/
          Commissive already declare their recipients structurally).
        - Waiting-party derivation: the recipient to add depends on the OPEN
          node's illocution type, not always the sender.
        - Dedup: recipients already delivered (in new_node.recipients or
          prior SRAC completions) are skipped.
        """
        # Illocution gate: only complete for allowed illocution types.
        if allowed_illocutions and new_node.illocution not in allowed_illocutions:
            return []

        try:
            self._load_model()
        except Exception:
            logger.warning("SRAC model unavailable; skipping completion.")
            return []

        candidates_pool = [
            n
            for n in open_nodes
            if n.sender != new_node.sender
            and current_round - n.round < self.max_history_rounds
        ]
        if not candidates_pool:
            return []

        reachable = [n for n in candidates_pool if n.sender in sender_neighbors]
        if not reachable:
            return []

        scores = self.score_matrix([new_node], reachable)[:, 0]
        results = self.candidates_from_scores(
            new_node, list(zip(reachable, scores)),
            already_delivered=already_delivered,
        )
        max_score = float(scores.max()) if len(scores) else 0.0
        logger.info(
            "SRAC eval: sender=%s pool=%d reachable=%d max_sim=%.3f threshold=%.2f hits=%d",
            new_node.sender, len(candidates_pool), len(reachable),
            max_score, self.threshold, len(results),
        )
        return results

    @staticmethod
    def _waiting_parties(node) -> set[str]:
        """Derive who is waiting for a response to this OPEN speech act.

        Based on illocution type:
        - Directive → sender (the director awaits execution)
        - Assertive/Declarative → sender (the speaker awaits confirmation)
        - Commissive → sender (the promiser must discharge the commitment)
        - Expressive → empty (no structural obligation)
        """
        il = getattr(node, "illocution", "")
        if il == "Commissive":
            return {node.sender}
        elif il in ("Directive", "Assertive", "Declarative"):
            return {node.sender}
        return set()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)

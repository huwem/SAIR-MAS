"""Two-stage semantic polarity scorer for TLSG state inference.

Reuses the same Sentence-BERT model as SRAC.  Uses a two-stage judgment:

  Stage 1 — Relevance: cosine similarity between parent and child content.
            Below unrelated_threshold → UNRELATED (different topic, not a
            response to this parent).

  Stage 2 — Sentiment (only if related): dual-anchor comparison.  Child
            embedding is compared against both positive and negative anchor
            sets.  If positive-anchor max sim ≥ positive_threshold AND
            ≥ negative-anchor max sim → POSITIVE.  If negative-anchor max
            sim ≥ negative_threshold AND > positive-anchor max sim →
            NEGATIVE.  Otherwise → NEUTRAL (substantive second pair part
            that carries neither explicit agreement nor explicit dissent).

This two-stage design separates two orthogonal questions that a single
parent-child cosine similarity conflates:
  - "Are they talking about the same thing?" (Stage 1)
  - "Is the child agreeing or disagreeing?" (Stage 2, now with explicit
    negative anchors so that neutral substantive replies are NOT mis-labelled
    as disputes).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default positive anchors — semantically encode "agreement, fulfilment,
# approval, completion".  Using English phrases for the default model
# (all-MiniLM-L6-v2) which is English-pretrained.
DEFAULT_POSITIVE_ANCHORS = [
    "good, I agree, this is correct",
    "done, completed, finished successfully",
    "approved, confirmed, verified",
    "yes, that works, looks right",
    "the answer is correct, well done",
]

DEFAULT_NEGATIVE_ANCHORS = [
    "i disagree, this is wrong",
    "there is an error, this fails",
    "no, that is incorrect",
]


class SemanticPolarity:
    """Two-stage four-way semantic polarity scorer.

    Stage 1 — Relevance:   parent ↔ child cosine similarity.
    Stage 2 — Sentiment:   child ↔ positive anchors vs negative anchors.

    Labels: positive | neutral | negative | unrelated

    Thresholds:
        unrelated_threshold — Stage 1: sim below this → UNRELATED.
        positive_threshold  — Stage 2: pos anchor sim ≥ this AND ≥ neg sim → POSITIVE.
        negative_threshold  — Stage 2: neg anchor sim ≥ this AND > pos sim → NEGATIVE.
                               Both below threshold → NEUTRAL.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        positive_threshold: float = 0.55,
        unrelated_threshold: float = 0.35,
        positive_anchors: Optional[List[str]] = None,
        negative_threshold: float = 0.55,
        negative_anchors: Optional[List[str]] = None,
        device: str = "cpu",
        model: Optional[object] = None,
    ):
        self._model_name = model_name
        self.positive_threshold = positive_threshold
        self.unrelated_threshold = unrelated_threshold
        self.negative_threshold = negative_threshold
        self._positive_anchors = positive_anchors or DEFAULT_POSITIVE_ANCHORS
        self._negative_anchors = negative_anchors or DEFAULT_NEGATIVE_ANCHORS
        self._device = device
        self._model: Optional[object] = model  # shared instance or lazy init
        self._anchor_embeds: Optional[np.ndarray] = None
        self._neg_anchor_embeds: Optional[np.ndarray] = None

    @property
    def model(self):
        if self._model is None:
            from mas_topo.tlsg.model_loader import load_sentence_transformer

            self._model = load_sentence_transformer(self._model_name, device=self._device)
        return self._model

    def _get_anchor_embeds(self) -> np.ndarray:
        """Pre-encode positive anchor phrases once."""
        if self._anchor_embeds is None:
            logger.info("Encoding %d positive anchors for polarity", len(self._positive_anchors))
            self._anchor_embeds = self.model.encode(
                self._positive_anchors,
                show_progress_bar=False,
            )
            # Normalize for cosine similarity.
            norms = np.linalg.norm(self._anchor_embeds, axis=1, keepdims=True)
            self._anchor_embeds = self._anchor_embeds / (norms + 1e-8)
        return self._anchor_embeds

    def _get_neg_anchor_embeds(self) -> np.ndarray:
        """Pre-encode negative anchor phrases once."""
        if self._neg_anchor_embeds is None:
            logger.info("Encoding %d negative anchors for polarity", len(self._negative_anchors))
            self._neg_anchor_embeds = self.model.encode(
                self._negative_anchors,
                show_progress_bar=False,
            )
            norms = np.linalg.norm(self._neg_anchor_embeds, axis=1, keepdims=True)
            self._neg_anchor_embeds = self._neg_anchor_embeds / (norms + 1e-8)
        return self._neg_anchor_embeds

    def polarity(
        self,
        parent_content: str,
        child_content: str,
    ) -> Tuple[str, float]:
        """Return (polarity_label, score).

        polarity_label ∈ {"unrelated", "positive", "neutral", "negative"}

        UNRELATED — child is about a different topic (Stage 1 failed).
        POSITIVE  — related, pos anchors ≥ threshold AND ≥ neg anchors.
        NEGATIVE  — related, neg anchors ≥ threshold AND > pos anchors.
        NEUTRAL   — related but neither anchor set strongly matches.
        """
        if not parent_content.strip() or not child_content.strip():
            return "unrelated", 0.0

        # Encode parent and child together in one batch.
        embeds = self.model.encode(
            [parent_content, child_content],
            show_progress_bar=False,
        )
        parent_embed = embeds[0]
        child_embed = embeds[1]

        # Stage 1: Relevance — parent ↔ child cosine similarity.
        parent_norm = float(np.linalg.norm(parent_embed))
        child_norm = float(np.linalg.norm(child_embed))
        relevance_sim = float(
            np.dot(parent_embed, child_embed) / (parent_norm * child_norm + 1e-8)
        )

        if relevance_sim < self.unrelated_threshold:
            return "unrelated", relevance_sim

        # Stage 2: Dual-anchor sentiment.
        child_normalized = child_embed / (child_norm + 1e-8)
        pos_anchor_embeds = self._get_anchor_embeds()
        neg_anchor_embeds = self._get_neg_anchor_embeds()
        pos_sim = float(np.dot(child_normalized, pos_anchor_embeds.T).max())
        neg_sim = float(np.dot(child_normalized, neg_anchor_embeds.T).max())

        if pos_sim >= self.positive_threshold and pos_sim >= neg_sim:
            return "positive", pos_sim
        elif neg_sim >= self.negative_threshold and neg_sim > pos_sim:
            return "negative", neg_sim
        else:
            return "neutral", max(pos_sim, neg_sim)

    def batch_is_related(self, pairs: list[tuple[str, str]]) -> list[bool]:
        """Batch-check whether each (parent, child) pair is semantically related.

        Runs only Stage 1 (relevance filter).  Stage 2 (sentiment) is
        skipped — the state machine only needs to know whether the child
        engages with the parent at all, not the polarity.

        Deduplicates texts so that the same parent/child string is
        encoded only once, then computes cosine per pair in pure numpy.
        """
        if not pairs:
            return []

        # Collect unique texts.
        text_to_idx: dict[str, int] = {}
        unique_texts: list[str] = []
        pair_indices: list[tuple[int, int]] = []
        for p_text, c_text in pairs:
            if p_text not in text_to_idx:
                text_to_idx[p_text] = len(unique_texts)
                unique_texts.append(p_text)
            if c_text not in text_to_idx:
                text_to_idx[c_text] = len(unique_texts)
                unique_texts.append(c_text)
            pair_indices.append((text_to_idx[p_text], text_to_idx[c_text]))

        # Single batch encode.
        embeds = self.model.encode(unique_texts, show_progress_bar=False)

        # Normalize all embeddings for cosine similarity.
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / (norms + 1e-8)

        # Pure-numpy cosine per pair.
        results: list[bool] = []
        for pi, ci in pair_indices:
            sim = float(np.dot(embeds[pi], embeds[ci]))
            results.append(sim >= self.unrelated_threshold)

        return results

    def batch_polarity(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Batch four-way polarity for (parent, child) text pairs.

        Returns one label per pair: "positive" | "neutral" | "negative" | "unrelated".
        Stage 1 (relevance) is identical to batch_is_related; related pairs
        proceed to Stage 2 dual-anchor comparison in the same numpy pass.

        v3 four-way polarity: only explicit negative signals are disputes.
        A neutral substantive reply (related but neither strongly positive
        nor strongly negative) is NOT a dispute — it closes the interaction.
        """
        if not pairs:
            return []

        # Collect unique texts so each string is encoded only once.
        text_to_idx: dict[str, int] = {}
        unique_texts: list[str] = []
        pair_indices: list[tuple[int, int]] = []
        for p_text, c_text in pairs:
            for t in (p_text, c_text):
                if t not in text_to_idx:
                    text_to_idx[t] = len(unique_texts)
                    unique_texts.append(t)
            pair_indices.append((text_to_idx[p_text], text_to_idx[c_text]))

        embeds = self.model.encode(unique_texts, show_progress_bar=False)
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / (norms + 1e-8)

        pos_anchor_embeds = None  # lazy: encode anchors only if some pair is related
        neg_anchor_embeds = None
        results: list[str] = []
        for pi, ci in pair_indices:
            sim = float(np.dot(embeds[pi], embeds[ci]))
            if sim < self.unrelated_threshold:
                results.append("unrelated")
                continue
            if pos_anchor_embeds is None:
                pos_anchor_embeds = self._get_anchor_embeds()
                neg_anchor_embeds = self._get_neg_anchor_embeds()
            pos_sim = float(np.dot(embeds[ci], pos_anchor_embeds.T).max())
            neg_sim = float(np.dot(embeds[ci], neg_anchor_embeds.T).max())
            if pos_sim >= self.positive_threshold and pos_sim >= neg_sim:
                results.append("positive")
            elif neg_sim >= self.negative_threshold and neg_sim > pos_sim:
                results.append("negative")
            else:
                results.append("neutral")
        return results

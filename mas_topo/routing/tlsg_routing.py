"""TLSG-aware routing strategy with SRAC integration.

Each round, agent utterances are recorded as speech acts in the TLSG
collaboration ledger.  The routing layer encodes three dimensions:

  Locutionary  — content transmission (self-routing via private_routing).
  Illocutionary — force categorization (illocution type on each node).
  Perlocutionary — effect tracking (response edges, OPEN/CLOSED lifecycle,
                   SRAC uptake assurance).

SRAC (Speech-act Role Attention Completion) ensures that the illocutionary
force of each speech act reaches its intended audience — it is a pragmatic
repair mechanism, not a semantic matching one.
"""

import logging
from typing import Optional

import numpy as np

from mas_topo._registries import routing_registry
from mas_topo.routing.self_routing import SelfRouting
from mas_topo.tlsg.config import load_srac_config
from mas_topo.tlsg.graph import IllocutionNode, TemporalLayeredSpeechActGraph
from mas_topo.tlsg.srac_completion import SRACCompletion
from mas_topo.tlsg.semantic_polarity import SemanticPolarity
from mas_topo.tlsg.state_machine import TLSGStateMachine

logger = logging.getLogger(__name__)


class TlsgRouting(SelfRouting):
    """Speech-act-aware routing: records utterances to the collaboration ledger.

    Each agent utterance is written to the TLSG graph as a speech act (locutionary
    content + illocutionary force), then the perlocutionary dimension is handled
    through auto-linked response edges, SRAC uptake assurance, and state machine
    perlocutionary effect tracking.

    SRAC and SemanticPolarity are core TLSG mechanisms — always constructed.
    The srac_enabled flag only gates execution (whether completions are
    actually applied), not model loading.
    """

    # Declares that this strategy uses a dedicated TLSG graph stored in
    # ``context.tlsg_graph``, keeping it separate from the legacy linear ledger.
    uses_tlsg_graph = True

    def __init__(
        self,
        memory_truncate: int = 3000,
        manager_name: Optional[str] = None,
        neighbor_policy: str = "strict",
        agent_neighbor_map: Optional[dict[str, list[str]]] = None,
        max_out_degree: Optional[int] = None,
        srac_enabled: bool = True,
        srac_config: Optional[dict] = None,
        first_round_exclude: Optional[list[str]] = None,
    ):
        super().__init__(
            memory_truncate=memory_truncate,
            manager_name=manager_name,
            neighbor_policy=neighbor_policy,
            agent_neighbor_map=agent_neighbor_map,
            max_out_degree=max_out_degree,
        )
        # 首轮不激活的 Agent 名单（如验证角色 Tester / Verifier）：
        # 首轮尚无任何产出可供验证，它们由后续路由自然激活。
        self.first_round_exclude = set(first_round_exclude or ())
        self._srac_cfg = load_srac_config(srac_config)
        # srac_enabled 构造参数与 srac_config.enabled 任一关闭即禁用补全
        self.srac_enabled = srac_enabled and self._srac_cfg.enabled

        # SRAC and SemanticPolarity are core TLSG mechanisms — always load.
        # srac_enabled only gates whether completions are applied, not model loading.
        self._srac = SRACCompletion(
            model_name=self._srac_cfg.model_name,
            threshold=self._srac_cfg.threshold,
            device=self._srac_cfg.device,
            max_history_rounds=self._srac_cfg.max_history_rounds,
            batch_size=self._srac_cfg.batch_size,
        )
        # Share SentenceTransformer if already loaded; otherwise both
        # components lazy-load independently.  Eager access is avoided
        # to prevent torch/sentence-transformers compatibility issues
        # during construction.
        shared_model = self._srac._model  # may be None (not loaded yet)
        self._polarity = SemanticPolarity(
            model=shared_model,
            model_name=self._srac_cfg.model_name,
            positive_threshold=self._srac_cfg.polarity_positive_threshold,
            unrelated_threshold=self._srac_cfg.polarity_unrelated_threshold,
            positive_anchors=self._srac_cfg.polarity_positive_anchors,
            negative_threshold=self._srac_cfg.polarity_negative_threshold,
            negative_anchors=self._srac_cfg.polarity_negative_anchors,
            device=self._srac_cfg.device,
        )

        self._state_machine = TLSGStateMachine(
            polarity_model=self._polarity,
            auto_close_rounds=self._srac_cfg.auto_close_rounds,
        )

    def _state_backend(self, context):
        """TLSG routes write into the dedicated ``context.tlsg_graph``.

        Falls back to ``context.ledger`` for legacy callers/tests that still
        attach a graph there.
        """
        return getattr(context, "tlsg_graph", None) or getattr(context, "ledger", None)

    def get_active_workers(
        self,
        context,
        worker_names,
        previous_outputs=None,
        **kwargs,
    ):
        if not previous_outputs:
            # 首轮全激活（SAIR 设计决策，覆写基类的种子角色激活）：
            # 每个 Agent 从第一轮起即获得发言权与投票机会，
            # “知情沉默”语义对全员自第一轮成立。
            # first_round_exclude 可按配置排除个别角色（如验证角色），默认不排除。
            active = set(worker_names) - self.first_round_exclude
            if active:
                excluded = sorted(set(worker_names) & self.first_round_exclude)
                logger.info(
                    "TlsgRouting: first round, activating %s (excluded: %s)",
                    sorted(active), excluded,
                )
                return active
            # 排除名单覆盖全员（非常规配置）时回退全激活，避免死锁
            logger.info("TlsgRouting: first round, exclude list covers all; activating all")
            return set(worker_names)
        active = super().get_active_workers(
            context, worker_names, previous_outputs=previous_outputs, **kwargs
        )
        graph = self._state_backend(context)
        if not isinstance(graph, TemporalLayeredSpeechActGraph):
            return active
        prev_round = getattr(context, "round_num", 0) - 1
        if prev_round < 1:
            return active
        for node in graph.nodes_by_round(prev_round):
            # B3: Expressive nodes do not structurally obligate recipients.
            if node.illocution == "Expressive":
                continue
            for recipient in node.recipients:
                if recipient in worker_names:
                    active.add(recipient)

        # v3: pending objector reflow — activation from consensus metadata.
        metadata = getattr(context, "metadata", None) or {}
        for name in metadata.get("tlsg_pending_objectors", {}):
            if name in worker_names:
                active.add(name)

        return active

    def route_and_log(
        self,
        agent_names,
        agent_outputs,
        adjacency_matrix,
        context,
        aggregation_order=None,
    ):
        """Execute one round of speech-act recording and perlocutionary tracking.

        1. Record: Parse agent utterances, write speech acts to TLSG ledger.
        2. Auto-link: Create structural response edges (adjacent rounds).
        3. Unified perlocutionary pass: one embedding batch -> sim matrix ->
           semantic edges, SRAC delivery, and closure relatedness.
        4. State machine: Close speech acts whose illocutionary force has
           been discharged (reuses the pass's relatedness when available).
        """
        graph = self._state_backend(context)
        if not isinstance(graph, TemporalLayeredSpeechActGraph):
            # Defensive: create a graph if the strategy was invoked outside Graph.run().
            graph = TemporalLayeredSpeechActGraph()
            context.tlsg_graph = graph

        # 1. Standard LLM routing.  SelfRouting writes directly into the graph
        #    via the ledger-compatible append() interface.
        routed_messages, ledger_entries = super().route_and_log(
            agent_names, agent_outputs, adjacency_matrix, context, aggregation_order
        )

        # 2. Auto-link response edges between newly added nodes and prior OPEN nodes.
        #    Edges are only created across adjacent rounds (previous round ->
        #    current round) and each new node gets at most one parent, preferring
        #    a direct reply thread.  This keeps TLSG parent chains semantically
        #    meaningful and prevents spurious cross-thread edges from polluting
        #    agent memory.
        current_round = getattr(context, "round_num", 0)
        prev_round = current_round - 1
        if prev_round >= 1:
            prior_open = [
                n for n in graph.nodes_by_round(prev_round) if n.status == "OPEN"
            ]
            for node in ledger_entries:
                candidates = [
                    n for n in prior_open
                    if node.sender in n.recipients
                ]
                if not candidates:
                    continue
                # Prefer the direct reply thread when the prior sender is a
                # recipient of the new node; fall back to the most recent match.
                direct = [n for n in candidates if n.sender in node.recipients]
                chosen = max((direct or candidates), key=lambda n: n.entry_id)
                graph.add_edge(chosen.entry_id, node.entry_id, "response")

        # 3. Unified perlocutionary pass: one embedding batch produces the
        #    OPEN x new similarity matrix that drives semantic linking,
        #    SRAC delivery, and closure relatedness (single source of truth).
        related_pairs = None
        if self.srac_enabled and self._srac is not None:
            try:
                # ledger_entries are already IllocutionNode instances when the
                # graph is the ledger; map each entry to itself.
                entry_to_node = {id(entry): entry for entry in ledger_entries}
                related_pairs = self._perlocutionary_pass(
                    routed_messages,
                    ledger_entries,
                    entry_to_node,
                    graph,
                    agent_names,
                    adjacency_matrix,
                    context,
                )
            except Exception as e:
                logger.warning("TlsgRouting: perlocutionary pass failed: %s", e)

        # 4. TLSG state machine update.  When the unified pass ran, closure
        #    reuses its similarity matrix instead of re-embedding pairs.
        current_round = getattr(context, "round_num", 0)
        try:
            sm_report = self._state_machine.update(
                graph, current_round, related_pairs=related_pairs
            )
            logger.debug(
                "TLSG state machine: closed=%d open=%d",
                sm_report.closed_count,
                sm_report.open_count,
            )
        except Exception as e:
            logger.warning("TlsgRouting: state machine update failed: %s", e)

        return routed_messages, ledger_entries

    def _perlocutionary_pass(
        self,
        routed_messages,
        ledger_entries,
        entry_to_node,
        graph,
        agent_names,
        adjacency_matrix,
        context,
    ):
        """Unified perlocutionary pass: one embedding batch, one sim matrix.

        The matrix ``M[open, new]`` is the single relatedness source for:

        - semantic edges (``M >= link threshold``): cross-round engagement
          evidence, recorded as ``srac_completion`` edges (audit metadata);
        - SRAC delivery: waiting parties of strongly related OPEN acts
          receive the new act (uptake assurance);
        - closure: pairs with ``M >= unrelated threshold`` are handed to the
          state machine as precomputed relatedness.

        Same-sender pairs are excluded EXCEPT for Commissive OPEN acts —
        the promiser discharges their own commitment, so self-fulfilment
        must be linkable.

        Returns the related pair set, or None when the pass is skipped
        (mode off / anomaly gate) so the state machine falls back to the
        polarity model.
        """
        if self._srac_cfg.mode == "off":
            return None
        current_round = getattr(context, "round_num", 0)
        open_nodes = graph.open_nodes(round_window=self._srac.max_history_rounds)
        if self._srac_cfg.mode == "anomaly_only":
            # Trigger when any OPEN node exists — collaboration is still active.
            if not open_nodes:
                return None

        new_nodes = [entry_to_node.get(id(entry)) for entry in ledger_entries]
        new_nodes = [n for n in new_nodes if n is not None]
        if not open_nodes or not new_nodes:
            return set()

        related_threshold = self._polarity.unrelated_threshold
        allowed = self._srac_cfg.allowed_illocutions
        related_pairs: set[tuple[int, int]] = set()

        # Stage 1: structural pre-filter — separate gated new_nodes (illocution
        # type not in allowed) before any embedding work, then exclude same-sender
        # non-Commissive pairs from the eligible set.
        active_nodes: list[tuple[int, object]] = []  # (original_ni, new_node)
        gated_senders: list[str] = []
        for ni, new_node in enumerate(new_nodes):
            if allowed and new_node.illocution not in allowed:
                gated_senders.append(new_node.sender)
            else:
                active_nodes.append((ni, new_node))

        eligible_pairs: list[tuple[int, object, int, object]] = []  # (oi, open_node, active_idx, new_node)
        needed_oi: set[int] = set()
        sender_neighbors: dict[str, set[str]] = {}
        for _, new_node in active_nodes:
            sender_neighbors[new_node.sender] = self._get_allowed_neighbors(
                new_node.sender, agent_names, adjacency_matrix, context
            )
        for active_idx, (_, new_node) in enumerate(active_nodes):
            for oi, open_node in enumerate(open_nodes):
                same_sender = open_node.sender == new_node.sender
                if same_sender and open_node.illocution != "Commissive":
                    continue
                eligible_pairs.append((oi, open_node, active_idx, new_node))
                needed_oi.add(oi)

        if not eligible_pairs:
            for new_node in new_nodes:
                logger.info(
                    "SRAC eval: sender=%s pool=0 reachable=0 max_sim=0.000 threshold=%.2f hits=0",
                    new_node.sender, self._srac.threshold,
                )
            return related_pairs

        # Stage 2: compute similarity only for the structurally eligible subset.
        # Only active (non-gated) new_nodes participate; gated ones never reach
        # edge creation or delivery.
        needed_open = [open_nodes[i] for i in sorted(needed_oi)]
        oi_to_row = {oi: r for r, oi in enumerate(sorted(needed_oi))}
        active_new_nodes = [n for _, n in active_nodes]
        matrix = self._srac.score_matrix(active_new_nodes, needed_open)

        # Stage 3: per-sender processing with precomputed scores.  Candidates are
        # collected before delivery so an optional dose ceiling can keep the
        # highest-similarity completions when more qualify than it admits.  With
        # no ceiling configured the collection order is delivered untouched,
        # leaving the uncapped path behaviourally unchanged.
        sender_stats: dict[str, dict] = {}
        for new_node in new_nodes:
            sender_stats[new_node.sender] = {
                'pool': 0, 'reachable': 0, 'max_sim': 0.0, 'deliveries': 0,
            }

        # (sim, collection_order, new_node, candidate)
        proposals: list[tuple[float, int, object, object]] = []
        # Mirrors the uncapped path, where new_node.recipients was mutated
        # between candidates_from_scores() calls to dedup across OPEN nodes.
        proposed: dict[int, set[str]] = {
            id(new_node): set(new_node.recipients) for _, new_node in active_nodes
        }

        for oi, open_node, active_idx, new_node in eligible_pairs:
            row = oi_to_row[oi]
            sim = float(matrix[row, active_idx])
            stats = sender_stats[new_node.sender]
            neighbors = sender_neighbors[new_node.sender]

            if sim >= related_threshold:
                related_pairs.add((open_node.entry_id, new_node.entry_id))

            same_sender = open_node.sender == new_node.sender
            if same_sender:
                pass  # Commissive self-fulfilment: no pool/reachable tracking
            else:
                stats['pool'] += 1
                if open_node.sender not in neighbors:
                    continue
                stats['reachable'] += 1

            stats['max_sim'] = max(stats['max_sim'], sim)

            if sim < self._srac.threshold:
                continue

            # Provenance edge records the matched OPEN interaction regardless of
            # whether the ceiling later admits the delivery.
            graph.add_edge(open_node.entry_id, new_node.entry_id, "srac_completion")

            covered = proposed[id(new_node)]
            candidates = self._srac.candidates_from_scores(
                new_node, [(open_node, sim)],
                already_delivered=covered,
            )
            for cand in candidates:
                if cand.recipient in covered:
                    continue
                covered.add(cand.recipient)
                proposals.append((sim, len(proposals), new_node, cand))

        for _, _, new_node, cand in self._admit_within_dose(proposals, context):
            new_node.recipients.append(cand.recipient)
            sender_stats[new_node.sender]['deliveries'] += 1
            logger.info(
                "SRAC completion: %s -> %s (ref_entry=%d, sim=%.3f)",
                new_node.sender, cand.recipient, cand.ref_entry_id, cand.score,
            )
            self._add_srac_message(routed_messages, new_node, cand.recipient)

        for sender, stats in sender_stats.items():
            logger.info(
                "SRAC eval: sender=%s pool=%d reachable=%d max_sim=%.3f threshold=%.2f hits=%d",
                sender, stats['pool'], stats['reachable'], stats['max_sim'],
                self._srac.threshold, stats['deliveries'],
            )

        return related_pairs

    def _admit_within_dose(self, proposals, context):
        """Cap how many SRAC completions are actually delivered.

        Trace audits over the reported runs show the accuracy effect of SRAC is
        monotone in its realized volume: at a fraction of a completion per task
        it helps, and it degrades as the volume grows.  A ceiling bounds that
        volume without rescaling the already-small doses the way a higher
        similarity threshold would.  Ties break toward the earlier proposal, and
        admitted completions are delivered in collection order so the recipient
        list stays independent of the ranking.
        """
        per_round = self._srac_cfg.max_completions_per_round or 0
        per_task = self._srac_cfg.max_completions_per_task or 0
        if per_round <= 0 and per_task <= 0:
            return proposals

        metadata = getattr(context, "metadata", None)
        if per_task > 0 and metadata is None:
            logger.warning(
                "SRAC dose: context exposes no metadata dict; per-task ceiling ignored."
            )

        limit = len(proposals)
        if per_round > 0:
            limit = min(limit, per_round)
        spent = 0
        if per_task > 0 and metadata is not None:
            spent = int(metadata.get("srac_completions_spent", 0))
            limit = min(limit, max(per_task - spent, 0))

        ranked = sorted(proposals, key=lambda p: (-p[0], p[1]))[:limit]
        admitted = sorted(ranked, key=lambda p: p[1])

        if per_task > 0 and metadata is not None:
            metadata["srac_completions_spent"] = spent + len(admitted)

        dropped = len(proposals) - len(admitted)
        if dropped:
            # ``spent`` is only meaningful while a per-task ceiling is tracking it.
            task_spent = spent + len(admitted) if per_task > 0 else None
            logger.info(
                "SRAC dose: admitted=%d dropped=%d task_spent=%s "
                "(per_round=%s per_task=%s)",
                len(admitted), dropped, task_spent,
                per_round or None, per_task or None,
            )
        return admitted

    def _add_srac_message(self, routed_messages, entry, recipient):
        """Append an SRAC-derived message to the recipient's inbox."""
        from mas_topo.message import Channel, Message

        content = f"[SRAC][{entry.illocution}]: {entry.content}"
        if self.memory_truncate and self.memory_truncate > 0:
            content = content[: self.memory_truncate]
        routed_messages.setdefault(recipient, [])
        routed_messages[recipient].append(
            Message(
                content=content,
                sender_id=entry.sender,
                receiver_id=recipient,
                channel=Channel.FT_PRIVATE,
                metadata={
                    "illocution_category": entry.illocution,
                    "illocution_content": entry.content,
                    "is_srac_completion": True,
                },
            )
        )


@routing_registry.register(
    "tlsg_routing",
    display_name="TLSG 图路由",
    description="TLSG 图路由：Agent 通过 private_routing 自主声明通信对象和语旨，"
    "框架写入时间分层语旨图并运行 SRAC 语义补全和 TLSG 状态机。",
    config_schema={
        "memory_truncate": {
            "type": "int",
            "default": 3000,
            "description": "消息内容截断长度，0 表示不截断",
        },
        "manager_name": {
            "type": "str|null",
            "default": None,
            "description": "兼容旧配置，P2P 中通常不设置",
        },
        "neighbor_policy": {
            "type": "str",
            "default": "strict",
            "description": "邻居校验策略: strict | filter | permissive",
        },
        "agent_neighbor_map": {
            "type": "object|null",
            "default": None,
            "description": "显式邻居映射 {agent_name: [neighbor_name, ...]}",
        },
        "max_out_degree": {
            "type": "int|null",
            "default": None,
            "description": "每个 Agent 每轮最多可发送的不同接收者数",
        },
        "srac_enabled": {
            "type": "bool",
            "default": True,
            "description": "是否执行 SRAC 语义补全（SRAC 模型始终加载，此开关仅控制是否应用补全结果）",
        },
        "srac_config": {
            "type": "object|null",
            "default": None,
            "description": "SRAC 配置项",
        },
        "first_round_exclude": {
            "type": "list|null",
            "default": None,
            "description": "首轮不激活的 Agent 名单（如验证角色 Tester/Verifier），None 表示首轮全激活",
        },
    },
)
class TlsgRoutingRegistered(TlsgRouting):
    """Registered variant for YAML-driven construction."""

    def __init__(
        self,
        memory_truncate: int = 3000,
        manager_name: Optional[str] = None,
        neighbor_policy: str = "strict",
        agent_neighbor_map: Optional[dict[str, list[str]]] = None,
        max_out_degree: Optional[int] = None,
        srac_enabled: bool = True,
        srac_config: Optional[dict] = None,
        first_round_exclude: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            memory_truncate=memory_truncate,
            manager_name=manager_name,
            neighbor_policy=neighbor_policy,
            agent_neighbor_map=agent_neighbor_map,
            max_out_degree=max_out_degree,
            srac_enabled=srac_enabled,
            srac_config=srac_config,
            first_round_exclude=first_round_exclude,
        )

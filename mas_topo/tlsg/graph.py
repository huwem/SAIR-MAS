"""Temporal-Layered Illocutionary Graph (TLSG).

Speech Act Theory (Austin 1962, Searle 1969) teaches that every utterance
simultaneously performs three acts:

  Locutionary act  — the literal content spoken.
  Illocutionary act — the force/intent carried by the utterance (Directive,
                      Commissive, Assertive, Declarative, Expressive).
  Perlocutionary act — the effect produced on the hearer (uptake, compliance,
                       confirmation, etc.).

TLSG is a layered DAG that hardens these three dimensions into a structured
collaboration ledger:

  Node = Illocutionary act (illocution type + content + sender/recipients)
  Edge = Perlocutionary effect chain (response, srac_completion)
  Status (OPEN/CLOSED) = Whether the intended perlocutionary effect has been
                          achieved — an OPEN node represents an unresolved
                          speech act whose force has not yet been discharged.

协作的可靠性不在于信息是否被传达，而在于意图是否被交换，义务是否被锁定，
效果是否被确认。TLSG is the single source of truth for the PragmaMAS system.
"""

from dataclasses import dataclass, field
from typing import Optional

# 命令型语旨：开启一次交互（born OPEN），需要满足条件才闭合。
# Declarative 归入命令型：宣告改变共享状态，听众认可（recognition）是其
# 构成性适切条件，且 OPEN 状态保留 SRAC 异议回流通道。
# 其余（Assertive/Expressive，纯信息型）写入即 CLOSED，
# 不产生路由状态——每条信息型发言本身就是内容记录，不索取回应。
OPENING_ILLOCUTIONS = frozenset({"Directive", "Commissive", "Declarative"})


@dataclass
class IllocutionNode:
    """A single speech act recorded in the collaboration ledger.

    Encodes both the illocutionary force (Searle's taxonomy) and the locutionary
    content.  The perlocutionary dimension is tracked via status (whether the
    intended effect has been achieved) and parent/child edges (effect chains).

    Attributes:
        illocution: One of Directive / Commissive / Assertive / Declarative /
                    Expressive.  The illocutionary type determines who bears the
                    obligation to respond — not the speaker's role or status,
                    but the speech act type itself.
        status: OPEN = act awaiting discharge (Directive: the requested party
                must act; Commissive: the promiser must deliver; Declarative:
                the audience must recognize); CLOSED = discharged, or pure
                informational act (Assertive/Expressive) — closed on write.
    """

    entry_id: int
    round: int
    sender: str
    recipients: list[str]
    illocution: str
    content: str
    status: str | None = None
    parent_ids: list[int] = field(default_factory=list)
    answer_snapshot: Optional[str] = None
    verification_status: Optional[str] = None

    # Alias for ledger-compatible access.
    round_num = property(lambda self: self.round)

    def __post_init__(self) -> None:
        # LLM outputs are case-unstable ('declarative', 'DIRECTIVE', ...).
        # The state machine, anomaly detection and SRAC all compare against
        # capitalized category names, so normalize once at construction —
        # otherwise miscased nodes never match and stay OPEN forever.
        self.illocution = self.illocution.strip().capitalize()
        if self.status is None:
            # 二分类状态模型：命令型 born OPEN 待闭合；信息型写入即 CLOSED。
            self.status = (
                "OPEN" if self.illocution in OPENING_ILLOCUTIONS else "CLOSED"
            )

    def to_embedding_text(self) -> str:
        """Joint illocution-content encoding for embedding models.

        Shared by SRACCompletion and TLSGStateMachine — the canonical
        text representation for perlocutionary satisfaction scoring.
        """
        recipients = ", ".join(self.recipients)
        return (
            f"[{self.illocution}] from {self.sender}"
            f" to [{recipients}]: {self.content}"
        )

    @property
    def expected_respondents(self) -> set[str]:
        """Who must act for this speech act to achieve its perlocutionary effect.

        Directive   → recipients (the requested party must act).
        Commissive  → {sender}  (the promiser must deliver).
        Assertive / Declarative → recipients (listeners confirm / challenge).
        Expressive  → set()     (no reply expected).
        """
        il = self.illocution
        if il == "Directive":
            return set(self.recipients)
        elif il == "Commissive":
            return {self.sender}
        elif il in ("Assertive", "Declarative"):
            return set(self.recipients)
        return set()


@dataclass
class AnomalyReport:
    entry_id: int
    anomaly_type: str  # "orphan" | "fake_declarative"
    description: str
    sender: str = ""
    round: int = 0


class TemporalLayeredSpeechActGraph:
    """In-memory collaboration ledger: speech acts with perlocutionary effect tracking.

    Each node records an illocutionary act (what an agent intended to do with
    their utterance).  Each edge records a perlocutionary link — one speech act
    responding to or affecting another.  The OPEN/CLOSED status tracks whether
    the intended perlocutionary effect has been confirmed, turning speech acts
    from ephemeral rhetoric into auditable, machine-enforced collaboration
    protocol (cf. TCP SYN/ACK).
    """

    def __init__(self):
        self._nodes: dict[int, IllocutionNode] = {}
        self._edges: list[tuple[int, int, str]] = []
        self._children: dict[int, set[int]] = {}  # src → {dst, ...}
        self._parents: dict[int, set[int]] = {}   # dst → {src, ...}
        self._max_round: int = 0
        self._next_node_id: int = 1

    # -- mutation ----------------------------------------------------------

    def add_node(self, node: IllocutionNode) -> int:
        """Store node, assign a graph-scoped unique id. Returns the assigned id."""
        node.entry_id = self._next_node_id
        self._next_node_id += 1
        self._nodes[node.entry_id] = node
        self._max_round = max(self._max_round, node.round)
        return node.entry_id

    def add_node_keep_id(self, node: IllocutionNode) -> None:
        """Store node preserving its existing entry_id (for subgraph copies)."""
        self._nodes[node.entry_id] = node
        if node.entry_id >= self._next_node_id:
            self._next_node_id = node.entry_id + 1
        self._max_round = max(self._max_round, node.round)

    def add_edge(self, src_id: int, dst_id: int, edge_type: str) -> None:
        # Deduplicate: skip if this exact edge already exists.
        if (src_id, dst_id, edge_type) in self._edges:
            return
        self._edges.append((src_id, dst_id, edge_type))
        self._children.setdefault(src_id, set()).add(dst_id)
        self._parents.setdefault(dst_id, set()).add(src_id)
        # Sync parent_ids on the child node for serialization traceability.
        dst_node = self._nodes.get(dst_id)
        if dst_node is not None and src_id not in dst_node.parent_ids:
            dst_node.parent_ids.append(src_id)

    def update_status(self, entry_id: int, status: str) -> None:
        node = self._nodes.get(entry_id)
        if node is not None:
            node.status = status

    def append(
        self,
        round_num: int,
        sender: str,
        recipients: list[str],
        illocution: str,
        content: str,
        status: str | None = None,
        deadline_round: Optional[int] = None,
        expected_effect: str = "",
        evidence_entries: Optional[list[int]] = None,
        answer_snapshot: Optional[str] = None,
        verification_status: Optional[str] = None,
    ) -> IllocutionNode:
        """Ledger-compatible append: create and store an IllocutionNode.

        Parameters matching IllocutionaryLedger.append() for drop-in use by
        SelfRouting.  Ledger-specific fields (deadline_round, expected_effect,
        evidence_entries) are accepted for compatibility but ignored because
        TLSG state machine currently only tracks OPEN/CLOSED.
        """
        node = IllocutionNode(
            entry_id=0,  # assigned by add_node
            round=round_num,
            sender=sender,
            recipients=list(recipients),
            illocution=illocution,
            content=content,
            status=status,
            answer_snapshot=answer_snapshot,
            verification_status=verification_status,
        )
        self.add_node(node)
        return node

    # -- basic query -------------------------------------------------------

    def get(self, entry_id: int) -> Optional[IllocutionNode]:
        return self._nodes.get(entry_id)

    def all_nodes(self) -> list[IllocutionNode]:
        return list(self._nodes.values())

    def all_entries(self) -> list[IllocutionNode]:
        """Alias for ledger compatibility (graph.py expects .all_entries())."""
        return self.all_nodes()

    def nodes_by_round(self, round_num: int) -> list[IllocutionNode]:
        return [n for n in self._nodes.values() if n.round == round_num]

    def open_nodes(self, round_window: int) -> list[IllocutionNode]:
        """Return OPEN nodes within the last `round_window` rounds."""
        if not self._nodes:
            return []
        return [
            n
            for n in self._nodes.values()
            if n.status == "OPEN" and self._max_round - n.round < round_window
        ]

    def max_round(self) -> int:
        return self._max_round

    # -- children / parents ------------------------------------------------

    def children_of(
        self, entry_id: int, edge_type: Optional[str] = None
    ) -> list[IllocutionNode]:
        """Return child nodes of *entry_id*, optionally filtered by edge type.

        Edge types are provenance metadata: the state machine inspects ALL
        children and judges discharge from child properties; memory tracing
        may still filter to ``response`` chains to skip post-hoc links.
        """
        if edge_type is None:
            child_ids = self._children.get(entry_id, ())
        else:
            child_ids = {
                dst for src, dst, et in self._edges
                if src == entry_id and et == edge_type
            }
        return [self._nodes[cid] for cid in child_ids if cid in self._nodes]

    def parents_of(
        self, entry_id: int, edge_type: Optional[str] = None
    ) -> list[IllocutionNode]:
        """Return parent nodes of *entry_id*.

        If *edge_type* is given, only follow edges of that type.  This lets
        memory tracing stay on ``response`` parent chains and ignore
        ``srac_completion`` post-hoc links.
        """
        if edge_type is None:
            parent_ids = self._parents.get(entry_id, ())
        else:
            parent_ids = {
                src for src, dst, et in self._edges
                if dst == entry_id and et == edge_type
            }
        return [self._nodes[pid] for pid in parent_ids if pid in self._nodes]

    # -- frontier ----------------------------------------------------------

    def frontier(self) -> list[IllocutionNode]:
        """Leaf nodes (no outgoing edges) that are OPEN and non-Expressive."""
        return [
            n
            for n in self._nodes.values()
            if n.entry_id not in self._children
            and n.status == "OPEN"
            and n.illocution != "Expressive"
        ]

    # -- critical paths ----------------------------------------------------

    def critical_paths(self, key_roles: set[str]) -> list[list[IllocutionNode]]:
        """Return key-role-touching paths from root to frontier leaves.

        A node "touches" a key role if its sender or any recipient is a key role.
        """
        frontier_leaves = self.frontier()
        if not frontier_leaves:
            return []

        def _touches_key(node: IllocutionNode) -> bool:
            return (
                node.sender in key_roles
                or any(r in key_roles for r in node.recipients)
            )

        # Sliding-window check: whether this leaf-to-root walk touches key roles.
        paths = []
        for leaf in frontier_leaves:
            # BFS/DFS upward to reconstruct paths.
            stack = [(leaf.entry_id, [leaf.entry_id])]
            while stack:
                cur_id, path_ids = stack.pop()
                parents = list(self._parents.get(cur_id, ()))
                if not parents:
                    # Reached root.
                    path_nodes = [self._nodes[pid] for pid in path_ids if pid in self._nodes]
                    if any(_touches_key(n) for n in path_nodes):
                        paths.append(path_nodes)
                    continue
                for p in parents:
                    if p in path_ids:
                        continue  # cycle guard
                    stack.append((p, [p] + path_ids))

        return paths

    # -- closure -----------------------------------------------------------

    def is_closed(self, key_roles: set[str]) -> bool:
        """Check whether all speech acts have achieved their perlocutionary effects.

        Returns True when:
        - Graph is non-empty.
        - All frontier leaves are CLOSED (their intended effects are confirmed).
        - When key_roles is given, at least one critical path exists and
          all speech acts on critical paths are CLOSED.
        """
        if not self._nodes:
            return False

        frontier_leaves = self.frontier()
        if any(n.status != "CLOSED" for n in frontier_leaves):
            return False

        if key_roles:
            cp = self.critical_paths(key_roles)
            if not cp:
                return False
            for path in cp:
                if any(n.status not in ("CLOSED",) for n in path):
                    return False

        return True

    # -- subgraph ----------------------------------------------------------

    def subgraph_for(self, agent_name: str) -> "TemporalLayeredSpeechActGraph":
        """Return the local subgraph relevant to *agent_name*.

        Includes:
        - nodes sent by the agent and all their descendants
        - nodes received by the agent and all their ancestors
        - sibling nodes (share a parent with the agent's nodes)
        - any OPEN node involving the agent
        """
        sub = TemporalLayeredSpeechActGraph()
        included: set[int] = set()

        def _add_descendants(node_id: int):
            if node_id in included:
                return
            included.add(node_id)
            for child in self.children_of(node_id):
                _add_descendants(child.entry_id)

        def _add_ancestors(node_id: int):
            if node_id in included:
                return
            included.add(node_id)
            for parent in self.parents_of(node_id):
                _add_ancestors(parent.entry_id)

        for node in self._nodes.values():
            if node.sender == agent_name:
                _add_descendants(node.entry_id)
            if agent_name in node.recipients:
                _add_ancestors(node.entry_id)
            if node.status == "OPEN" and (
                node.sender == agent_name or agent_name in node.recipients
            ):
                included.add(node.entry_id)

        # Siblings: nodes that share a parent with any included node.
        extra: set[int] = set()
        for nid in list(included):
            for parent in self.parents_of(nid):
                for sibling in self.children_of(parent.entry_id):
                    if sibling.entry_id not in included:
                        extra.add(sibling.entry_id)
        included.update(extra)

        for nid in included:
            if nid in self._nodes:
                sub.add_node_keep_id(self._nodes[nid])
        for src, dst, etype in self._edges:
            if src in included and dst in included:
                sub.add_edge(src, dst, etype)

        return sub

    # -- anomalies ---------------------------------------------------------

    def has_anomalies(self) -> bool:
        """Return True if graph contains structural anomalies (orphans, fake declaratives)."""
        return bool(self.anomalies())

    def anomalies(self) -> list[AnomalyReport]:
        """Structural anomaly detection on the graph."""
        reports: list[AnomalyReport] = []

        # Fake Declarative: Declarative with no parents.
        # Rounds are 1-based; round-1 nodes can never have parents by design
        # (auto-link requires a previous round), so only round > 1 is checked.
        for node in self._nodes.values():
            if node.illocution == "Declarative" and node.round > 1:
                parents = self.parents_of(node.entry_id)
                if not parents:
                    reports.append(AnomalyReport(
                        entry_id=node.entry_id,
                        anomaly_type="fake_declarative",
                        description=f"Declarative node {node.entry_id} has no parents (possibly fake completion)",
                        sender=node.sender,
                        round=node.round,
                    ))

        # Orphan nodes (no parent, not root).  Same round > 1 rationale as above.
        for node in self._nodes.values():
            if node.round > 1:
                parents = self.parents_of(node.entry_id)
                if not parents:
                    already_reported = any(
                        r.entry_id == node.entry_id for r in reports
                    )
                    if not already_reported:
                        reports.append(AnomalyReport(
                            entry_id=node.entry_id,
                            anomaly_type="orphan",
                            description=f"Node {node.entry_id} is orphan (no parent edge)",
                            sender=node.sender,
                            round=node.round,
                        ))

        return reports

    # -- serialization -----------------------------------------------------

    def to_dict_list(self) -> list[dict]:
        """Serialize all nodes to a list of dicts (for trace / audit)."""
        result = []
        for node in self._nodes.values():
            result.append({
                "entry_id": node.entry_id,
                "round": node.round,
                "sender": node.sender,
                "recipients": node.recipients,
                "illocution": node.illocution,
                "content": node.content[:200],
                "status": node.status,
                "parent_ids": list(node.parent_ids),
                "answer_snapshot": node.answer_snapshot,
                "verification_status": node.verification_status,
            })
        result.sort(key=lambda d: (d["round"], d["entry_id"]))
        return result

    def to_graph_dict(self) -> dict:
        """Full graph serialization with nodes and edges (for trace completeness)."""
        return {
            "nodes": self.to_dict_list(),
            "edges": [
                {"src": src, "dst": dst, "type": etype}
                for src, dst, etype in self._edges
            ],
        }

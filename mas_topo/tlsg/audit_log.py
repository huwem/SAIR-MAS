"""TLSG audit log — read-only export of graph state.

Exports TLSG graph as structured logs for trace, reproducibility, and
post-hoc analysis.
"""

import json
from collections import defaultdict

from mas_topo.tlsg.graph import TemporalLayeredSpeechActGraph


class TlsgAuditLog:
    """Read-only export of TLSG state for audit and trace purposes."""

    @staticmethod
    def export(graph: TemporalLayeredSpeechActGraph) -> list[dict]:
        """Export graph as list of dicts (one per node)."""
        return graph.to_dict_list()

    @staticmethod
    def export_json(graph: TemporalLayeredSpeechActGraph, indent: int = 2) -> str:
        """Export graph as JSON string."""
        return json.dumps(TlsgAuditLog.export(graph), indent=indent, ensure_ascii=False)

    @staticmethod
    def summary(graph: TemporalLayeredSpeechActGraph) -> dict:
        """High-level summary of graph state."""
        all_nodes = graph.all_nodes()
        status_counts: dict[str, int] = defaultdict(int)
        for n in all_nodes:
            status_counts[n.status] += 1

        rounds = sorted({n.round for n in all_nodes})
        graph_anomalies = graph.anomalies()
        return {
            "total_nodes": len(all_nodes),
            "total_edges": len(graph._edges),
            "rounds": rounds,
            "max_round": max(rounds) if rounds else 0,
            "status_counts": dict(status_counts),
            "has_anomalies": bool(graph_anomalies),
            "anomaly_count": len(graph_anomalies),
        }

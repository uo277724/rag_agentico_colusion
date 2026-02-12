from typing import Any, Dict, List, Optional

from retrieval.retriever import Retriever
from graph.graph_store import GraphStore


class GraphRetriever:
    """
    Hybrid retriever that expands vector hits using the graph.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        embedder,
        vectorstore,
        top_k_primary: int = 8,
        max_total: int = 20,
        lazy_typer=None,
        enable_lazy_typing: bool = False,
    ):
        self.graph_store = graph_store
        self.retriever = Retriever(
            embedder=embedder,
            vectorstore=vectorstore,
            top_k_primary=top_k_primary,
            top_k_secondary=top_k_primary,
            lazy_typer=lazy_typer,
            enable_lazy_typing=enable_lazy_typing,
        )
        self.max_total = max_total

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        base_docs = self.retriever.retrieve(query)
        if not base_docs:
            return []

        primary_ids: List[str] = []
        for doc in base_docs:
            if not isinstance(doc, dict):
                continue
            meta = doc.get("metadata", {})
            chunk_id = self.graph_store.get_chunk_node_id(
                meta.get("source"),
                meta.get("page"),
                doc.get("content"),
            )
            if chunk_id:
                primary_ids.append(chunk_id)

        expanded_ids: List[str] = list(primary_ids)

        for chunk_id in list(primary_ids):
            doc_id = self.graph_store.get_doc_for_chunk(chunk_id)
            if doc_id:
                expanded_ids.extend(self.graph_store.get_chunks_for_doc(doc_id))

            for bid_id in self.graph_store.get_bid_nodes_for_chunk(chunk_id):
                expanded_ids.extend(self.graph_store.get_chunks_for_bid(bid_id))

        for bidder_id in self.graph_store.find_bidder_nodes(query):
            for bid_id in self.graph_store.get_bids_for_bidder(bidder_id):
                expanded_ids.extend(self.graph_store.get_chunks_for_bid(bid_id))

        final_ids: List[str] = []
        seen = set()
        for node_id in expanded_ids:
            if node_id in seen:
                continue
            if self.graph_store.get_node_type(node_id) != "chunk":
                continue
            seen.add(node_id)
            final_ids.append(node_id)
            if len(final_ids) >= self.max_total:
                break

        if not final_ids:
            return base_docs

        graph_docs: List[Dict[str, Any]] = []
        for node_id in final_ids:
            attrs = self.graph_store.get_node_attrs(node_id)
            content = attrs.get("content")
            if not content:
                continue

            meta = {
                "source": attrs.get("source"),
                "page": attrs.get("page"),
                "semantic_type": attrs.get("semantic_type") or attrs.get("type"),
                "type": attrs.get("type"),
                "confidence": attrs.get("confidence", 1.0),
            }
            graph_docs.append({"content": content, "metadata": meta})

        return graph_docs or base_docs

import hashlib
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx


class GraphStore:
    """
    In-memory graph store for hybrid GraphRAG.

    Node types:
    - document
    - chunk
    - bidder
    - bid
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.chunk_index_by_source_page: Dict[Tuple[str, Optional[int]], List[str]] = {}
        self.bidder_name_index: Dict[str, str] = {}

    # --------------------------------------------------
    # ID helpers
    # --------------------------------------------------
    def _doc_id(self, source: str) -> str:
        return f"doc:{source}"

    def _chunk_id(self, source: str, page: Optional[int], content: str) -> str:
        key = f"{source}|{page}|{content}"
        digest = hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()
        return f"chunk:{digest}"

    def _bid_id(self) -> str:
        return f"bid:{uuid.uuid4().hex}"

    def _bidder_id(self, name: str) -> str:
        digest = hashlib.md5(name.lower().strip().encode("utf-8", errors="ignore")).hexdigest()
        return f"bidder:{digest}"

    # --------------------------------------------------
    # Public helpers
    # --------------------------------------------------
    def get_node_attrs(self, node_id: str) -> Dict[str, Any]:
        return dict(self.graph.nodes.get(node_id, {}))

    def get_node_type(self, node_id: str) -> Optional[str]:
        attrs = self.graph.nodes.get(node_id)
        if not attrs:
            return None
        return attrs.get("node_type")

    def get_chunk_node_id(
        self,
        source: Optional[str],
        page: Optional[int],
        content: Optional[str],
    ) -> Optional[str]:
        if not source or not content:
            return None
        node_id = self._chunk_id(source, page, content)
        return node_id if node_id in self.graph.nodes else None

    def get_chunks_for_source_page(
        self,
        source: Optional[str],
        page: Optional[int],
    ) -> List[str]:
        if not source:
            return []
        key = (source, page)
        return list(self.chunk_index_by_source_page.get(key, []))

    def get_doc_for_chunk(self, chunk_id: str) -> Optional[str]:
        if chunk_id not in self.graph.nodes:
            return None
        for node_id, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get("type") == "HAS_CHUNK" and self.get_node_type(node_id) == "document":
                return node_id
        return None

    def get_chunks_for_doc(self, doc_id: str) -> List[str]:
        if doc_id not in self.graph.nodes:
            return []
        chunks = []
        for _, node_id, data in self.graph.out_edges(doc_id, data=True):
            if data.get("type") == "HAS_CHUNK" and self.get_node_type(node_id) == "chunk":
                chunks.append(node_id)
        return chunks

    def get_bid_nodes_for_chunk(self, chunk_id: str) -> List[str]:
        if chunk_id not in self.graph.nodes:
            return []
        bids = []
        for node_id, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get("type") == "MENTIONED_IN_CHUNK" and self.get_node_type(node_id) == "bid":
                bids.append(node_id)
        return bids

    def get_chunks_for_bid(self, bid_id: str) -> List[str]:
        if bid_id not in self.graph.nodes:
            return []
        chunks = []
        for _, node_id, data in self.graph.out_edges(bid_id, data=True):
            if data.get("type") == "MENTIONED_IN_CHUNK" and self.get_node_type(node_id) == "chunk":
                chunks.append(node_id)
        return chunks

    def get_bids_for_bidder(self, bidder_id: str) -> List[str]:
        if bidder_id not in self.graph.nodes:
            return []
        bids = []
        for _, node_id, data in self.graph.out_edges(bidder_id, data=True):
            if data.get("type") == "OFFERS" and self.get_node_type(node_id) == "bid":
                bids.append(node_id)
        return bids

    def find_bidder_nodes(self, query: str) -> List[str]:
        query_lower = query.lower()
        matches = []
        for name_lower, node_id in self.bidder_name_index.items():
            if name_lower and name_lower in query_lower:
                matches.append(node_id)
        return matches

    # --------------------------------------------------
    # Mutation API
    # --------------------------------------------------
    def add_document(self, source: str) -> str:
        doc_id = self._doc_id(source)
        if doc_id not in self.graph.nodes:
            self.graph.add_node(doc_id, node_type="document", source=source)
        return doc_id

    def add_chunk(self, doc: Dict[str, Any]) -> Optional[str]:
        source = doc.get("source")
        content = doc.get("content")
        page = doc.get("page")
        doc_type = doc.get("type")
        semantic_type = doc.get("semantic_type") or doc_type

        if not source or not content:
            return None

        doc_id = self.add_document(source)
        chunk_id = self._chunk_id(source, page, content)

        if chunk_id not in self.graph.nodes:
            self.graph.add_node(
                chunk_id,
                node_type="chunk",
                source=source,
                page=page,
                type=doc_type,
                semantic_type=semantic_type,
                content=content,
                confidence=doc.get("confidence", 1.0),
            )
            self.graph.add_edge(doc_id, chunk_id, type="HAS_CHUNK")

            key = (source, page)
            self.chunk_index_by_source_page.setdefault(key, []).append(chunk_id)

        return chunk_id

    def add_chunks(self, docs: Iterable[Dict[str, Any]]):
        for doc in docs:
            self.add_chunk(doc)

    def add_bidder(self, name: str) -> str:
        bidder_id = self._bidder_id(name)
        if bidder_id not in self.graph.nodes:
            self.graph.add_node(bidder_id, node_type="bidder", name=name)
        self.bidder_name_index[name.lower()] = bidder_id
        return bidder_id

    def add_bid(
        self,
        bid: Dict[str, Any],
        source_refs: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        bid_id = self._bid_id()
        self.graph.add_node(
            bid_id,
            node_type="bid",
            amount=bid.get("amount"),
            currency=bid.get("currency"),
            tax_included=bid.get("tax_included"),
            confidence=bid.get("confidence"),
            bidder=bid.get("bidder"),
        )

        bidder_name = bid.get("bidder")
        if bidder_name:
            bidder_id = self.add_bidder(bidder_name)
            self.graph.add_edge(bidder_id, bid_id, type="OFFERS")

        refs = source_refs or []
        for ref in refs:
            source = ref.get("source")
            page = ref.get("page")
            if not source:
                continue

            doc_id = self.add_document(source)
            self.graph.add_edge(bid_id, doc_id, type="MENTIONED_IN_DOC")

            chunk_ids = self.get_chunks_for_source_page(source, page)
            for chunk_id in chunk_ids[:3]:
                self.graph.add_edge(bid_id, chunk_id, type="MENTIONED_IN_CHUNK")

        return bid_id

    def add_bids(
        self,
        bids: Iterable[Dict[str, Any]],
        default_source_refs: Optional[List[Dict[str, Any]]] = None,
    ):
        for bid in bids:
            refs = bid.get("source_refs") or default_source_refs
            self.add_bid(bid, refs)

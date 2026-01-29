# agents/consolidation/bid_consolidator.py

from typing import List, Dict, Any
from collections import defaultdict


class BidConsolidationAgent:
    """
    Consolidates structured bid candidates into a clean list of final bid amounts.

    Responsibilities:
    - Select ONE bid per bidder (when possible)
    - Prefer tax-included amounts if explicitly stated
    - Prefer higher-confidence bids
    - Remove duplicates
    - NEVER compute or infer missing values
    """

    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence

    def consolidate(self, bid_payload: Dict[str, Any]) -> Dict[str, Any]:
        bids = bid_payload.get("bids", [])
        if not isinstance(bids, list) or not bids:
            raise ValueError("No bid candidates provided")

        decisions: List[str] = []
        discarded: List[Dict[str, Any]] = []

        # -----------------------------------------
        # STEP 1: Group bids by bidder
        # -----------------------------------------
        grouped = defaultdict(list)
        for b in bids:
            bidder = b.get("bidder") or "UNKNOWN_BIDDER"
            grouped[bidder].append(b)

        final_bids: List[float] = []
        used_amounts = set()
        currency = None
        confidences = []

        # -----------------------------------------
        # STEP 2: Select best bid per bidder
        # -----------------------------------------
        for bidder, candidates in grouped.items():

            # Filter by minimum confidence
            valid = [
                b for b in candidates
                if isinstance(b.get("amount"), (int, float))
                and b.get("confidence", 0) >= self.min_confidence
            ]

            if not valid:
                discarded.extend(candidates)
                decisions.append(
                    f"{bidder}: descartado por baja confianza"
                )
                continue

            # Prefer tax-included if explicitly stated
            tax_included = [b for b in valid if b.get("tax_included") is True]
            pool = tax_included if tax_included else valid

            # Pick highest confidence
            selected = sorted(
                pool,
                key=lambda b: b.get("confidence", 0),
                reverse=True
            )[0]

            amount = float(selected["amount"])

            # Avoid duplicates across bidders
            if amount in used_amounts:
                discarded.append(selected)
                decisions.append(
                    f"{bidder}: importe {amount} descartado por duplicado"
                )
                continue

            used_amounts.add(amount)
            final_bids.append(amount)
            confidences.append(selected.get("confidence", 0.5))

            if selected.get("tax_included") is True:
                decisions.append(
                    f"{bidder}: seleccionado importe con IVA explícito ({amount})"
                )
            else:
                decisions.append(
                    f"{bidder}: seleccionado único importe disponible ({amount})"
                )

            if not currency and selected.get("currency"):
                currency = selected.get("currency")

            # Discard non-selected candidates
            for b in candidates:
                if b is not selected:
                    discarded.append(b)

        if not final_bids:
            raise ValueError("No valid bids after consolidation")

        # -----------------------------------------
        # STEP 3: Aggregate confidence
        # -----------------------------------------
        overall_confidence = round(
            sum(confidences) / len(confidences), 2
        ) if confidences else 0.0

        return {
            "final_bids": final_bids,
            "currency": currency,
            "confidence": overall_confidence,
            "decisions": decisions,
            "discarded": discarded,
        }

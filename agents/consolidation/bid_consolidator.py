# agents/consolidation/bid_consolidator.py

from typing import List, Dict, Any
from collections import defaultdict


class BidConsolidationAgent:
    """
    Consolidates structured bid candidates into a clean list of final bid amounts.

    PRINCIPIO CLAVE:
    - Separación explícita entre DATOS y DECISIONES
    - No se infiere ni se calcula información inexistente
    """

    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence

    def consolidate(self, bid_payload: Dict[str, Any]) -> Dict[str, Any]:
        bids = bid_payload.get("bids", [])
        if not isinstance(bids, list) or not bids:
            raise ValueError("No bid candidates provided")

        # ======================================================
        # DATA LAYER (datos puros)
        # ======================================================
        data: Dict[str, Any] = {
            "all_bids": bids,
            "grouped_bids": {},
            "discarded": [],
        }

        # ======================================================
        # DECISION LOG
        # ======================================================
        decisions: List[str] = []

        # ======================================================
        # RESULT (lo que usarán los cálculos)
        # ======================================================
        final_bids: List[float] = []
        used_amounts = set()
        confidences = []
        currency = None

        # ------------------------------------------------------
        # STEP 1: Agrupar por licitador (dato estructural)
        # ------------------------------------------------------
        grouped = defaultdict(list)
        for b in bids:
            bidder = b.get("bidder") or "UNKNOWN_BIDDER"
            grouped[bidder].append(b)

        data["grouped_bids"] = dict(grouped)

        # ------------------------------------------------------
        # STEP 2: Selección por licitador (DECISIONES)
        # ------------------------------------------------------
        for bidder, candidates in grouped.items():

            # --- Regla 1: confianza mínima
            valid = [
                b for b in candidates
                if isinstance(b.get("amount"), (int, float))
                and b.get("confidence", 0) >= self.min_confidence
            ]

            if not valid:
                data["discarded"].extend(candidates)
                decisions.append(
                    f"{bidder}: descartado (ninguna oferta supera confianza mínima)"
                )
                continue

            # --- Regla 2: preferir IVA explícito
            tax_included = [b for b in valid if b.get("tax_included") is True]
            pool = tax_included if tax_included else valid

            if tax_included:
                decisions.append(
                    f"{bidder}: preferidas ofertas con IVA explícito"
                )

            # --- Regla 3: mayor confianza
            selected = sorted(
                pool,
                key=lambda b: b.get("confidence", 0),
                reverse=True
            )[0]

            amount = float(selected["amount"])

            # --- Regla 4: evitar duplicados globales
            if amount in used_amounts:
                data["discarded"].append(selected)
                decisions.append(
                    f"{bidder}: importe {amount} descartado por duplicado"
                )
                continue

            # --------------------------------------------------
            # ACEPTACIÓN
            # --------------------------------------------------
            used_amounts.add(amount)
            final_bids.append(amount)
            confidences.append(selected.get("confidence", 0.5))

            if selected.get("tax_included") is True:
                decisions.append(
                    f"{bidder}: seleccionado importe con IVA ({amount})"
                )
            else:
                decisions.append(
                    f"{bidder}: seleccionado importe sin IVA ({amount})"
                )

            if not currency and selected.get("currency"):
                currency = selected.get("currency")

            # --- Registrar descartes restantes
            for b in candidates:
                if b is not selected:
                    data["discarded"].append(b)

        if not final_bids:
            raise ValueError("No valid bids after consolidation")

        # ------------------------------------------------------
        # STEP 3: Confianza agregada (resultado, no decisión)
        # ------------------------------------------------------
        overall_confidence = round(
            sum(confidences) / len(confidences), 2
        ) if confidences else 0.0

        # ======================================================
        # OUTPUT EXPLÍCITO
        # ======================================================
        return {
            "data": data,               # qué había
            "decisions": decisions,     # qué reglas se aplicaron
            "result": {                 # qué se usa aguas abajo
                "final_bids": final_bids,
                "currency": currency,
                "confidence": overall_confidence,
            },
        }

import numpy as np


class CVAgent:
    """
    Coefficient of Variation (CV) calculation agent.
    """

    def compute(self, bids: list[float]) -> dict:
        if len(bids) < 2:
            raise ValueError("CV requires at least two bids")

        bids_array = np.array(bids, dtype=float)

        mean = bids_array.mean()
        if mean == 0:
            raise ValueError("Mean of bids is zero, CV undefined")

        std = bids_array.std(ddof=1)
        cv = std / mean

        return {
            "metric": "cv",
            "value": float(cv),
            "mean": float(mean),
            "std": float(std),
            "n_bids": len(bids)
        }

import numpy as np


class SPDAgent:
    """
    Spread inside Tender (SPD)

    SPD = (max_bid - min_bid) / min_bid
    """

    def compute(self, bids: list[float]) -> dict:
        if len(bids) < 2:
            raise ValueError("SPD requires at least two bids")

        bids_array = np.array(bids, dtype=float)

        min_bid = bids_array.min()
        max_bid = bids_array.max()

        if min_bid == 0:
            raise ValueError("Minimum bid is zero, SPD undefined")

        spd = (max_bid - min_bid) / min_bid

        return {
            "metric": "spd",
            "value": float(spd),
            "min_bid": float(min_bid),
            "max_bid": float(max_bid),
            "n_bids": len(bids)
        }

import numpy as np


class DIFFPAgent:
    """
    Difference between Two Lowest Bids (DIFFP)

    DIFFP = (second_lowest - lowest) / lowest
    """

    def compute(self, bids: list[float]) -> dict:
        if len(bids) < 2:
            raise ValueError("DIFFP requires at least two bids")

        bids_array = np.sort(np.array(bids, dtype=float))

        lowest = bids_array[0]
        second_lowest = bids_array[1]

        if lowest == 0:
            raise ValueError("Lowest bid is zero, DIFFP undefined")

        diffp = (second_lowest - lowest) / lowest

        return {
            "metric": "diffp",
            "value": float(diffp),
            "lowest_bid": float(lowest),
            "second_lowest_bid": float(second_lowest),
            "n_bids": len(bids)
        }

import numpy as np


class RDAgent:
    """
    Relative Distance (RD)

    RD = (second_lowest - lowest) / std(losing_bids)
    """

    def compute(self, bids: list[float]) -> dict:
        if len(bids) < 3:
            raise ValueError("RD requires at least three bids")

        bids_array = np.sort(np.array(bids, dtype=float))

        lowest = bids_array[0]
        second_lowest = bids_array[1]
        losing_bids = bids_array[1:]

        std_losing = losing_bids.std(ddof=1)
        if std_losing == 0:
            raise ValueError(
                "Standard deviation of losing bids is zero, RD undefined"
            )

        rd = (second_lowest - lowest) / std_losing

        return {
            "metric": "rd",
            "value": float(rd),
            "lowest_bid": float(lowest),
            "second_lowest_bid": float(second_lowest),
            "std_losing_bids": float(std_losing),
            "n_bids": len(bids)
        }

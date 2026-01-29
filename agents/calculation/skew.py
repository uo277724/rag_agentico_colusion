import numpy as np


class SkewnessAgent:
    """
    Skewness (asymmetry)

    Uses the unbiased estimator for skewness.
    """

    def compute(self, bids: list[float]) -> dict:
        if len(bids) < 3:
            raise ValueError("SKEW requires at least three bids")

        bids_array = np.array(bids, dtype=float)
        n = len(bids_array)

        mean = bids_array.mean()
        std = bids_array.std(ddof=1)

        if std == 0:
            raise ValueError("Standard deviation is zero, SKEW undefined")

        standardized = (bids_array - mean) / std
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(standardized ** 3)

        return {
            "metric": "skew",
            "value": float(skewness),
            "mean": float(mean),
            "std": float(std),
            "n_bids": n
        }

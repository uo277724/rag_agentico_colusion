import numpy as np


class KurtosisAgent:
    """
    Excess Kurtosis (KURT)

    Uses the unbiased estimator for excess kurtosis.
    """

    def compute(self, bids: list[float]) -> dict:
        if len(bids) < 4:
            raise ValueError("KURT requires at least four bids")

        bids_array = np.array(bids, dtype=float)
        n = len(bids_array)

        mean = bids_array.mean()
        std = bids_array.std(ddof=1)

        if std == 0:
            raise ValueError("Standard deviation is zero, KURT undefined")

        standardized = (bids_array - mean) / std
        fourth_moment = np.sum(standardized ** 4)

        numerator = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * fourth_moment
        correction = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        kurtosis = numerator - correction

        return {
            "metric": "kurt",
            "value": float(kurtosis),
            "mean": float(mean),
            "std": float(std),
            "n_bids": n
        }

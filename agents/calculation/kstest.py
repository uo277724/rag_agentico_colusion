import numpy as np
from scipy.stats import ks_1samp, uniform


class KSTestAgent:
    """
    Kolmogorov-Smirnov Test (KSTEST)

    Tests whether bids follow a uniform distribution.
    """

    def compute(self, bids: list[float]) -> dict:
        if len(bids) < 3:
            raise ValueError("KSTEST requires at least three bids")

        bids_array = np.array(bids, dtype=float)

        min_val = bids_array.min()
        max_val = bids_array.max()

        if min_val == max_val:
            raise ValueError("All bids are equal, KSTEST undefined")

        normalized = (bids_array - min_val) / (max_val - min_val)

        ks_stat, p_value = ks_1samp(normalized, uniform.cdf)

        return {
            "metric": "kstest",
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "uniform_distribution": bool(p_value > 0.05),
            "n_bids": len(bids)
        }

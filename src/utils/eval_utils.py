import numpy as np
from rliable import library as rly
from rliable import metrics
from scipy.stats import iqr

def compute_metrics(rewards):
    rewards = np.array(rewards)
    avg_returns = rewards.reshape(1, -1)
    
    def aggregate_func(x):
        return np.array([
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
        ])
    
    avg_return_dict = {"evaluation": avg_returns}
    avg_return_scores, avg_return_cis = rly.get_interval_estimates(
        avg_return_dict, aggregate_func, reps=50000, task_bootstrap=True
    )
    iqr_value = iqr(avg_returns)
    std = np.std(avg_returns)
    stats = {
            "median": float(avg_return_scores["evaluation"][0]),
            "iqm": float(avg_return_scores["evaluation"][1]),
            "mean": float(avg_return_scores["evaluation"][2]),
            "iqr": float(iqr_value),
            "std": float(std),
            "median_ci": [
                float(avg_return_cis["evaluation"][0][0]), 
                float(avg_return_cis["evaluation"][1][0])
            ],
            "iqm_ci": [
                float(avg_return_cis["evaluation"][0][1]), 
                float(avg_return_cis["evaluation"][1][1])
            ],
            "mean_ci": [
                float(avg_return_cis["evaluation"][0][2]), 
                float(avg_return_cis["evaluation"][1][2])
            ],
            "num_samples": len(rewards)
        }
    return stats
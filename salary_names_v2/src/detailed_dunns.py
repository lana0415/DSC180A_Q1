import pandas as pd
import numpy as np
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multitest import multipletests

def detailed_dunns(data, val_col, group_col, p_adjust='bonferroni', total_comparisons=1):
    # Calculate group medians
    medians = data.groupby(group_col)[val_col].median()
    means = data.groupby(group_col)[val_col].mean()
    groups = data[group_col].unique()
    
    # Perform Dunn's test without p-value adjustment
    p_values = posthoc_dunn(data, val_col=val_col, group_col=group_col, p_adjust=None)
    
    # Prepare results list to collect all comparisons
    results_list = []

    # Collect data and calculate stats
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            median1 = medians[g1]
            median2 = medians[g2]
            median_diff = median1 - median2
            mean1 = means[g1]
            mean2 = means[g2]
            mean_diff = mean1 - mean2
            mean_ranks1 = data[data[group_col] == g1][val_col].rank() - len(data[data[group_col] == g1]) / 2
            mean_ranks2 = data[data[group_col] == g2][val_col].rank() - len(data[data[group_col] == g2]) / 2
            mean_ranks_diff = mean_ranks1.mean() - mean_ranks2.mean()
            test_stat = np.abs(mean_ranks_diff) / np.sqrt((np.var(mean_ranks1, ddof=1) + np.var(mean_ranks2, ddof=1)) / 2)
            p_value = p_values.loc[g1, g2]

            # Adjust p-values using Bonferroni method
            adj_p_value = 0.05 / total_comparisons
            significant = p_value < adj_p_value
            
            # Append to results list
            results_list.append({
                'Group 1': g1,
                'Group 2': g2,
                'Group 1 Median': median1,
                'Group 2 Median': median2,
                'Group 1 Mean': mean1.round(2),
                'Group 2 Mean': mean2.round(2),
                'Group 1 Mean Rank': mean_ranks1.round(2),
                'Group 2 Mean Rank': mean_ranks2.round(2),
                'Median Difference': median_diff,
                'Mean Difference': mean_diff.round(2),
                'Mean Rank Difference': mean_ranks_diff.round(2),                
                'Test statistic': test_stat.round(3),
                'p-value': p_value,
                'p_less_05': p_value < 0.05/total_comparisons,
                'p_less_0005': p_value < 0.0005/total_comparisons
            })

    # Convert list to DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df

import numpy as np
import pandas as pd
import itertools as it
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations
from statsmodels.stats.multitest import multipletests

def parse_pval(pval):
    try:
        pval = float(pval)
        if pval == 0.0:
            return "<1e-15"
        elif pval < 1e-15:
            return "<1e-15"
        elif pval > 1e-6:
            return "{:.6f}".format(pval)
        else:
            return "{:.2e}".format(pval)
    except Exception as e:
        print(pval, e)

def better_posthoc_dunns(a, val_col=None, group_col=None, p_adjust='bonferroni', sort=True, total_comparisons=1):
    def compare_dunn(i, j):
        # Compute the standard error and Z-value using ranks, as per Dunn's test
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        A = n * (n + 1.) / 12.
        B = (1. / x_lens[i] + 1. / x_lens[j])
        z_value = diff / np.sqrt((A - x_ties) * B)
        p_value = 2 * norm.sf(np.abs(z_value))  # Two-tailed p-value

        # Calculate mean and median differences using actual values
        mean_diff = np.abs(x_means[i] - x_means[j])
        med_diff = np.abs(x_medians[i] - x_medians[j])
        mean_diff_noabs = x_means[i] - x_means[j]
        med_diff_noabs = x_medians[i] - x_medians[j]

        return mean_diff, med_diff, z_value, p_value, mean_diff_noabs, med_diff_noabs

    def __convert_to_df(a, val_col, group_col):
        if isinstance(a, pd.DataFrame):
            return a, val_col, group_col
        else:
            df = pd.DataFrame(a)
            return df, df.columns[-1], df.columns[-2]

    x, _val_col, _group_col = __convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_lens = x.groupby(_group_col)[_val_col].count()

    x['ranks'] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col)['ranks'].mean()
    x_means = x.groupby(_group_col)[_val_col].mean()
    x_medians = x.groupby(_group_col)[_val_col].median()

    vals = x.groupby('ranks').count()[_val_col].values
    tie_sum = np.sum(vals[vals != 1] ** 3 - vals[vals != 1])
    x_ties = tie_sum / (12. * (n - 1)) if tie_sum else 0

    results = []
    for i, j in combinations(x_groups_unique, 2):
        mean_diff, median_diff, z_value, p_value, mean_diff_noabs, med_diff_noabs = compare_dunn(i, j)
        p_adj = p_value * total_comparisons
        # Apply Bonferroni correction for multiple comparisons
        reject_p05 = p_adj < 0.05
        reject_p0005 = p_adj< 0.0005 

        results.append({
            group_col + '1': i,
            group_col + '2': j,
            'median_diff': med_diff_noabs,
            'mean_diff': mean_diff_noabs.round(0),
            'Z_score': z_value.round(2),
            'p_value': p_value,
            'p_adj': p_adj,
            'reject_p05': reject_p05,
            'reject_p0005': reject_p0005
        })

    results = pd.DataFrame(results)
    return results
# Example usage:
# results = better_posthoc_dunns(df, 'query_response_parsed', 'model', total_comparisons=10)
# print(results)

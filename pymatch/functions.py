from __future__ import division
from pymatch import *
import sys
import numpy as np


def drop_static_cols(df, yvar, cols=None):
    if not cols:
        cols = list(df.columns)
    # will be static for both groups
    cols.pop(cols.index(yvar))
    for col in df[cols]:
        n_unique = len(np.unique(df[col]))
        if n_unique == 1:
            df.drop(col, axis=1, inplace=True)
            sys.stdout.write('\rStatic column dropped: {}'.format(col))
    return df
  
  
def ks_boot(tr, co, nboots=1000):
    nx = len(tr)
    w = np.concatenate((tr, co))
    obs = len(w)
    cutp = nx
    bbcount = 0
    ss = []
    fs_ks, _ = stats.ks_2samp(tr, co)
    for bb in range(nboots):
        sw = np.random.choice(w, obs, replace=True)
        x1tmp = sw[:cutp]
        x2tmp = sw[cutp:]
        s_ks, _ = stats.ks_2samp(x1tmp, x2tmp)
        ss.append(s_ks)
        if s_ks >= fs_ks:
            bbcount += 1
    ks_boot_pval = bbcount * 1.0 / nboots
    return ks_boot_pval


def _chi2_distance(tb, cb):
    dist = 0
    for b in set(np.union1d(list(tb.keys()), list(cb.keys()))):
        if b not in tb:
            tb[b] = 0
        if b not in cb:
            cb[b] = 0
        xi, yi = tb[b], cb[b]
        dist += ((xi - yi) ** 2) * 1.0 / (xi + yi)
    return dist * 1.0 / 2


def chi2_distance(t, c):
    tb, cb, bins = which_bin_hist(t, c)
    tb, cb = bin_hist(tb, cb, bins)
    return _chi2_distance(tb,cb)


def which_bin_hist(t, c):
    comb = np.concatenate((t, c))
    bins = np.arange(np.percentile(comb, 99), step=10)
    t_binned = np.digitize(t, bins)
    c_binned = np.digitize(c, bins)
    return t_binned, c_binned, bins


def bin_hist(t, c, bins):
    tc, cc = Counter(t), Counter(c)

    def idx_to_value(d, bins):
        result = {}
        for k, v, in d.items():
            result[int(bins[k-1])] = v
        return result

    return idx_to_value(tc, bins), idx_to_value(cc, bins)


def grouped_permutation_test(f, t, c, n_samples=1000):
    truth = f(t, c)
    comb = np.concatenate((t, c))
    times_geq=0
    samp_arr = []
    for i in range(n_samples):
        tn = len(t)
        combs = comb[:]
        np.random.shuffle(combs)
        tt = combs[:tn]
        cc = combs[tn:]
        sample_truth = f(np.array(tt), np.array(cc))
        if sample_truth >= truth:
            times_geq += 1
        samp_arr.append(sample_truth)
    return (times_geq * 1.0) / n_samples, truth


def std_diff(a, b):
    sd = np.std(a.append(b))
    med = (np.median(a) - np.median(b)) * 1.0 / sd
    mean = (np.mean(a) - np.mean(b)) * 1.0 / sd
    return med, mean


def progress(i, n, prestr=''):
    sys.stdout.write('\r{}: {}\{}'.format(prestr, i, n))


def is_continuous(colname, dmatrix):
    """
    Check if the colname was treated as continuous in the patsy.dmatrix
    Would look like colname[<factor_value>] otherwise
    """
    return (colname in dmatrix.columns) or ("Q('{}')".format(colname) in dmatrix.columns)

def closest_index(a, b):
    n0 = a.shape[0]
    n1 = b.shape[0]
    a = a.scores.sort_values()
    b = b.scores.sort_values()
    results = []
    index1 = 0
    if a.iloc[0] >= b.iloc[0]:
        results.extend([b.index[0], a.index[0]])
        index1 += 1
    index0 = 1

    while (index1 < n1) and (index0 < n0):
        while (index0 + 1 < n0) and (a.iloc[index0 + 1] < b.iloc[index1]):
            index0 += 1
        if (index0+1 < n0) and (a.iloc[index0+1] - b.iloc[index1] <= b.iloc[index1] - a.iloc[index0]): index0 += 1
        results.extend([b.index[index1], a.index[index0]])
        index1 += 1

    if index1 < n1:
        results = results + [a.index[-1]*(n1-index1)]
    match_ids = np.repeat(b.index, 2)
    return results, match_ids

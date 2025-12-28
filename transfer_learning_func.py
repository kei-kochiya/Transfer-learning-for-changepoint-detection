import numpy as np
import math
import qp_solver_fusedlasso as qp_solver    

def form_truth(jump_mean, jump_location, n_new):
    tmp = np.linspace(0, 1, n_new)
    jump_indices = []
    for x in jump_location:
        idx = np.searchsorted(tmp, x)
        jump_indices.append(max(idx, 0))
    
    jump_indices = sorted(list(set(jump_indices)))
    signal = np.zeros(n_new)
    
    boundaries = [0] + [j for j in jump_indices if j > 0 and j < n_new] + [n_new]
    means = list(jump_mean)
    if len(means) < len(boundaries) - 1:
        means = means + [means[-1]] * ((len(boundaries)-1) - len(means))

    for i in range(len(boundaries) - 1):
        signal[boundaries[i]:boundaries[i+1]] = means[i]
    return signal

def construct_P_n0_n(n0_new, n_new):
    P = np.zeros((n0_new, n_new))
    m_ratio = n_new / n0_new
    for i in range(n0_new):
        idx_start = math.ceil(i * m_ratio)
        idx_end = math.ceil((i + 1) * m_ratio)
        indices = list(range(idx_start, idx_end))
        if len(indices) > 0:
            P[i, indices] = 1.0 / len(indices)
    return P

def construct_P_n_n0(n0_new, n_new):
    P = np.zeros((n_new, n0_new))
    m_ratio = n_new / n0_new
    for j in range(n0_new):
        idx_start = math.ceil(j * m_ratio)
        idx_end = math.ceil((j + 1) * m_ratio)
        indices = list(range(idx_start, idx_end))
        if len(indices) > 0:
            P[indices, j] = 1.0
    return P

def f_gen(jump_mean, jump_locs, n0, n_source, size_K, size_A, H_k, 
          sig_delta_1, sig_delta_2, h_1, h_2, exact=True):
    f_0 = form_truth(jump_mean, jump_locs, n0)
    P_up = construct_P_n_n0(n0, n_source)
    W = np.zeros((size_K, n_source))
    
    for k in range(size_K):
        sample_k = np.random.choice(n_source, int(H_k), replace=False)
        projected_f0 = P_up @ f_0
        W[k, :] = projected_f0
        
        is_informative = (k < size_A)
        if is_informative:
            perturbation = sig_delta_1 if exact else np.random.normal(0, h_1/100, size=int(H_k))
        else:
            perturbation = sig_delta_2 if exact else np.random.normal(0, h_2/100, size=int(H_k))
        W[k, sample_k] += perturbation
    return f_0, W

def obs_gen(f_0, W, sigma):
    y_0 = f_0 + np.random.normal(0, sigma, size=len(f_0))
    size_K, n_source = W.shape
    obs_y = np.zeros_like(W)
    for k in range(size_K):
        obs_y[k, :] = W[k, :] + np.random.normal(0, sigma, size=n_source)
    return y_0, obs_y

def get_l1_estimate(y, lam):
    n = y.shape[0]
    no_vars = n + 2 * (n - 1)
    P, q, G, h, A, b = qp_solver.construct_P_q_G_h_A_b(y, lam)
    x, prob = qp_solver.run(P, q, G, h, A, b, no_vars)
    x = x.value
    x = x[:n]
    return x

def calculate_statistic(y_source, y_target, n_source, n_target, t_hat_k):
    P_up = construct_P_n_n0(n_target, n_source)
    delta = (y_source / np.sqrt(n_source)) - ((P_up @ y_target) / np.sqrt(n_source))
    delta_sorted = np.sort(np.abs(delta))[::-1]
    return np.sum(delta_sorted[:t_hat_k]**2)

def estimate_threshold(y_source, y_target, n_source, n_target, t_hat_k, trials=50):
    beta_hat = get_l1_estimate(y_source, lam=0.2) 
    residuals = y_source - beta_hat
    initial_stat = calculate_statistic(y_source, y_target, n_source, n_target, t_hat_k)
    
    stats_permuted = []
    for _ in range(trials):
        res_shuffled = np.random.permutation(residuals)
        y_source_boot = beta_hat + res_shuffled
        stat = calculate_statistic(y_source_boot, y_target, n_source, n_target, t_hat_k)
        stats_permuted.append(stat)
        
    return max(np.quantile(stats_permuted, 0.95), initial_stat)

def algorithm_1_select_sources(obs_y, y_0, n_k, n_0, t_hat_k=50):
    K = obs_y.shape[0]
    selected_indices = []
    
    for k in range(K):
        y_k = obs_y[k, :]
        tau_k = estimate_threshold(y_k, y_0, n_k, n_0, t_hat_k)
        stat_k = calculate_statistic(y_k, y_0, n_k, n_0, t_hat_k)
        
        if stat_k <= tau_k:
            selected_indices.append(k)
    return selected_indices

def combine_target_with_sources(y_target, obs_y, selected_indices, n_target, n_source):
    ratio = n_source / n_target
    
    P_down = construct_P_n0_n(n_target, n_source)
    
    weighted_sum = y_target.copy()
    total_samples = np.ones(n_target) 
    
    for k in selected_indices:
        y_s_aligned = P_down @ obs_y[k, :]
        
        weighted_sum += y_s_aligned * ratio
        total_samples += ratio
            
    y_combined = weighted_sum / total_samples
    
    return y_combined
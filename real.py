from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import transfer_learning_func as tl
import util

n_0 = 200
n_k = 800
K_sources = 2
size_A = 1
sigma = 1.0
h_1 = 40
h_2 = 600

sig_delta_good = 0.05 
sig_delta_bad = 11.0  

jump_locs = [0.2, 0.4, 0.6, 0.8, 0.95]
jump_mean = np.array([1, 5, 2, 6, 1, 6])
H_k = 0.25 * n_k

P_down = tl.construct_P_n0_n(n_0, n_k)
P_down2 = tl.construct_P_n0_n(n_0, n_k + n_0)

epoch = 1000

target_score = good_score = bad_score = good_target_combined_score = bad_target_combined_score = 0

for i in tqdm(range(epoch), desc="Đang xử lý"):
    f_0, W = tl.f_gen(jump_mean, jump_locs, n_0, n_k, K_sources, size_A, H_k, 
                  sig_delta_good, 
                  sig_delta_bad, 
                  h_1, h_2, False)
    y_0, obs_y = tl.obs_gen(f_0, W, sigma)

    good_source_signal = P_down @ W[0, :]
    good_source_obs = P_down @ obs_y[0, :]
    bad_source_signal = P_down @ W[-1, :]
    bad_source_obs = P_down @ obs_y[-1, :]
    y_combined_good = P_down2 @ tl.combine(y_0, obs_y[0, :])
    y_combined_bad = P_down2 @ tl.combine(y_0, obs_y[-1, :])

    f_hat_target_only = tl.get_l1_estimate(y_0, lam=15)
    f_hat_good = tl.get_l1_estimate(good_source_obs, lam=15)
    f_hat_bad = tl.get_l1_estimate(bad_source_obs, lam=15)
    f_hat_combined_good = tl.get_l1_estimate(y_combined_good, lam=15)
    f_hat_combined_bad = tl.get_l1_estimate(y_combined_bad, lam=15)

    true_cp = util.find_list_cp(f_0, f_0.shape[0])
    target_cp_estimated = util.find_list_cp(f_hat_target_only, f_hat_target_only.shape[0])
    good_source_cp_estimated = util.find_list_cp(f_hat_good, f_hat_good.shape[0])
    bad_source_cp_estimated = util.find_list_cp(f_hat_bad, f_hat_bad.shape[0])
    combined_good_cp_estimated = util.find_list_cp(f_hat_combined_good, f_hat_combined_good.shape[0])
    combined_bad_cp_estimated = util.find_list_cp(f_hat_combined_bad, f_hat_combined_bad.shape[0])

    target_mask = np.isin(true_cp, target_cp_estimated)
    good_mask = np.isin(true_cp, good_source_cp_estimated)
    bad_mask = np.isin(true_cp, bad_source_cp_estimated)
    good_target_combined_mask = np.isin(true_cp, combined_good_cp_estimated)
    bad_target_combined_mask = np.isin(true_cp, combined_bad_cp_estimated)

    target_ratio = np.sum(target_mask) / (len(true_cp) + len(target_cp_estimated) - np.sum(target_mask)) 
    good_ratio = np.sum(good_mask) / (len(true_cp) + len(good_source_cp_estimated) - np.sum(good_mask))
    bad_ratio = np.sum(bad_mask) / (len(true_cp) + len(bad_source_cp_estimated) - np.sum(bad_mask))
    good_target_combined_ratio = np.sum(good_target_combined_mask) / (len(true_cp) + len(combined_good_cp_estimated) - np.sum(good_target_combined_mask))
    bad_target_combined_ratio = np.sum(bad_target_combined_mask) / (len(true_cp) + len(combined_bad_cp_estimated) - np.sum(bad_target_combined_mask))

    if target_ratio >= good_ratio and target_ratio >= bad_ratio:
        target_score += 1  
    if good_ratio >= target_ratio and good_ratio >= bad_ratio:
        good_score += 1
    if bad_ratio >= target_ratio and bad_ratio >= good_ratio:
        bad_score += 1  
    if good_target_combined_ratio >= bad_target_combined_ratio:
        good_target_combined_score += 1
    if bad_target_combined_ratio >= good_target_combined_ratio:
        bad_target_combined_score += 1

print(target_score, good_score, bad_score, good_target_combined_score, bad_target_combined_score)

# f_0, W = tl.f_gen(jump_mean, jump_locs, n_0, n_k, K_sources, size_A, H_k, 
#                 sig_delta_good, 
#                 sig_delta_bad, 
#                 h_1, h_2, False)
# y_0, obs_y = tl.obs_gen(f_0, W, sigma)

# P_down = tl.construct_P_n0_n(n_0, n_k)
# P_down2 = tl.construct_P_n0_n(n_0, n_k + n_0)

# good_source_signal = P_down @ W[0, :]
# good_source_obs = P_down @ obs_y[0, :]
# bad_source_signal = P_down @ W[-1, :]
# bad_source_obs = P_down @ obs_y[-1, :]
# y_combined_good = P_down2 @ tl.combine(y_0, obs_y[0, :])
# y_combined_bad = P_down2 @ tl.combine(y_0, obs_y[-1, :])

# f_hat_target_only = tl.get_l1_estimate(y_0, lam=15)
# f_hat_good = tl.get_l1_estimate(good_source_obs, lam=15)
# f_hat_bad = tl.get_l1_estimate(bad_source_obs, lam=15)
# f_hat_combined_good = tl.get_l1_estimate(y_combined_good, lam=15)
# f_hat_combined_bad = tl.get_l1_estimate(y_combined_bad, lam=15)

# true_cp = util.find_list_cp(f_0, f_0.shape[0])
# target_cp_estimated = util.find_list_cp(f_hat_target_only, f_hat_target_only.shape[0])
# good_source_cp_estimated = util.find_list_cp(f_hat_good, f_hat_good.shape[0])
# bad_source_cp_estimated = util.find_list_cp(f_hat_bad, f_hat_bad.shape[0])
# combined_good_cp_estimated = util.find_list_cp(f_hat_combined_good, f_hat_combined_good.shape[0])
# combined_bad_cp_estimated = util.find_list_cp(f_hat_combined_bad, f_hat_combined_bad.shape[0])

# plt.figure(figsize=(14, 8))

# plt.subplot(2, 2, 1)
# plt.plot(good_source_obs, 'g-', alpha=0.5, label='good source obs')
# plt.plot(bad_source_obs, 'r-', alpha=0.5, label='bad source obs')
# plt.plot(f_0, 'k--', linewidth=2, label='Truth')
# plt.title("Good Source obs vs Bad Source obs")
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(good_source_signal, 'g-', alpha=0.5, label='good source signal')
# plt.plot(bad_source_signal, 'r-', alpha=0.5, label='bad source signal')
# plt.plot(f_0, 'k--', linewidth=2, label='Truth')
# plt.title("Good Source signal vs Bad Source signal")
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(f_hat_good, 'g-', alpha=0.5, label='estimation using good source')
# plt.plot(f_hat_target_only, 'r-', alpha=0.5, label='estimation using target')
# plt.plot(f_0, 'k--', linewidth=2, label='Truth')
# plt.title("estimation using good source vs estimation using target")
# plt.legend()

# plt.subplot(2, 2, 4)
# plt.plot(f_hat_bad, 'g-', alpha=0.5, label='estimation using bad source')
# plt.plot(f_hat_target_only, 'r-', alpha=0.5, label='estimation using target')
# plt.plot(f_0, 'k--', linewidth=2, label='Truth')
# plt.title("estimation using bad source vs estimation using target")
# plt.legend()

# plt.subplot(2, 2, 1)
# plt.plot(f_hat_combined_good, 'g-', alpha=0.5, label='estimation using good source and target combined')
# plt.plot(f_hat_target_only, 'r-', alpha=0.5, label='estimation using target')
# plt.plot(f_0, 'k--', linewidth=2, label='Truth')
# plt.title("estimation using good source and target combined vs estimation using target")
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(f_hat_combined_bad, 'g-', alpha=0.5, label='estimation using bad source and target combined')
# plt.plot(f_hat_target_only, 'r-', alpha=0.5, label='estimation using target')
# plt.plot(f_0, 'k--', linewidth=2, label='Truth')
# plt.title("estimation using bad source and target combined vs estimation using target")
# plt.legend()

# plt.tight_layout()
# plt.show()

# print(true_cp)
# print(target_cp_estimated)
# print(good_source_cp_estimated)
# print(bad_source_cp_estimated)
# print(combined_good_cp_estimated)
# print(combined_bad_cp_estimated)
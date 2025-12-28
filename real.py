import numpy as np
import matplotlib.pyplot as plt
import transfer_learning_func as tl
import util

n_0 = 200
n_k = 700
K_sources = 2
size_A = 1
sigma = 1.0
gamma = 1.0
h_1 = 40
h_2 = 600

sig_delta_good = 0.05 
sig_delta_bad = 11.0  

jump_locs = [0.2, 0.4, 0.6, 0.8, 0.95]
jump_mean = np.array([1, 7, 2, 5, 1, 7]) * gamma
H_k = 0.3 * n_k

f_0, W = tl.f_gen(jump_mean, jump_locs, n_0, n_k, K_sources, size_A, H_k, 
                  sig_delta_good, 
                  sig_delta_bad, 
                  h_1, h_2, False)
y_0, obs_y = tl.obs_gen(f_0, W, sigma)

# selected_indices = tl.algorithm_1_select_sources(obs_y, y_0, n_k, n_0, t_hat_k=50)

# if len(selected_indices) > 0:
#     P_down = tl.construct_P_n0_n(n_0, n_k)
#     y_combined = np.zeros(n_0)
    
#     for k in selected_indices:
#         y_aligned = P_down @ obs_y[k, :]
#         y_combined += y_aligned
    
#     y_transfer = y_combined / len(selected_indices)
# else:
#     y_transfer = y_0

P_down = tl.construct_P_n0_n(n_0, n_k)
good_source_signal = P_down @ W[0, :]
good_source_obs = P_down @ obs_y[0, :]
bad_source_signal = P_down @ W[-1, :]
bad_source_obs = P_down @ obs_y[-1, :]

f_hat_target_only = tl.get_l1_estimate(y_0, lam=13)
f_hat_good = tl.get_l1_estimate(good_source_obs, lam=13)
f_hat_bad = tl.get_l1_estimate(bad_source_obs, lam=13)

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(good_source_obs, 'g-', alpha=0.5, label='good source obs')
plt.plot(bad_source_obs, 'r-', alpha=0.5, label='bad source obs')
plt.plot(f_0, 'k--', linewidth=2, label='Truth')
plt.title("Good Source obs vs Bad Source obs")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(good_source_signal, 'g-', alpha=0.5, label='good source signal')
plt.plot(bad_source_signal, 'r-', alpha=0.5, label='bad source signal')
plt.plot(f_0, 'k--', linewidth=2, label='Truth')
plt.title("Good Source signal vs Bad Source signal")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(f_hat_good, 'g-', alpha=0.5, label='estimation using good source')
plt.plot(f_hat_target_only, 'r-', alpha=0.5, label='estimation using target')
plt.plot(f_0, 'k--', linewidth=2, label='Truth')
plt.title("estimation using good source vs estimation using target")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(f_hat_bad, 'g-', alpha=0.5, label='estimation using bad source')
plt.plot(f_hat_target_only, 'r-', alpha=0.5, label='estimation using target')
plt.plot(f_0, 'k--', linewidth=2, label='Truth')
plt.title("estimation using bad source vs estimation using target")
plt.legend()

plt.tight_layout()
plt.show()

true_cp = util.find_list_cp(f_0, f_0.shape[0])
target_cp_estimated = util.find_list_cp(f_hat_target_only, f_hat_target_only.shape[0])
good_source_cp_estimated = util.find_list_cp(f_hat_good, f_hat_good.shape[0])
bad_source_cp_estimated = util.find_list_cp(f_hat_bad, f_hat_bad.shape[0])

print(true_cp)
print(target_cp_estimated)
print(good_source_cp_estimated)
print(bad_source_cp_estimated)

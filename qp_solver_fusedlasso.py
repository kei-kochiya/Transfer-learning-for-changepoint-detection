import cvxpy as cp
import numpy as np

def construct_P_q_G_h_A_b(y, lamda):
    n = y.shape[0]
    X = np.identity(n)
    D = (np.diag([-1] * n, k=0) + np.diag([1] * (n - 1), k=1))[:-1]
    dim_beta = n
    dim_z = n - 1
    no_vars = n + 2 * dim_z

    # construct P
    e_1 = np.hstack((X, np.zeros((n, 2 * dim_z))))
    P = np.dot(e_1.T, e_1)

    # construct q
    e_1 = lamda * np.hstack((np.zeros(dim_beta), np.ones(2 * dim_z)))
    e_2 = np.hstack((np.dot(X.T, y).flatten(), np.zeros(2 * dim_z)))
    q = e_1 - e_2
    q = q.reshape((no_vars, 1))

    # construct G
    G = np.zeros((no_vars, no_vars))
    G[dim_beta:, dim_beta:] = np.zeros((2 * dim_z, 2 * dim_z)) - np.identity(2 * dim_z)

    # construct h
    h = np.zeros((no_vars, 1))

    # construct A
    e_1 = np.hstack((np.identity(dim_z), np.zeros((dim_z, dim_z)) - np.identity(dim_z)))
    A = np.hstack((-D, e_1))

    # construct b
    b = np.zeros((D.shape[0], 1))

    return P, q, G, h, A, b



def run(P, q, G, h, A, b, no_vars):
    x = cp.Variable(no_vars)

    q = q.flatten()
    h = h.flatten()
    b = b.flatten()

    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x),
                      [G @ x <= h,
                       A @ x == b])

    # prob.solve(verbose=True)
    prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, verbose=False)

    return x, prob
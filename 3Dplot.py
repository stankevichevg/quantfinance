# coding=utf-8
import matplotlib.pyplot as plt

import numpy as np

import scipy.integrate as integrate
from matplotlib import cm

def create_distribution_w(params):
    def st_distr_pdf(x):
        f = 0
        for (i, param) in enumerate(params[3:]):
            f += param * np.power(x, i + 1)
        return params[2] * np.exp(f)
    A = params[0]
    k = params[1]
    def distr_pdf(x, t):
        return (1 + A * np.exp(-np.power(k, 2) * t) * np.sin(k * (0.5 - integrate.quad(st_distr_pdf, 0, x)[0])))
    return distr_pdf

def create_distribution_pdf(params):
    def st_distr_pdf(x):
        f = 0
        for (i, param) in enumerate(params[3:]):
            f += param * np.power(x, i + 1)
        return params[2] * np.exp(f)
    A = params[0]
    k = params[1]
    def distr_pdf(x, t):
        return st_distr_pdf(x) * (1 + A * np.exp(-np.power(k, 2) * t) * np.sin(k * (0.5 - integrate.quad(st_distr_pdf, 0, x)[0])))
    return distr_pdf

# params = [1.0/3, 5 * np.pi, -4.692, 11.688, -7.936, 1.209]

# Iteration 13
# [ 1.00900246  1.61334457  0.85157195  1.07405514  0.95207598  1.06925569
#   1.01800942  0.98670262  1.01347658  1.03172111  1.01347053  1.10796528]
# Optimization terminated successfully.    (Exit mode 0)
#             Current function value: -6.90266025355
#             Iterations: 41
#             Function evaluations: 355
#             Gradient evaluations: 41
# [  0.8          7.98365511   0.03610337  -3.22173186  14.19823513
#   -7.33709619]

# params = [  8.00000000e-01,   4.68933136e+01,   1.08023146e-02,  -2.06637257e+01,
#    9.80018362e+01,  -1.22675117e+02,   6.16430648e+01,  -1.11771562e+01]

params = [  8.00000000e-01,
            8.962,
            2.762e-05,
            18.105,
            5.091,
            -6.336,
            -9.272,
            2.86
            ]

params = [  8.00000000e-01,   6.05674767e+00,   1.03814102e-06,  -1.10247410e+02,
   3.96117928e+02,  -4.48043419e+02,   2.12209082e+02,  -3.62852998e+01]
params = [  8.00000000e-01,   1.11555783e+01,   3.01876071e-03,   2.09413377e+01,
  -1.11540221e+01,  -3.41219536e+01,   5.76438807e+01,  -2.85155923e+01]
params = [  7.91171990e-01,   7.61544091e+00,   1.39106937e-03,   1.54006369e+01,
  -1.15453382e+01,  -4.60845846e+01,   9.14219549e+01,  -4.19077571e+01]

def st_distr_pdf(x):
    f = 0
    for (i, param) in enumerate(params[3:]):
        f += param * np.power(x, i + 1)
    return params[2] * np.exp(f)

pdf = create_distribution_pdf(params)
pdf_w = create_distribution_w(params)

x = np.linspace(0.3, 2.5, 150)
t = np.linspace(0, 0.045, 30)

X, T, W, P, S = [], [], [], [], []

for j in range(t.shape[0]):
    if j % 5 == 0:
        P.append([])
    for i in range(x.shape[0]):
        if j == 0:
            S.append(st_distr_pdf(x[i]))
        if j % 5 == 0:
            P[int(j / 5)].append(pdf(x[i], t[j]))
        X.append(x[i])
        T.append(t[j])
        W.append(pdf_w(x[i], t[j]))

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z_W = np.array(X), np.array(T), np.array(W)
Z_S = np.array(S)

ax.plot_trisurf(X, Y, Z_W, cmap=cm.jet, linewidth=0.2, alpha=0.2)

ax.plot(x, ys=np.zeros(x.shape[0]), zs=Z_S, zdir='z', label='t=0', c='r', linewidth=2)

for j, ps in enumerate(P):
    ax.plot(x, ys=np.repeat(t[j * 5], x.shape[0]), zs=np.array(P[j]), zdir='z', label='t=0', c='k' if j == 0 else 'k', linewidth= 2 if j == 0 else 0.5)

plt.show()

d = 4

# [ 1.09399699  0.84419031  0.79841417  1.09051233  0.91066274  0.99308847
#   1.14471561  0.92661277  0.86144638  1.08683492  0.89801384  0.93614292]
#Python code for Newman problem 8.8 - Jake Abraham
import numpy as np
import matplotlib.pyplot as plt

def F(t, Y): #array of the two differential equations for x and y
    x, y = Y
    return np.array([1 - (b + 1) * x + a * (x**2) * y, b * x - a * y * x**2])

# Modified midpoint over [t, t+h] with m substeps (m even)
def mm_step(Func, t, y, H, m):
    hm = H / m
    y0 = y
    y1 = y0 + hm * Func(t, y0)
    yp, yc = y0, y1
    tc = t + hm
    for _ in range(2, m + 1):
        yc, yp, tc = yp + 2*hm*Func(tc, yc), yc, tc + hm
    return 0.5 * (yc + yp + hm * Func(t + H, yc))

def extrapolate(seq, m_seq):
    T = seq.copy()
    diag = [T[0]]
    for k in range(1, len(T)):
        mk = m_seq[k]
        col = T[k]
        for j in range(1, k + 1):
            mj = m_seq[k - j]
            r = (mk / mj)**2
            col = col + (col - T[k - 1]) / (r - 1.0)
        T[k] = col
        diag.append(col)
    return diag

# One BS attempt on [t, t+h]; tol is per-unit-time, so threshold = tol*|h|
def bs_try(F, t, y, h, tol, max_levels=8):
    m_list = [2*(k+1) for k in range(max_levels)]  # 2,4,...,16
    seq = []
    for k, m in enumerate(m_list):
        seq.append(mm_step(F, t, y, h, m))
        diag = extrapolate(seq, m_list[:k+1])
        if k >= 1:
            err = np.max(np.abs(diag[-1] - diag[-2]))
            if err <= tol * abs(h):
                return True, diag[-1], err, k+1
    return False, None, np.inf, max_levels

# Integrate from t0 to t1 with initial H; halve H on failure
def integrate_bs(F, t0, t1, y0, H, tol):
    t = t0
    y = np.array(y0, dtype=float)
    ts, ys = [t], [y.copy()]
    H = np.sign(t1 - t0) * abs(H)
    while (t - t1) * np.sign(H) < 0:
        h = np.sign(H) * min(abs(H), abs(t1 - t))
        ok, yn, err, _ = bs_try(F, t, y, h, tol, max_levels=8)
        if ok:
            t += h
            y = yn
            ts.append(t); ys.append(y.copy())
            # keep H as is (spec); could grow/shrink if desired
        else:
            H *= 0.5
            if abs(H) < 1e-16:
                raise RuntimeError("Step size underflow.")
    return np.array(ts), np.vstack(ys)

a, b = 1, 3
x_i = y_i = 0
t0, t1 = 0.0, 20.0
H = 20.0
tolerance = 1e-9
t, Y = integrate_bs(F, t0, t1, (x_i, y_i), H, tolerance)
x, y = Y[:,0], Y[:,1]

plt.figure(figsize=(8,5))
plt.plot(t, x, label='x(t)', linewidth=0.5)
plt.plot(t, y, label='y(t)', linewidth=0.5)
plt.plot(t, x, 'o', markersize=1)
plt.plot(t, y, 'o', markersize=1)
plt.xlabel('time t')
plt.ylabel('value')
plt.title('Bulirsch–Stoer Integration')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

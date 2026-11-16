import math
import os
import random
import time

import matplotlib.pyplot as plt

from tsp_helpers import random_cities, tour_length

os.makedirs("figures", exist_ok=True)
random.seed(7)


def simulated_annealing(points, initial_tour, cooling_fn, T0, max_iter):
    n = len(initial_tour)
    cur = initial_tour[:]
    cur_len = tour_length(points, cur)
    best, best_len = cur[:], cur_len
    history = [(0, cur_len, T0)]
    T = T0
    t0 = time.perf_counter()

    for it in range(1, max_iter + 1):
        T = cooling_fn(it, T0)
        if T <= 1e-8:
            break
        i = random.randint(1, n - 2)
        k = random.randint(i + 1, n - 1)
        if k - i < 2:
            continue
        cand = cur[:i] + cur[i:k + 1][::-1] + cur[k + 1:]
        L = tour_length(points, cand)
        delta = L - cur_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            cur, cur_len = cand, L
            if L < best_len:
                best, best_len = cand[:], L
        if it % 50 == 0:
            history.append((it, best_len, T))

    return best, best_len, history, time.perf_counter() - t0


def linear(it, T0):
    return max(T0 * (1 - it / 20_000), 1e-9)

def exponential(it, T0):
    return T0 * (0.995 ** it)

def logarithmic(it, T0):
    return T0 / math.log(1 + it + 1e-9)


if __name__ == "__main__":
    pts = random_cities(20, seed=123)

    schedules = {
        "Lineal (T0=100)":      linear,
        "Exponencial (α=0.995)": exponential,
        "Logarítmica (T0=100)":  logarithmic,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    results = {}

    for name, fn in schedules.items():
        random.seed(11)  
        init = list(range(20)); random.shuffle(init)
        T0 = 100.0
        best, L, hist, dt = simulated_annealing(pts, init, fn, T0, max_iter=20_000)
        results[name] = (L, dt)
        its, lens = zip(*[(h[0], h[1]) for h in hist])
        ax1.plot(its, lens, label=f"{name} → L={L:.2f}")
        print(f"{name:24s} → L = {L:8.2f}   tiempo = {dt:5.2f}s")

    ax1.set_xlabel("Iteración")
    ax1.set_ylabel("Mejor longitud encontrada")
    ax1.set_title("Curvas de convergencia")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    names = list(results.keys())
    times  = [results[n][1] for n in names]
    ls     = [results[n][0] for n in names]
    ax2.scatter(times, ls, s=120)
    for t, l, n in zip(times, ls, names):
        ax2.annotate(n.split(" (")[0], (t, l), textcoords="offset points",
                     xytext=(8, 5), fontsize=9)
    ax2.set_xlabel("Tiempo de ejecución (s)")
    ax2.set_ylabel("Longitud final del tour")
    ax2.set_title("Trade-off tiempo vs. calidad")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Ejercicio 4 — Simulated Annealing con 2-opt")
    plt.tight_layout()
    plt.savefig("figures/ex4_annealing.png", dpi=150)
    plt.show()
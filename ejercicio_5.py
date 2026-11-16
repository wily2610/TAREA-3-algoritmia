import math
import os
import random

import matplotlib.pyplot as plt

from tsp_helpers import (tour_length, two_opt, brute_force_optimal,
                         nearest_neighbor, random_cities)

os.makedirs("figures", exist_ok=True)


def mst_prim(points):
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children, root=0):
    return [root] + [c for ch in children[root] for c in preorder(children, ch)]


if __name__ == "__main__":
    N, CITIES = 100, 9
    ratios_mst, L_mst, L_mst_2opt, L_nn, L_opt = [], [], [], [], []

    for trial in range(N):
        pts = random_cities(CITIES, seed=1000 + trial)
        approx = preorder(mst_prim(pts))
        opt_tour, opt_len = brute_force_optimal(pts)

        L_approx = tour_length(pts, approx)
        _, _, L_improved = two_opt(pts, approx)

        L_mst.append(L_approx)
        L_mst_2opt.append(L_improved)
        L_nn.append(tour_length(pts, nearest_neighbor(pts, start=0)))
        L_opt.append(opt_len)
        ratios_mst.append(L_approx / opt_len)

    print(f"Razón MST/óptimo: media={sum(ratios_mst)/N:.3f}  "
          f"máx={max(ratios_mst):.3f}  (cota teórica = 2.0)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(ratios_mst, bins=15, edgecolor="black", color="#3b82f6", alpha=0.85)
    ax1.axvline(2.0, color="red", linestyle="--", label="Cota teórica = 2.0")
    ax1.axvline(sum(ratios_mst)/N, color="green", linestyle="-.",
                label=f"Media = {sum(ratios_mst)/N:.2f}")
    ax1.set_xlabel("Razón C_MST / C_óptimo")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Distribución de la razón de aproximación (MST)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    ax2.boxplot([L_mst, L_mst_2opt, L_nn, L_opt],
                tick_labels=["MST puro", "MST + 2-opt", "NN", "Óptimo (bf)"])
    ax2.set_ylabel("Longitud del tour")
    ax2.set_title("Calidad de cada método (100 instancias)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Ejercicio 5 — MST sobre {N} instancias de {CITIES} ciudades")
    plt.tight_layout()
    plt.savefig("figures/ex5_mst.png", dpi=150)
    plt.show()

    print(f"Media MST puro   : {sum(L_mst)/N:.2f}")
    print(f"Media MST + 2opt : {sum(L_mst_2opt)/N:.2f}")
    print(f"Media NN         : {sum(L_nn)/N:.2f}")
    print(f"Media óptimo     : {sum(L_opt)/N:.2f}")
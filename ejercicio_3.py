import os
import random
import matplotlib.pyplot as plt

from tsp_helpers import (random_cities, nearest_neighbor, two_opt,
                         tour_length)

os.makedirs("figures", exist_ok=True)
random.seed(2024)

N_CITIES, N_TRIALS = 20, 25
improvements_random, improvements_nn = [], []
length_random,     length_nn        = [], []

for trial in range(N_TRIALS):
    pts = random_cities(N_CITIES, seed=trial)


    rand_tour = list(range(N_CITIES)); random.shuffle(rand_tour)
    L0_rand = tour_length(pts, rand_tour)
    t1, imp_r, L1 = two_opt(pts, rand_tour)
    improvements_random.append(imp_r)
    length_random.append(L1)

    start = random.randrange(N_CITIES)
    nn_tour = nearest_neighbor(pts, start=start)
    L0_nn = tour_length(pts, nn_tour)
    t2, imp_n, L2 = two_opt(pts, nn_tour)
    improvements_nn.append(imp_n)
    length_nn.append(L2)

    print(f"trial {trial:>2}: "
          f"rand L0={L0_rand:7.2f} → L={L1:7.2f} (mejoras={imp_r:>3}) | "
          f"NN   L0={L0_nn:7.2f} → L={L2:7.2f} (mejoras={imp_n:>3})")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].boxplot([improvements_random, improvements_nn],
                tick_labels=["Aleatorio", "Vecino más cercano"])
axes[0].set_ylabel("# mejoras 2-opt hasta óptimo local")
axes[0].set_title("Mejoras necesarias para converger")
axes[0].grid(True, alpha=0.3)

axes[1].boxplot([length_random, length_nn],
                tick_labels=["Aleatorio + 2-opt", "NN + 2-opt"])
axes[1].set_ylabel("Longitud del tour final")
axes[1].set_title("Calidad del óptimo local alcanzado")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Ejercicio 3 — 2-opt: aleatorio vs. NN (20 ciudades, 25 trials)")
plt.tight_layout()
plt.savefig("figures/ex3_2opt.png", dpi=150)
plt.show()

print(f"\nMedia mejoras  — aleatorio: {sum(improvements_random)/N_TRIALS:.1f}")
print(f"Media mejoras  — NN:        {sum(improvements_nn)/N_TRIALS:.1f}")
print(f"Media longitud — aleatorio: {sum(length_random)/N_TRIALS:.2f}")
print(f"Media longitud — NN:        {sum(length_nn)/N_TRIALS:.2f}")
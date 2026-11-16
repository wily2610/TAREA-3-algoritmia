import multiprocessing as mp
import os
import random
import time
import matplotlib.pyplot as plt

def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    rng = random.Random(seed)
    inside = sum(rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples

def run_seq(seeds, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        [estimate_pi(s) for s in seeds]
        best = min(best, time.perf_counter() - t0)
    return best

def run_par(seeds, procs, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        with mp.Pool(processes=procs) as pool:
            pool.map(estimate_pi, seeds)
        best = min(best, time.perf_counter() - t0)
    return best

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    seeds = list(range(8))
    procs_list = [1, 2, 4, 8]

    print("Iniciando prueba secuencial...")
    t_seq = run_seq(seeds)
    print(f"Secuencial: {t_seq:.3f}s  (cores disponibles: {mp.cpu_count()})")

    times = {}
    for p in procs_list:
        print(f"Ejecutando con {p} procesos...")
        times[p] = run_par(seeds, p)
        print(f"  procs={p:>2}  {times[p]:.3f}s   speedup={t_seq/times[p]:.2f}x")

    speedups = [t_seq / times[p] for p in procs_list]
    ideal    = [min(p, len(seeds)) for p in procs_list]

   #GRAFICA
    plt.figure(figsize=(8, 5))
    plt.plot(procs_list, speedups, "o-", color="#3b82f6", linewidth=2, label="Medido")
    plt.plot(procs_list, ideal, "k--", label="Ideal (Amdahl sin overhead)")
    
    plt.xlabel("Número de Procesos")
    plt.ylabel("Speedup  (T_seq / T_par)")
    plt.title("Ejercicio 1 — Speedup vs. procesos (Monte Carlo π)")
    plt.xticks(procs_list)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("figures/ex1_speedup.png", dpi=150)
    print("\nGráfica guardada exitosamente en 'figures/ex1_speedup.png'")
    

    plt.show()
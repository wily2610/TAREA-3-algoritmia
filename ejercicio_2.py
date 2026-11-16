import os
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def greedy_knapsack(items, values, weights, capacity, priority):
    """priority(i) = clave de orden (mayor primero)."""
    order = sorted(range(len(items)), key=priority, reverse=True)
    chosen, tw, tv = [], 0, 0
    for i in order:
        if tw + weights[i] <= capacity:
            chosen.append(i)
            tw += weights[i]
            tv += values[i]
    return sorted(chosen), tw, tv


def by_value_weight(vals, wts):
    return lambda i: vals[i] / wts[i]


def by_least_weight(vals, wts):
    return lambda i: -wts[i]        


items1    = ["A", "B", "C"]
values1   = [60, 100, 120]
weights1  = [10,  20,  30]
cap1      = 50


items2   = ["A", "B", "C", "D"]
values2  = [1,  100, 100, 100]
weights2 = [1,   15,  15,  15]
cap2     = 30


def report_instance(name, items, vals, wts, cap):
    s_vw = greedy_knapsack(items, vals, wts, cap, by_value_weight(vals, wts))
    s_lw = greedy_knapsack(items, vals, wts, cap, by_least_weight(vals, wts))
    print(f"\n== {name} (capacidad={cap}) ==")
    for label, (ch, tw, tv) in [("valor/peso", s_vw), ("menor peso", s_lw)]:
        print(f"  {label:12s}: items={[items[i] for i in ch]}, "
              f"peso={tw}/{cap}, valor={tv}")
    return s_vw, s_lw


r1 = report_instance("Instancia 1 (original)", items1, values1, weights1, cap1)
r2 = report_instance("Instancia 2 (contraejemplo)", items2, values2, weights2, cap2)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (title, items, r_vw, r_lw) in zip(
    axes,
    [("Instancia 1", items1, r1[0], r1[1]),
     ("Instancia 2 (contraejemplo)", items2, r2[0], r2[1])],
):
    labels = ["valor/peso", "menor peso"]
    valores = [r_vw[2], r_lw[2]]
    pesos   = [r_vw[1], r_lw[1]]
    x = range(2)
    ax.bar([i - 0.2 for i in x], valores, width=0.4, label="Valor total", color="#3b82f6")
    ax.bar([i + 0.2 for i in x], pesos,   width=0.4, label="Peso usado", color="#f59e0b")
    ax.set_xticks(list(x), labels)
    ax.set_title(title)
    ax.set_ylim(0, max(valores + pesos) * 1.25)
    for i, (v, p) in enumerate(zip(valores, pesos)):
        ax.text(i - 0.2, v + 1, str(v), ha="center", fontsize=10)
        ax.text(i + 0.2, p + 1, str(p), ha="center", fontsize=10)
    ax.legend()

plt.suptitle("Ejercicio 2 — Knapsack voraz: valor/peso vs. menor peso")
plt.tight_layout()
plt.savefig("figures/ex2_knapsack.png", dpi=150)
plt.show()
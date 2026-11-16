import math, os, random, json
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from ejercicio_4 import simulated_annealing, exponential
from tsp_helpers import random_cities

MATRICULA = 1823198
SIZES = [5, 10, 15]
REPS, MAX_ITER = 10, 5000
T_MIN, T_MAX = 1.0, 100.0
os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)

# niveles aleatorios 
random.seed(MATRICULA); r = random.random()
fracs = {"bajo": 0.3*r, "medio": 0.3*r + 0.3, "alto": 0.3*r + 0.6}
temps = {k: T_MIN + f*(T_MAX - T_MIN) for k, f in fracs.items()}
print("r =", r, temps)

# experimento: 3 temps x 3 tamanos x 10 replicas = 90 corridas 
rows = []
for n in SIZES:
    pts = random_cities(n, seed=123)
    for lvl, T0 in temps.items():
        for rep in range(REPS):
            seed = MATRICULA + 10_000*n + 100*list(temps).index(lvl) + rep
            random.seed(seed)
            init = list(range(n)); random.shuffle(init)
            _, L, _, _ = simulated_annealing(pts, init, exponential, T0, MAX_ITER)
            rows.append(dict(n=n, nivel=lvl, temperatura=round(T0, 4), rep=rep, seed=seed, costo=L))
df = pd.DataFrame(rows); df.to_csv("results/doe_corridas.csv", index=False)

# ANOVA de dos factores con interaccion 
def anova2(df, a="nivel", b="n", y="costo"):
    ga, gb = df[a].nunique(), df[b].nunique(); N = len(df); rr = N // (ga*gb)
    gm = df[y].mean()
    ssa = sum(len(g)*(g[y].mean()-gm)**2 for _, g in df.groupby(a))
    ssb = sum(len(g)*(g[y].mean()-gm)**2 for _, g in df.groupby(b))
    ssc = sum(len(g)*(g[y].mean()-gm)**2 for _, g in df.groupby([a, b]))
    ssab = ssc - ssa - ssb
    sse = sum(((g[y]-g[y].mean())**2).sum() for _, g in df.groupby([a, b]))
    dfs = dict(A=ga-1, B=gb-1, AB=(ga-1)*(gb-1), E=N-ga*gb)
    ms = dict(A=ssa/dfs["A"], B=ssb/dfs["B"], AB=ssab/dfs["AB"], E=sse/dfs["E"])
    out = []
    for k, name, ss in [("A", "Temperatura", ssa), ("B", "Tamano", ssb), ("AB", "Temperatura x Tamano", ssab)]:
        F = ms[k]/ms["E"]; p = stats.f.sf(F, dfs[k], dfs["E"])
        out.append(dict(fuente=name, SC=ss, gl=dfs[k], CM=ms[k], F=F, p=p))
    out.append(dict(fuente="Error", SC=sse, gl=dfs["E"], CM=ms["E"], F=np.nan, p=np.nan))
    return pd.DataFrame(out)

tabla = anova2(df); tabla.to_csv("results/anova.csv", index=False); print(tabla)
sig_T = tabla.p[0] < 0.05; sig_int = tabla.p[2] < 0.05

# ANOVA 
por_n = []
for n in SIZES:
    g = [d.costo.values for _, d in df[df.n == n].groupby("nivel")]
    if all(np.var(x) == 0 for x in g) and len({x[0] for x in g}) == 1:
        F, p = np.nan, np.nan
    else:
        F, p = stats.f_oneway(*g)
    por_n.append(dict(n=n, F=F, p=p))
por_n = pd.DataFrame(por_n); print(por_n)

# bootstrap 
rng = np.random.default_rng(MATRICULA)
def boot_med(x, B=10_000):
    m = np.median(rng.choice(x, (B, len(x))), axis=1)
    return np.median(m), *np.percentile(m, [2.5, 97.5])
bt = []
for (n, lvl), g in df.groupby(["n", "nivel"]):
    med, lo, hi = boot_med(g.costo.values)
    bt.append(dict(n=n, nivel=lvl, temperatura=g.temperatura.iloc[0], media=g.costo.mean(),
                   mediana_boot=med, ic_inf=lo, ic_sup=hi,
                   significativo=bool(sig_T or sig_int or (por_n.set_index("n").p.get(n, 1) < 0.05))))
bt = pd.DataFrame(bt); bt.to_csv("results/bootstrap.csv", index=False)

orden = {"bajo": 0, "medio": 1, "alto": 2}
best = (bt.assign(o=bt.nivel.map(orden)).sort_values(["n", "mediana_boot", "media", "o"])
          .groupby("n").first().reset_index().drop(columns="o"))
best.to_csv("results/mejor_temperatura.csv", index=False); print(best)

# graficas
niveles = list(temps)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
m = df.groupby("nivel").costo.mean().reindex(niveles)
ax[0].plot([round(temps[k], 1) for k in niveles], m.values, "o-")
ax[0].set(title="Efecto principal: temperatura", xlabel="T0", ylabel="Costo medio"); ax[0].grid(alpha=.3)
m = df.groupby("n").costo.mean()
ax[1].plot(m.index, m.values, "s-", color="C1")
ax[1].set(title="Efecto principal: tamano", xlabel="n", ylabel="Costo medio", xticks=SIZES); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/doe_efectos_principales.png", dpi=150); plt.show()

plt.figure(figsize=(6.5, 4.5))
for n in SIZES:
    m = df[df.n == n].groupby("nivel").costo.mean().reindex(niveles)
    plt.plot([round(temps[k], 1) for k in niveles], m.values, "o-", label=f"n={n}")
plt.xlabel("T0"); plt.ylabel("Costo medio"); plt.title("Interaccion temperatura x tamano")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("figures/doe_interaccion.png", dpi=150); plt.show()

json.dump(dict(r=r, temps=temps, fracs=fracs, sig_T=bool(sig_T), sig_int=bool(sig_int)),
          open("results/meta.json", "w"), indent=1)

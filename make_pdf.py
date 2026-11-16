import json, pandas as pd, numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
meta = json.load(open("results/meta.json")); an = pd.read_csv("results/anova.csv")
bt = pd.read_csv("results/bootstrap.csv"); best = pd.read_csv("results/mejor_temperatura.csv")
st = getSampleStyleSheet(); S = []
P = lambda t, s="BodyText": S.append(Paragraph(t, st[s]))
def T(data, w=None):
    t = Table(data, colWidths=w, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))
    S.append(t); S.append(Spacer(1,8))
f = lambda x, d=3: "-" if pd.isna(x) else (f"{x:.2e}" if (abs(x) < 1e-3 and x != 0) else f"{x:.{d}f}")
P("Tarea 3 - Ajuste de parametros con DOE (Opcion A)", "Title")
P("Recocido Simulado con 2-opt sobre TSP. Matricula 1823198. Repositorio: <b>[PEGA AQUI LA URL PUBLICA DE TU REPOSITORIO]</b>")
P("1. Diseno experimental", "Heading2")
P(f"Factor: temperatura inicial T0 (enfriamiento exponencial, alfa=0.995, 5000 iteraciones). Bloque: n en {{5,10,15}} (una instancia por tamano, semilla 123). "
  f"Diseno factorial 3x3 con 10 replicas por punto (90 corridas), semillas distintas por corrida. Rango [T_min, T_max] = [1, 100]. "
  f"Con random.seed(1823198), r = {meta['r']:.6f} (fijo en todo el experimento). Niveles: T = T_min + fraccion*(T_max - T_min).")
T([["Nivel","Fraccion","T0"]] + [[k, f"{meta['fracs'][k]:.4f}", f"{meta['temps'][k]:.2f}"] for k in meta["temps"]])
P("2. ANOVA de dos factores (alfa = 0.05)", "Heading2")
T([["Fuente","SC","gl","CM","F","p"]] + [[r.fuente,f(r.SC,1),int(r.gl),f(r.CM,1),f(r.F),f(r.p)] for r in an.itertuples()])
pT, pB, pI = an.p[0], an.p[1], an.p[2]
P(f"El tamano del problema es altamente significativo (p = {pB:.1e}), como se esperaba, pues el costo crece con n. "
  f"La temperatura <b>no</b> resulto significativa al nivel global (p = {pT:.3f}), ni la interaccion (p = {pI:.3f}). "
  "Aun asi, el p de la temperatura (0.072) es cercano a 0.05, por lo que se complemento con ANOVA de un factor dentro de cada tamano (bloque): "
  "n=5: p=0.322; n=10: p=0.024 (significativo); n=15: p=0.130.")
S.append(Image("figures/doe_efectos_principales.png", 460, 168)); S.append(Image("figures/doe_interaccion.png", 300, 208))
P("3. Mejor temperatura por tamano (mediana bootstrap, IC 95%, 10,000 remuestreos)", "Heading2")
T([["n","T0 optima (nivel)","Mediana bootstrap","IC 95%","ANOVA 1 factor"]] +
  [[int(r.n), f"{r.temperatura:.2f} ({r.nivel})", f"{r.mediana_boot:.2f}", f"[{r.ic_inf:.2f}, {r.ic_sup:.2f}]", ["p=0.322 (n.s.)","p=0.024 (sig.)","p=0.130 (n.s.)"][i]] for i, r in enumerate(best.itertuples())])
P("El criterio fue elegir el nivel con menor mediana bootstrap del costo (desempate por media). Solo n=10 tiene un efecto de temperatura estadisticamente significativo; "
  "para n=5 y n=15 la mejor temperatura es solo la de menor mediana y debe leerse como indicativa. Medianas de todos los puntos:")
T([["n","Nivel","T0","Media","Mediana boot","IC inf","IC sup"]] + [[int(r.n), r.nivel, f"{r.temperatura:.2f}", f"{r.media:.2f}", f"{r.mediana_boot:.2f}", f"{r.ic_inf:.2f}", f"{r.ic_sup:.2f}"] for r in bt.itertuples()])
P("4. Analisis", "Heading2")
P("La temperatura optima no sigue una tendencia monotona clara con n: es el nivel medio (35.4) para n=5 y n=10, y el alto (65.1) para n=15. "
  "La unica senal con soporte estadistico es la de n=10; para los demas tamanos los ICs de las temperaturas se traslapan, de modo que no se puede afirmar que una temperatura sea realmente mejor. "
  "Que el nivel bajo (5.7) tienda a ser peor es coherente con que una T0 pequena respecto a las diferencias de costo (decenas de unidades) vuelve al algoritmo casi voraz y propenso a quedar atrapado en minimos locales, "
  "mientras que en n=15 (mas diferencias de costo y mas iteraciones necesarias) una T0 mayor explora mas. Como la interaccion no fue significativa (p=0.207), no hay evidencia global de que el efecto de T0 dependa de n. "
  "Limitaciones: una sola instancia por tamano, 10 replicas y presupuesto fijo de 5000 iteraciones; un rango de T mas amplio o mas replicas darian conclusiones mas firmes.")
SimpleDocTemplate("Tarea3_DOE_reporte.pdf", pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40).build(S)

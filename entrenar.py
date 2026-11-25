"""Tarpuy Analytics ML: segmentación de participantes con K-Means.

Genera segmentos.json, el archivo que el organizador carga en el panel de analítica
(POST /api/admin/analytics/importar). Ver ml/README.md.

Uso:
    python entrenar.py --excel dataset_tarpuy_inscritos_2026.xlsx       # hoja Dataset_ML
    python entrenar.py --db "$DATABASE_URL"                              # inscripciones reales de PostgreSQL
Opciones: --k 4 (número de clústeres), --salida segmentos.json, --solo-reales
"""
import argparse
import json
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
VARIABLES = ["Grado_Num", "Tam_IE", "Con_Asesor", "Dias_Antes_Cierre", "Es_Libre"]
GRADOS = ["Inicial · 4 años", "Inicial · 5 años"] + [f"{i}° de Primaria" for i in range(1, 7)] + [
    f"{i}° de Secundaria" for i in range(1, 6)]
GRADO_NUM = {g: i + 1 for i, g in enumerate(GRADOS)}
CIERRE = pd.Timestamp("2026-10-31")
MINIMO_AGREGADO = 5  # RN-03: no se muestran conteos menores a 5
SIN_ASESOR = {"", "SIN TUTOR", "SIN ASESOR", "NINGUNO"}


# ---------------------------------------------------------------- 1. Extracción
def leer_excel(ruta):
    df = pd.read_excel(ruta, sheet_name="Dataset_ML")
    faltan = set(VARIABLES[:-1] + ["ID", "Modalidad", "Nivel", "Fecha", "Origen"]) - set(df.columns)
    if faltan:
        sys.exit(f"Faltan columnas en Dataset_ML: {sorted(faltan)}")
    return df


def leer_db(url, cierre):
    import psycopg2  # solo se necesita con --db

    warnings.filterwarnings("ignore", message=".*SQLAlchemy.*")  # pandas acepta la conexión de psycopg2

    with psycopg2.connect(url) as con:
        alumnos = pd.read_sql(
            "SELECT a.id, a.ficha_id, a.nivel, a.grado, f.ie, f.tipo, f.medio, f.verificado, f.created_at "
            "FROM alumnos a JOIN fichas f ON f.id = a.ficha_id", con)
        asesores = pd.read_sql("SELECT ficha_id, nivel, nombre FROM asesores", con)
        libres = pd.read_sql("SELECT id, nivel, grado, medio, verificado, created_at FROM libres", con)

    asesores["valido"] = ~asesores["nombre"].fillna("").str.strip().str.upper().isin(SIN_ASESOR)
    con_asesor = set(map(tuple, asesores.loc[asesores["valido"], ["ficha_id", "nivel"]].values))
    tam = alumnos.groupby("ficha_id")["id"].transform("size")
    a = pd.DataFrame({
        "ID": "A-" + alumnos["id"].astype(str), "Modalidad": "Institucional", "IE": alumnos["ie"],
        "Tipo_IE": alumnos["tipo"].replace({"Estatal": "Público"}), "Nivel": alumnos["nivel"],
        "Grado": alumnos["grado"], "Tam_IE": tam,
        "Con_Asesor": [int((f, n) in con_asesor) for f, n in zip(alumnos["ficha_id"], alumnos["nivel"])],
        "Medio_Pago": alumnos["medio"], "Estado_Pago": np.where(alumnos["verificado"], "Verificado", "Pendiente"),
        "Fecha": alumnos["created_at"]})
    l = pd.DataFrame({
        "ID": "L-" + libres["id"].astype(str), "Modalidad": "Libre", "IE": "Independiente",
        "Tipo_IE": "Independiente", "Nivel": libres["nivel"], "Grado": libres["grado"], "Tam_IE": 1,
        "Con_Asesor": 0, "Medio_Pago": libres["medio"],
        "Estado_Pago": np.where(libres["verificado"], "Verificado", "Pendiente"), "Fecha": libres["created_at"]})
    df = pd.concat([a, l], ignore_index=True)
    df["Fecha"] = pd.to_datetime(df["Fecha"], utc=True).dt.tz_convert("America/Lima").dt.tz_localize(None)
    df["Grado_Num"] = df["Grado"].map(GRADO_NUM)
    df["Dias_Antes_Cierre"] = (cierre.normalize() - df["Fecha"].dt.normalize()).dt.days.clip(lower=0)
    df["Origen"] = "Real"
    return df


# ---------------------------------------------------------------- 2-4. Anonimización, limpieza y variables
def preparar(df):
    df = df.drop(columns=[c for c in ["DNI", "Apellidos y nombres", "Nombre", "Telefono", "Celular"] if c in df],
                 errors="ignore").copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Es_Libre"] = (df["Modalidad"] == "Libre").astype(int)
    if df["Grado_Num"].isna().any():
        sys.exit("Hay grados fuera del catálogo; revisa la columna Grado.")
    validacion = {
        "registros": int(len(df)),
        "nulos": int(df[VARIABLES].isna().sum().sum()),
        "duplicados": int(df.duplicated(subset=["ID"]).sum()),
        "pagos_efectivo": int((df.get("Medio_Pago") == "Efectivo").sum()) if "Medio_Pago" in df else 0,
        "inicial_4_anios": int((df["Grado"] == "Inicial · 4 años").sum()),
    }
    return df, validacion


# ---------------------------------------------------------------- 5. Resumen descriptivo (solo agregados)
def contar(serie):
    return {str(k): int(v) for k, v in serie.value_counts().sort_index().items()}


def resumen(df):
    diarias = df.groupby([df["Fecha"].dt.date, "Origen"]).size().unstack(fill_value=0)
    return {
        "participantes": int(len(df)),
        "por_origen": contar(df["Origen"]),
        "instituciones": int(df.loc[df["Modalidad"] != "Libre", "IE"].nunique()),
        "por_nivel": contar(df["Nivel"]),
        "por_modalidad": contar(df["Modalidad"]),
        "por_medio_pago": contar(df["Medio_Pago"]) if "Medio_Pago" in df else {},
        "diarias": [{"fecha": str(f), **{str(o): int(diarias.loc[f, o]) for o in diarias.columns}}
                    for f in diarias.index],
    }


# ---------------------------------------------------------------- 6-9. Modelado y evaluación
def metricas_k(X, k_max):
    filas = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(X)
        filas.append({"k": k, "inercia": round(float(km.inertia_), 1),
                      "silueta": round(float(silhouette_score(X, km.labels_)), 3),
                      "davies_bouldin": round(float(davies_bouldin_score(X, km.labels_)), 3)})
    return filas


def nombrar(p):
    if p["Es_Libre"] >= 0.5:
        return "Alumnos libres"
    tam = "grandes" if p["Tam_IE"] >= 45 else "medianas" if p["Tam_IE"] >= 20 else "pequeñas"
    nombre = f"Delegaciones {tam} {'con' if p['Con_Asesor'] >= 0.5 else 'sin'} asesor"
    return nombre + (", inscripción anticipada" if p["Dias_Antes_Cierre"] >= 7 else "")


def describir(p, n, reales):
    if p["Es_Libre"] >= 0.5:
        base = "Estudiantes inscritos por su cuenta, sin delegación"
    else:
        base = (f"Participantes de delegaciones de unos {p['Tam_IE']:.0f} inscritos; "
                f"el {100 * p['Con_Asesor']:.0f} % tiene asesor en su nivel")
    dias = f"{p['Dias_Antes_Cierre']:.1f}".replace(".", ",")
    texto = f"{base}. Se inscriben en promedio {dias} días antes del cierre."
    if reales / n < 0.10:
        texto += " Casi todos sus registros son sintéticos: el perfil no se confirma con datos reales."
    return texto


def segmentar(df, k):
    escalador = StandardScaler().fit(df[VARIABLES])
    X = escalador.transform(df[VARIABLES])
    k_max = min(8, len(df) - 1)
    if k > k_max:
        sys.exit(f"Con {len(df)} registros, k debe ser como máximo {k_max}.")
    modelo = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(X)
    df = df.assign(_c=modelo.labels_)

    # Orden estable: de la inscripción más tardía a la más anticipada (promedio de días antes del cierre)
    orden = df.groupby("_c")["Dias_Antes_Cierre"].mean().sort_values().index
    codigo = {c: f"S{i + 1}" for i, c in enumerate(orden)}
    df["Segmento"] = df["_c"].map(codigo)

    segmentos = []
    for c in orden:
        g = df[df["_c"] == c]
        p = g[VARIABLES].mean()
        reales = int((g["Origen"] == "Real").sum())
        cruce = g.groupby(["Origen", "Nivel"]).size()
        segmentos.append({
            "codigo": codigo[c], "nombre": nombrar(p), "descripcion": describir(p, len(g), reales),
            "participantes": int(len(g)), "reales": reales,
            "confirmado_con_reales": bool(reales / len(g) >= 0.10),
            "perfil": {
                "tam_delegacion": round(float(p["Tam_IE"]), 1),
                "con_asesor_pct": round(float(100 * p["Con_Asesor"])),
                "dias_antes_cierre_media": round(float(p["Dias_Antes_Cierre"]), 1),
                "dias_antes_cierre_mediana": float(g["Dias_Antes_Cierre"].median()),
                "grado_medio": round(float(p["Grado_Num"]), 1),
                "libres_pct": round(float(100 * p["Es_Libre"])),
            },
            "centroide": dict(zip(VARIABLES, [round(float(v), 3) for v in
                                              escalador.inverse_transform(modelo.cluster_centers_[[c]])[0]])),
            "por_origen_nivel": [{"origen": o, "nivel": n, "participantes": int(v)} for (o, n), v in cruce.items()],
            "instituciones": int(g.loc[g["Modalidad"] != "Libre", "IE"].nunique()),
        })

    validacion = {"silueta": round(float(silhouette_score(X, modelo.labels_)), 3),
                  "davies_bouldin": round(float(davies_bouldin_score(X, modelo.labels_)), 3),
                  "inercia": round(float(modelo.inertia_), 1)}
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    validacion["ward"] = {"silueta": round(float(silhouette_score(X, ward.labels_)), 3),
                          "ari_vs_kmeans": round(float(adjusted_rand_score(modelo.labels_, ward.labels_)), 3)}
    reales = df["Origen"] == "Real"
    if reales.any() and not reales.all() and reales.sum() > k:
        Xr = StandardScaler().fit_transform(df.loc[reales, VARIABLES])
        kr = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(Xr)
        validacion["solo_reales"] = {"silueta": round(float(silhouette_score(Xr, kr.labels_)), 3),
                                     "davies_bouldin": round(float(davies_bouldin_score(Xr, kr.labels_)), 3),
                                     "ari_vs_modelo_completo": round(float(
                                         adjusted_rand_score(df.loc[reales, "_c"], kr.labels_)), 3)}
    return df, segmentos, metricas_k(X, k_max), validacion


# ---------------------------------------------------------------- 10-11. Exportación
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fuente = ap.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--excel", help="Excel con la hoja Dataset_ML")
    fuente.add_argument("--db", help="Cadena de conexión de PostgreSQL")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cierre", default="2026-10-31", help="Fecha de cierre de inscripciones (con --db)")
    ap.add_argument("--solo-reales", action="store_true", help="Descarta los registros sintéticos")
    ap.add_argument("--salida", default="segmentos.json")
    a = ap.parse_args(argv)

    df = leer_excel(a.excel) if a.excel else leer_db(a.db, pd.Timestamp(a.cierre))
    if a.solo_reales:
        df = df[df["Origen"] == "Real"]
    if len(df) < 3:
        sys.exit("Se necesitan al menos 3 participantes para segmentar.")
    df, calidad = preparar(df)
    df, segmentos, metricas, validacion = segmentar(df, a.k)

    salida = {
        "version": 1,
        "ejecucion": {"fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "fuente": "excel" if a.excel else "postgresql", "k": a.k,
                      "variables": VARIABLES, "n_registros": int(len(df)),
                      "incluye_sinteticos": bool((df["Origen"] != "Real").any()), **validacion},
        "calidad_datos": calidad,
        "resumen": resumen(df),
        "metricas": metricas,
        "segmentos": segmentos,
        "asignaciones": [{"registro": r, "segmento": s, "origen": o}
                         for r, s, o in zip(df["ID"], df["Segmento"], df["Origen"])],
    }
    with open(a.salida, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1)

    print(f"{len(df)} registros · k = {a.k} · silueta = {validacion['silueta']} · DB = {validacion['davies_bouldin']}")
    for s in segmentos:
        aviso = "" if s["confirmado_con_reales"] else "  (no confirmado con datos reales)"
        print(f"  {s['codigo']} {s['nombre']}: {s['participantes']} participantes, {s['reales']} reales{aviso}")
    print(f"Archivo generado: {a.salida}")


if __name__ == "__main__":
    main()

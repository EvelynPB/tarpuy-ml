"""Pruebas del script de entrenamiento (casos CP-02, CP-04, CP-05, CP-06 y CP-07 del informe).

Ejecutar desde la carpeta ml/:  python -m unittest -v
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import entrenar


def dataset_ficticio(n=120, semilla=0):
    r = np.random.default_rng(semilla)
    libre = r.random(n) < 0.2
    grados = r.choice(entrenar.GRADOS, n)
    return pd.DataFrame({
        "ID": [f"T-{i}" for i in range(n)], "DNI": r.integers(10_000_000, 99_999_999, n),
        "Modalidad": np.where(libre, "Libre", "Institucional"), "IE": np.where(libre, "Independiente", "IE Demo"),
        "Nivel": [g.split()[-1] if "Inicial" not in g else "Inicial" for g in grados], "Grado": grados,
        "Grado_Num": [entrenar.GRADO_NUM[g] for g in grados],
        "Tam_IE": np.where(libre, 1, r.integers(5, 90, n)), "Con_Asesor": np.where(libre, 0, r.integers(0, 2, n)),
        "Dias_Antes_Cierre": r.integers(0, 20, n), "Medio_Pago": "Yape",
        "Fecha": pd.Timestamp("2026-10-20"), "Origen": np.where(r.random(n) < 0.5, "Real", "Sintético")})


class PruebasEntrenamiento(unittest.TestCase):
    def test_cp02_anonimizacion(self):
        df, _ = entrenar.preparar(dataset_ficticio())
        self.assertNotIn("DNI", df.columns)

    def test_cp04_coherencia_libres(self):
        df, _ = entrenar.preparar(dataset_ficticio())
        libres = df[df["Es_Libre"] == 1]
        self.assertTrue((libres["Tam_IE"] == 1).all())
        self.assertTrue((libres["Con_Asesor"] == 0).all())

    def test_cp05_estandarizacion(self):
        df, _ = entrenar.preparar(dataset_ficticio())
        X = StandardScaler().fit_transform(df[entrenar.VARIABLES])
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-9)

    def test_cp06_reproducibilidad(self):
        df, _ = entrenar.preparar(dataset_ficticio())
        a, *_ = entrenar.segmentar(df, 4)
        b, *_ = entrenar.segmentar(df, 4)
        self.assertListEqual(a["Segmento"].tolist(), b["Segmento"].tolist())

    def test_cp07_rango_metricas(self):
        df, _ = entrenar.preparar(dataset_ficticio())
        _, segmentos, metricas, validacion = entrenar.segmentar(df, 4)
        self.assertEqual(len(segmentos), 4)
        self.assertEqual(sum(s["participantes"] for s in segmentos), len(df))
        for m in metricas:
            self.assertTrue(-1 <= m["silueta"] <= 1)
            self.assertGreaterEqual(m["davies_bouldin"], 0)
        self.assertIn("solo_reales", validacion)


if __name__ == "__main__":
    unittest.main()

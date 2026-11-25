# Tarpuy Analytics ML

Proyecto del **Taller de Desarrollo de Aplicaciones con Machine Learning** (SENATI, 2026): segmentación de los participantes del XI Concurso Nacional de Matemática Lima Este 2026 del Colegio Tarpuy con **K-Means**, a partir de los registros de su sistema web de inscripciones.

El archivo `segmentos.json` que genera este proyecto se carga en el panel de analítica de la plataforma web del concurso, desarrollada por el Grupo 2 en el curso Fullstack Developer Software.

## Archivos

| Archivo | Para qué sirve |
| --- | --- |
| `entrenar.py` | Entrena el modelo y genera `segmentos.json`, el archivo que se carga en el panel. |
| `segmentacion_kmeans.ipynb` | El mismo análisis paso a paso, para Google Colab. |
| `test_entrenar.py` | Pruebas automáticas (casos CP-02, CP-04 a CP-07 del informe). |
| `requirements.txt` | Librerías de Python necesarias. |

> **Privacidad (Ley N.° 29733):** el Excel de inscripciones contiene DNI y nombres de menores. **Nunca lo subas al repositorio**; el `.gitignore` ya excluye los archivos `.xlsx`, `.csv` y `segmentos.json`.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Entrenar el modelo

Con el Excel exportado (hoja `Dataset_ML`, 416 registros reales y 1084 sintéticos):

```bash
python entrenar.py --excel dataset_tarpuy_inscritos_2026.xlsx
```

Directamente con las inscripciones reales de PostgreSQL (solo registros reales):

```bash
python entrenar.py --db "postgresql://usuario:clave@host:5432/base"
```

Opciones:

- `--k 4`: número de clústeres; el valor por defecto es el del informe.
- `--solo-reales`: descarta los registros sintéticos.
- `--cierre 2026-10-31`: fecha de cierre de las inscripciones; solo se usa con `--db`.
- `--salida segmentos.json`: nombre del archivo que se genera.

Con el Excel del concurso 2026 el resultado esperado es: silueta 0,358, Davies-Bouldin 1,144 y cuatro segmentos (S1 a S4). S4 aparece marcado como *no confirmado con datos reales*.

## Qué hace el script

1. **Extracción:** Excel o PostgreSQL.
2. **Anonimización:** elimina el DNI, los nombres y los teléfonos.
3. **Validación:** revisa nulos, duplicados, pagos en efectivo y grados fuera del alcance.
4. **Variables:** `Grado_Num`, `Tam_IE`, `Con_Asesor`, `Dias_Antes_Cierre` y `Es_Libre`, estandarizadas.
5. **Métricas para k de 2 a 8:** inercia, silueta y Davies-Bouldin.
6. **Modelo final:** K-Means con `n_init=20` y `random_state=42`.
7. **Validación del modelo:** compara con el clustering jerárquico (Ward) y vuelve a entrenar solo con los registros reales.
8. **Exportación:** genera `segmentos.json` con perfiles y conteos agregados, y el segmento asignado a cada ID de registro.

Los segmentos se numeran de la inscripción más tardía a la más anticipada, según el promedio de días antes del cierre.

## Cargar los resultados en la plataforma

1. Inicia sesión como organizador en el Sistema de Inscripciones.
2. En **Tarpuy Analytics ML**, pulsa **Cargar segmentos.json** y elige el archivo.

También se puede cargar con la API (requiere la cookie de sesión):

```bash
curl -b cookies.txt -H "Content-Type: application/json" --data-binary @segmentos.json \
  https://concurso-tarpuy-2026.fly.dev/api/admin/analytics/importar
```

## Pruebas

```bash
python -m unittest -v
```

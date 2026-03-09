# 🍺 Predicción de Ventas en una Cervecería — Cervecería Grut '84

<p align="center">
  <img src="brewery_cover.png" alt="Cervecería Grut '84 — Predicción de Ventas" width="350"/>
</p>

> Proyecto de Machine Learning para predicción de ventas por categoría de producto, utilizando datos históricos 2022–2024 de una cervecería argentina.

**Autoras:** Jeasmine Ñahui · Leticia Colombo  
**Programa:** Postgrado en Ciencia de Datos e Inteligencia Artificial — UTEC, 2025

---

## 📋 Tabla de Contenidos

- [Contexto del Negocio](#contexto-del-negocio)
- [Objetivo](#objetivo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Datos](#datos)
- [Análisis Exploratorio (EDA)](#análisis-exploratorio-eda)
- [Modelos](#modelos)
- [Resultados](#resultados)
- [Conclusiones y Mejoras Propuestas](#conclusiones-y-mejoras-propuestas)
- [Recomendaciones para el Negocio](#recomendaciones-para-el-negocio)
- [Requisitos](#requisitos)

---

## 🏪 Contexto del Negocio

**Cervecería Grut '84** es una cervecería argentina que comenzó como un emprendimiento y logró expandir su local al doble de su tamaño original. Hoy dispone de los datos históricos de su primer local (2022–2024) y busca entender qué proyecciones de ventas habría tenido, con el fin de evaluar la viabilidad de mantener o no ese local y optimizar la operación del nuevo.

---

## 🎯 Objetivo

Predecir las ventas diarias por categoría de producto para:

- **Optimizar el stock** y reducir pérdidas por sobreabastecimiento o faltantes.
- **Mejorar la operación** del local a través de una planificación más eficiente.
- **Evaluar tendencias y estacionalidades** en el comportamiento de los clientes.

---

## 📁 Estructura del Proyecto

```
├── Ventas_-_Análisis_Exploratorio.ipynb   # EDA completo del dataset
├── Ventas_-_XGBoost___CatBoost.ipynb      # Modelos de gradient boosting
├── Ventas_-_LSTM.ipynb                    # Red neuronal recurrente LSTM
└── Predicción_de_ventas_en_una_cervecería.pptx  # Presentación del proyecto
```

---

## 📊 Datos

### Fuentes

Se integraron cuatro datasets:

| Dataset | Filas | Columnas | Descripción |
|---|---|---|---|
| Ventas | 47.101 | 18 | Registro de consumo por mesa/pedido. Período 03/01/2022 – 30/12/2024 |
| Detalle de ventas | 174.298 | 15 | Desglose de productos por venta |
| Descuentos | 2.464 | 4 | Descuentos aplicados (descartado por inconsistencias) |
| Cotización BCRA | 1.097 | 2 | Tipo de cambio USD oficial diario (apis.datos.gob.ar) |

### Dataset Final

Tras el proceso de limpieza e integración:

- **Filas:** 130.558 registros de productos vendidos
- **Columnas:** 30 variables
- **Ventas únicas:** 46.578
- **Período:** 03/01/2022 – 30/12/2024
- **Operación:** Lunes a Sábado, 19:00 a 02:00 hs

### Variables Clave

**Temporales:** `fecha`, `mes`, `dia_semana`, `hora_creacion`, `hora_cerrada`, `fin_de_semana`, `tiempo_en_mesa`

**Monetarias (dolarizadas):** `precio_en_dolares`, `precio_unitario_en_dolares`, `costo_total_en_dolares`, `total_cobrado_en_dolares`, `valor_descuento_dolares`

> ⚠️ Los montos fueron dolarizados utilizando la cotización oficial del BCRA para eliminar el ruido generado por la inflación en Argentina.

---

## 🔍 Análisis Exploratorio (EDA)

### Limpieza de Datos

- Eliminación del dataset de descuentos por inconsistencias (171 ventas con discrepancias). El descuento se recalculó a partir de los otros datasets.
- Imputación de `mesa` nula: valor **199** para consumo en barra (14% de ventas), valor **99** para delivery (31% de ventas).
- Imputación de `personas` nulas: 0 para delivery, mediana para consumo en barra.
- Forward-fill de la cotización del dólar para días no hábiles.
- Eliminación del 1,1% de ventas sin detalle de productos.

### Hallazgos Principales

**Distribución temporal:**
- Mayor actividad los **viernes y sábados**; domingo registra el menor nivel de ventas.
- Pico horario entre las **21:00 y 22:00 hs**.
- Meses más activos: **julio y diciembre**. Leve baja en abril–junio.

**Comportamiento de clientes:**
- Promedio de **2 personas por mesa**; grupos mayores asociados a eventos.
- Mediana de **88 minutos** en mesa.
- Consumo promedio por mesa: **~45 USD**.
- Solo el **~14% de ventas** tienen descuento (incluye 2x1 de pintas los miércoles/jueves).

**Canales de venta:**
- 55% mesas · 31% delivery · 14% barra

**Medios de pago:** Débito Taj (~31%), Crédito (~15%), Efectivo (~13%).

**Productos y categorías:**
- **24 categorías** y **497 productos** en total.
- **Tiradas** es la categoría dominante; **Pintas** es el producto más consumido.
- Productos de mayor margen identificados: Zapallo (~30 USD), Tabla (~29 USD).

**Correlaciones relevantes:**
- `personas` (0.70) y `tiempo_en_mesa` (0.63–0.64) correlacionan positivamente con el total cobrado.
- `precio_en_dolares` correlaciona fuerte con `cantidad`.
- Variables temporales (`hora`, `dia_del_mes`) no muestran correlación significativa con el monto.

**Evolución anual:**
- 2022 bajo (efecto post-COVID). Recuperación notable en 2023. Estabilización con estacionalidad clara en 2024.

---

## 🤖 Modelos

### Preparación del Dataset de Modelado

- Se agruparon las ventas por **categoría y día** (suma de cantidad de productos).
- Se seleccionaron las **9 categorías con mayor volumen** de ventas.
- Se utilizaron los **últimos 3 meses** como set de evaluación (test).
- Se mantuvieron los datos de 2022 para contar con mayor historial de entrenamiento.
- Métricas de evaluación: **MAE**, **RMSE** y **R²**

**Categorías seleccionadas (por volumen):**

| Categoría | Ventas Totales |
|---|---|
| Tiradas | 103.958 |
| Otras Bebidas | 21.138 |
| Pizzas | 18.617 |
| Hamburguesas | 17.761 |
| Bebidas sin alcohol | 16.617 |
| Papas | 11.844 |
| Gin tonic | 8.103 |
| Latas | 6.326 |
| Picoteo | 5.316 |

**Patrones identificados:**
- Meses fuertes: julio–agosto y noviembre–diciembre.
- Meses débiles: abril–mayo–junio.
- El viernes es el día de mayor venta en casi todas las categorías.

---

### Modelo 1: XGBoost & CatBoost

**Enfoque:** Un único modelo entrenado para todas las categorías.

**Variantes entrenadas:**
1. Baseline (sin ajuste)
2. Hiperparámetros optimizados (`RandomizedSearchCV` + `TimeSeriesSplit`)
3. Optimizado + lags y rolling features
4. Optimizado + lags, rolling + walk-forward validation

**Features de ingeniería:** Lags: 1, 2, 6 días · Rolling mean: 7, 14, 28 días · Rolling std: 7 días

**Mejor resultado por modelo:**

| Modelo | Mejor MAE | Mejor RMSE | Mejor R | Mejor R² |
|---|---|---|---|---|
| XGBoost | 6.96 (tuned) | 12.70 (tuned) | 0.929 (tuned) | 0.8449 (tuned) |
| CatBoost | 7.02 (baseline) | 12.43 (baseline) | 0.9248 (tuned) | 0.8513 (baseline) |

> **Modelo seleccionado:** CatBoost baseline. Los lags y rolling features no aportaron mejoras significativas, posiblemente por el alto ruido en las ventas diarias.

**Librerías:** `xgboost`, `catboost`, `scikit-learn`

---

### Modelo 2: LSTM (Long Short-Term Memory)

**Enfoque:** Un modelo LSTM independiente entrenado por cada categoría, dado el alto contraste de volumen entre ellas.

**Arquitectura:**
```
Input → LSTM → Dense (output)
```
- Lookback (sequence_length): **15 días**
- Normalización: `MinMaxScaler`
- Optimización de epochs y batch_size por categoría

**Librerías:** `tensorflow/keras`, `scikit-learn`, `numpy`, `pandas`

---

## 📈 Resultados

### Resultados CatBoost (modelo elegido)

| Categoría | MAE | RMSE | R | R² | MAE Rel | RMSE Rel |
|---|---|---|---|---|---|---|
| Tiradas | 21.76 | 31.02 | 0.788 | 0.603 | 0.219 | 0.214 |
| Otras Bebidas | 8.77 | 12.30 | 0.587 | 0.271 | 0.436 | 0.327 |
| Hamburguesas | 7.52 | 9.64 | 0.752 | 0.327 | 0.268 | 0.236 |
| Bebidas sin alcohol | 5.74 | 7.87 | 0.732 | 0.455 | 0.315 | 0.246 |
| Pizzas | 5.41 | 6.95 | 0.729 | 0.512 | 0.266 | 0.231 |
| Gin tonic | 4.86 | 6.91 | 0.209 | 0.209 | 0.461 | 0.302 |
| Papas | 4.24 | 5.37 | 0.721 | 0.052 | 0.379 | 0.311 |
| Latas | 3.73 | 4.86 | 0.499 | 0.217 | 0.631 | 0.326 |
| Picoteo | 2.35 | 3.08 | 0.641 | 0.344 | 0.365 | 0.285 |

### Resultados LSTM

| Categoría | MAE | RMSE | MAE Rel | RMSE Rel | R² |
|---|---|---|---|---|---|
| Tiradas | 23.68 | 31.39 | 0.24 | 0.22 | **0.57** |
| Pizzas | 6.06 | 7.95 | 0.30 | 0.26 | **0.47** |
| Hamburguesas | 6.76 | 8.92 | 0.24 | 0.22 | 0.44 |
| Bebidas sin alcohol | 5.82 | 7.72 | 0.32 | 0.24 | 0.43 |
| Papas | 3.31 | 4.62 | 0.30 | 0.27 | 0.41 |
| Otras Bebidas | 9.55 | 12.19 | 0.47 | 0.32 | 0.34 |
| Picoteo | 2.54 | 3.28 | 0.39 | 0.30 | 0.32 |
| Gin tonic | 4.88 | 6.40 | 0.46 | 0.28 | 0.31 |
| Latas | 3.92 | 4.97 | 0.66 | 0.33 | 0.22 |

---

## 🏁 Conclusiones y Mejoras Propuestas

### Conclusiones

- Los resultados de XGBoost, CatBoost y LSTM son **similares entre sí**, lo que demuestra que algoritmos más complejos no siempre garantizan mejores resultados.
- **Tiradas y Pizzas** logran los mejores desempeños en ambos enfoques.
- LSTM, al entrenarse por categoría, performa mejor en categorías de bajo volumen (ej. Papas) en comparación con CatBoost.
- **Todos los modelos tienden a subestimar las ventas**, lo que sugiere que existen factores no capturados por las variables actuales.
- La variabilidad natural de las ventas diarias (clima, eventos, turismo) limita el techo de performance.

### Mejoras Propuestas

- **Modelos específicos** para subconjuntos de categorías con bajo rendimiento (Latas, Gin Tonic, Otras Bebidas).
- Explorar **predicción semanal** en lugar de diaria para reducir el ruido.
- Investigar **lags y promedios móviles específicos** por categoría.
- Agregar **features externas**: promociones activas, condiciones climáticas, feriados, eventos.
- Utilizar **SHAP/LIME** para interpretar la contribución de cada variable.
- Explorar **Chronos** (modelo de series de tiempo basado en LLM) como alternativa.
- Aplicar `RandomizedSearchCV` con foco en hiperparámetros de regularización.

---

## 💡 Recomendaciones para el Negocio

- **Diferenciar en el sistema** los pedidos de delivery, barra y mesa. Los pedidos en barra actualmente pierden datos de personas y consumo.
- **Registrar eventos y campañas de marketing** para incorporarlos como features.
- **Llevar registro de stock** diario o semanal: stock inicial/final, entradas y salidas no asociadas a ventas.
- **Evaluar capacidad de almacenamiento** por tipo de producto para optimizar la logística.
- Para el **nuevo local**: implementar el sistema de registro desde el inicio con foco en la calidad del dato.

---

## ⚙️ Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost catboost
pip install tensorflow keras
pip install holidays openpyxl
```

**Entorno:** Google Colab (con acceso a Google Drive para la carga de datos)

---

*Proyecto desarrollado en el marco del Postgrado en Ciencia de Datos e Inteligencia Artificial — UTEC, 2025.*

# 🏦 ClaimStruct — Automatización Inteligente de Reclamos Bancarios

> Sistema de clasificación de intención y extracción de entidades (NER) para reclamos bancarios en texto libre, basado en un modelo Transformer multitarea sobre BERT en español.

**Autoras:** Leticia Colombo · Jeasmine Ñahui  
**Programa:** Especialización en Ciencia de Datos e Inteligencia Artificial — UTEC, 2025  
**Materia:** Deep Learning

---

## 📋 Tabla de Contenidos

- [Contexto y Problema](#contexto-y-problema)
- [Objetivo](#objetivo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Dataset](#dataset)
- [Análisis Exploratorio (EDA)](#análisis-exploratorio-eda)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Pipeline de Entrenamiento](#pipeline-de-entrenamiento)
- [Resultados](#resultados)
- [Conclusiones y Mejoras Propuestas](#conclusiones-y-mejoras-propuestas)
- [Requisitos](#requisitos)

---

## 🏦 Contexto y Problema

Las entidades financieras reciben un alto volumen de reclamos a través de canales digitales como chat, app móvil y web. Estos mensajes están redactados en **texto libre** y suelen contener múltiples intenciones y detalles implícitos.

El proceso de clasificación es actualmente **manual**, lo que implica:
- Costos operativos elevados
- Inconsistencias en la categorización
- Baja escalabilidad ante picos de demanda
- Impacto negativo en la velocidad de derivación a equipos especializados

---

## 🎯 Objetivo

Desarrollar un sistema basado en modelos Transformer capaz de:

1. **Clasificar la intención principal** del reclamo (Intent Detection)
2. **Extraer entidades relevantes** del texto (NER — Named Entity Recognition)

Generando así información estructurada que facilite el **enrutamiento interno**, el **análisis agregado** y la **trazabilidad** de los reclamos.

---

## 📁 Estructura del Proyecto

```
├── 1-_EDA_ClaimStruct.ipynb          # Análisis exploratorio del dataset
├── 2-_Model__ClaimStruct.ipynb       # Entrenamiento y evaluación del modelo
└── 3-_ClaimStruct_Presentation.pptx  # Presentación del proyecto
```

---

## 📊 Dataset

Se creó un **dataset sintético** de **8.000 conversaciones**, dado que los datos reales de reclamos bancarios no estaban disponibles por razones de confidencialidad.

### Estructura multitabla

| Tabla | Filas | Descripción |
|---|---|---|
| `conversations` | 8.000 | Metadatos de cada conversación (canal, producto, segmento) |
| `messages` | 46.713 | Mensajes individuales por conversación |
| `labels` | 8.000 | Etiquetas de intención y clasificación |
| `entities` | 16.502 | Entidades anotadas con spans y tipo |

### Características del dataset

- **Promedio de ~55 palabras** por conversación y mediana de **6 mensajes** por conversación
- **8 intenciones** (intent) principales
- **5 productos** bancarios
- **6 canales** de entrada
- Etiquetado adicional de entidad motivo

**Intenciones disponibles:**

| Intent | Descripción |
|---|---|
| `rechazo_prevencion_fraude` | Clase más frecuente (~2.400 casos) |
| `controversia_compra` | Segunda clase más frecuente (~1.800 casos) |
| `fraude_desconocimiento` | Casos de fraude no reconocido por el cliente |
| `acreditacion_transferencia` | Problemas con transferencias no acreditadas |
| `acreditacion_pago` | Pagos no reflejados en la cuenta |
| `cuenta_otros` | Reclamos varios de cuenta |
| `acceso_canales_digitales` | Problemas de acceso a banca digital |
| `token_2fa` | Problemas con autenticación de dos factores (clase menos frecuente) |

### Consideraciones del dataset sintético

- **Menor ruido** que datos reales; vocabulario más homogéneo por intención.
- **Data leakage** detectado en el split original por reutilización de frases (~3.271 textos únicos sobre 46.713 mensajes). Se aplicó re-split estratificado para mitigarlo.
- **Alta separabilidad** de clusters en TF-IDF + t-SNE, lo que puede llevar al modelo a memorizar patrones léxicos en lugar de semántica generalizable.
- Se aplicó **data augmentation** para diversificar el vocabulario y reducir la sobre-compactación de clusters.

---

## 🔍 Análisis Exploratorio (EDA)

### Calidad de datos

El dataset es limpio: no hay valores nulos en `conversations`, `messages` ni `entities`. Los nulos en `labels` son esperados (campo `intent_secondary` es opcional, presente solo en ~18,2% de los casos).

### Hallazgos principales

**Intenciones:**
- `rechazo_prevencion_fraude` domina con ~2.400 casos; `token_2fa` es la clase más escasa (~250).
- El intent secundario, disponible en solo el 18,2% de las conversaciones, no se utilizó para entrenamiento.

**Productos y canales:**
- Las **tarjetas de crédito y débito** concentran más de la mitad de los casos.
- La distribución entre canales y productos es relativamente balanceada.

**Longitud de mensajes:**
- Distribución **bimodal**: mensajes cortos (saludos, confirmaciones) y mensajes largos (descripción del reclamo).
- Mediana de **58 caracteres** por mensaje.
- Las conversaciones completas se ajustan dentro del límite de tokens de BERT.

**Entidades (NER):**
- La mayoría de las conversaciones tienen entre 1 y 4 entidades (pico en 3).
- Alta co-ocurrencia entre `AMOUNT` y `CURRENCY`, lo que es semánticamente coherente.
- Los intents de fraude y acreditación tienen mayor densidad de entidades por conversación.

**Correlaciones semánticas:**
- Intents de fraude (`fraude_desconocimiento`, `rechazo_prevencion_fraude`) se asocian con mayor severidad y más frecuente escalado.
- Intents simples (`token_2fa`, `acceso_canales_digitales`) se resuelven más frecuentemente en el primer contacto.

---

## 🤖 Arquitectura del Modelo

El modelo es un **Transformer multitarea** que comparte un encoder BERT preentrenado para resolver simultáneamente las dos tareas.

```
Input Text
    │
    ▼
Tokenizer (bert-base-spanish-wwm-cased)
    │
    ▼
┌─────────────────────────────────────┐
│     Pretrained BERT Encoder         │
│  (bert-base-spanish-wwm-cased)      │
│  [Congelado durante entrenamiento]  │
└────────────┬────────────────────────┘
             │ last_hidden_state
     ┌───────┴────────┐
     │                │
     ▼                ▼
Intent Classification  NER (Token-level)
  Attention Pooling    Dense Layer
  Dropout              (num_ner_labels)
  LayerNorm
  Dense Layer
  (num_intents)
     │                │
     ▼                ▼
 Intent Logits    Token-level Logits
```

**Función de pérdida combinada:**

$$L_{Total} = \lambda_1 \cdot L_{Intent} + \lambda_2 \cdot L_{NER}$$

Donde $\lambda_1$ y $\lambda_2$ son los pesos configurables de cada tarea (`intent_loss_weight`, `ner_loss_weight`).

### Componentes clave

**`MultiTaskBert`:** Modelo principal. Inicializa el encoder BERT compartido, aplica attention pooling para intent classification y proyección token-level para NER. El encoder se mantiene **congelado** durante el entrenamiento dado el origen sintético del dataset, evitando sobreajuste.

**`MaskedSparseCategoricalCrossentropy`:** Loss para NER con enmascaramiento de tokens de padding y especiales (`[CLS]`, `[SEP]`). Incluye clipping de pérdida entre 0 y 10 para evitar gradientes explosivos.

**`WeightedSparseCCE`:** Loss ponderada por clase para intent, útil ante el desbalance entre intenciones.

### Configuración de entrenamiento

- **Optimizador:** Adam con `clipnorm=1.0`
- **Learning rate scheduler:** CosineDecay adaptado a longitud del dataset
- **Early Stopping:** sobre pérdida de validación con restauración de mejores pesos
- **ReduceLROnPlateau:** reducción de LR ante estancamiento de validación
- **Etiquetado BIO** para NER: `O`, `B-ENTITY`, `I-ENTITY`

---

## ⚙️ Pipeline de Entrenamiento

1. **Consolidación por conversación:** Los mensajes se concatenan cronológicamente, se normalizan y los spans de entidades se reajustan a coordenadas globales.
2. **Split estratificado** por intención (train / val / test), regenerado para mitigar data leakage del split original.
3. **Construcción de vocabularios:** `intent2id` / `id2intent` y vocabulario BIO (`label2id` / `id2label`).
4. **Tokenización y alineamiento BIO:** con `bert-base-spanish-wwm-cased` (fast tokenizer), usando `offset_mapping` para proyectar spans de carácter a nivel token.
5. **Construcción de `tf.data.Dataset`** para cada split con batching y prefetch.
6. **Entrenamiento multitarea** con `model.fit()` evaluando en validación.
7. **Evaluación** con reportes de clasificación y matrices de confusión para intent y NER.

---

## 📈 Resultados

### Intent Detection

| Métrica | Valor |
|---|---|
| **Accuracy** | **98%** |
| **Weighted F1** | **0.99** |
| **Macro F1** | **0.97** |

**Mejores clases (alta frecuencia y separabilidad semántica):**

| Intent | F1 |
|---|---|
| `controversia_compra` | 1.00 |
| `fraude_desconocimiento` | 1.00 |
| `acreditacion_pago` | 1.00 |
| `acreditacion_transferencia` | 1.00 |

**Clases con menor desempeño (bajo soporte o solapamiento semántico):**

| Intent | Métrica | Valor |
|---|---|---|
| `token_2fa` | Recall | 0.81 |
| `acceso_canales_digitales` | Recall | 0.92 |

---

### NER (Named Entity Recognition)

| Métrica | Valor |
|---|---|
| **Accuracy** | **99%** |
| **Weighted F1** | **0.99** |
| **Macro F1** | **0.36** |

> ⚠️ El F1 macro bajo refleja el fuerte desbalance frente a la clase dominante `"O"` (no-entidad), que es el comportamiento esperado en NER con este tipo de distribuciones.

**Mejores entidades (mayor frecuencia y estructura consistente):**

| Entidad | F1 |
|---|---|
| `I-AMOUNT` | 0.77 |
| `B-CURRENCY` | 0.77 |
| `I-CURRENCY` | 0.71 |
| `I-MERCHANT` | 0.61 |

**Entidades con bajo desempeño (bajo soporte):**

| Entidad | F1 |
|---|---|
| `B-CHANNEL` | ~0 |
| `B-COUNTRY` | ~0 |
| `B-PRODUCT` | ~0 |

El modelo tiende a predecir `"O"` ante incertidumbre, generando falsos negativos en entidades poco frecuentes.

---

## 🏁 Conclusiones y Mejoras Propuestas

### Conclusiones

- El modelo multitarea logra un **desempeño global sólido** en ambas tareas, especialmente en Intent Detection.
- El uso de **BERT congelado** resulta suficiente para el dataset sintético, evitando sobreajuste sin perder capacidad discriminativa.
- La **alta separabilidad léxica** del dataset sintético favorece métricas altas pero advierte sobre posible fragilidad ante datos reales con mayor variabilidad lingüística.
- El bajo **F1 macro en NER** es consecuencia directa del desbalance de clases y no indica falla estructural del modelo.

### Mejoras propuestas

**Para Intent Detection:**
- Data augmentation sintáctico para clases minoritarias (`token_2fa`, `acceso_canales_digitales`).
- Uso de **Focal Loss** en lugar de CrossEntropy para penalizar más los errores en clases difíciles.
- Análisis de embeddings para detectar solapamiento semántico entre intents.
- Aumentar el peso de clases minoritarias.

**Para NER:**
- Aplicar **class weights** también a nivel token en NER.
- **Focal Loss token-level** para reducir el dominio de la clase `"O"`.
- Agregar una **capa CRF** sobre la salida NER para garantizar secuencias BIO válidas.
- Post-procesamiento para corregir secuencias BIO inválidas (`I-` sin `B-` previo).
- Evaluar métricas **entity-level** (span completo) en lugar de solo token-level.
- Balancear entidades raras mediante oversampling o augmentation específico.

**Validación en producción:**
- Validar el modelo con datos reales de reclamos bancarios para confirmar la generalización fuera del dominio sintético.

---

## ⚙️ Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install tensorflow
pip install transformers  # HuggingFace (BERT tokenizer y encoder)
```

**Modelo base:** [`dccuchile/bert-base-spanish-wwm-cased`](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) (BETO)

**Entorno:** Google Colab (con acceso a Google Drive para carga del dataset `claimstruct_multitable_v2.zip`)

---

*Proyecto desarrollado en el marco de la Especialización en Ciencia de Datos e Inteligencia Artificial — UTEC, 2025.*

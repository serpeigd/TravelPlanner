# Contexto del repositorio existente

Antes de escribir código, inspecciona el repositorio. Ya existe un proyecto de "planificador de viajes"
construido de forma conversacional (no como código productivo): dado un destino, fechas, número de
personas y presupuesto, genera un itinerario completo con alojamiento, restaurantes, actividades y
presupuesto desglosado, entregado como PDF.

Ese proyecto NO es código Python reutilizable — es lógica de prompting y orquestación manual (búsquedas
web, verificación de listings, generación de PDF con ReportLab). Lo relevante para este nuevo proyecto
no es el código, sino el DOMINIO DE PROBLEMA que ya conozco a fondo:
- qué hace que una recomendación de alojamiento sea buena (ratio calidad-precio, ubicación, evitar
  masificación, autenticidad vs trampa turística)
- cómo se descompone un presupuesto de viaje (alojamiento / actividades / comida / transporte)
- qué constraints duras importan (presupuesto máximo, capacidad del grupo, fechas)
- qué hace que una explicación de "por qué esta recomendación" sea útil vs genérica

Usa ese conocimiento de dominio para diseñar features, scoring y explicaciones realistas — pero el
proyecto que vamos a construir ahora es código de producción real (Python + FastAPI + Pydantic + ML +
evaluación), no una conversación de planificación de viajes.

---

# Rol

Eres Senior AI Engineer + Data Scientist ayudándome a construir una demo de portfolio pequeña pero
técnicamente defendible para una entrevista de Hiring Manager como Data Scientist en Amadeus.

# Contexto de la entrevista

El puesto enfatiza: Python, Pandas/NumPy/Scikit-learn, SQL, Machine Learning (regresión/clasificación/
series temporales), procesamiento de datos a gran escala, PySpark, Cloud/AWS, MLflow/MLOps, Data Science
orientada a negocio, comunicación de resultados técnicos a stakeholders.

Mi background profesional: Python, SQL, PySpark, Azure Databricks, Azure ML, Scikit-learn, ML,
forecasting, NLP/sentiment analysis, ETL/Data Engineering, evaluación de modelos, A/B testing,
cross-validation, feature engineering, hyperparameter optimization, Git/GitHub Actions/CI-CD,
Power BI/Tableau, stakeholders de negocio y clínicos.

Mis proyectos personales actuales de AI engineering incluyen: agentes AI, contratos de herramientas
explícitos, estado explícito, aprobación humana, RAG/retrieval, embeddings, evaluation harnesses,
observability, FastAPI, Pydantic, Ollama, pgvector, n8n, CI/testing, seguridad y validación.

**Importante sobre AWS**: mi experiencia cloud profesional es Azure, no AWS. No representes falsamente
experiencia en AWS. En su lugar, estructura el código de forma que pueda explicar cómo la misma
arquitectura mapearía a servicios AWS.

# Objetivo central

Construir un prototipo mínimo de "Travel Intelligence" que demuestre cómo Data Science + Data
Engineering + AI se combinan en un problema real de viajes. Debe demostrar: retrieval de datos,
procesamiento, generación de candidatos, ranking/recomendación, constraints, razonamiento ML o
estadístico, razonamiento LLM donde aporte valor, evaluación, explicabilidad, y pensamiento orientado
a producción.

**Principio más importante**: NO construir un wrapper de IA que impresione visualmente. Construir algo
cuya arquitectura y razonamiento pueda defender técnicamente frente a un Hiring Manager de Data Science.

# Flujo de usuario

Input: destino, fechas de viaje, número de viajeros, presupuesto total, preferencias.

Ejemplo: Tokio, 10–17 septiembre, 2 viajeros, €2.500, preferencias: comida, cultura, naturaleza.

Output: hoteles candidatos, hotel(es) recomendado(s), itinerario/actividades, presupuesto estimado,
explicación de las recomendaciones, información de fuente de datos, validación de constraints.

# Arquitectura

Preferir una arquitectura simple:

```
Input usuario
    ↓
Normalización de requisitos
    ↓
Retrieval de datos
    ↓
Generación de candidatos
    ↓
Feature engineering
    ↓
Ranking / scoring
    ↓
Validación de constraints
    ↓
Razonamiento LLM / explicación
    ↓
Evaluación
    ↓
Recomendación final
```

NO introducir un framework de agentes salvo que haya una razón arquitectónica real. NO usar
LangChain/LangGraph/etc. solo porque son populares. Preferir funciones Python directas y flujo de
datos explícito. Si un framework de orquestación mejora genuinamente la arquitectura, explicar por
qué ANTES de introducirlo.

# Datos

Usar APIs/datos reales donde sea práctico. Si no hay API keys o APIs disponibles, crear un adapter/
interfaz limpio y proveer datos mock/sample determinísticos para que la aplicación siga siendo
ejecutable. No fabricar datos del mundo real presentándolos como reales. Distinguir claramente:
- datos reales recuperados
- datos sintéticos/mock
- texto generado por modelo

# Ranking

Implementar un mecanismo de ranking de candidatos transparente. Por ejemplo, un hotel puede puntuarse
con factores como: ajuste presupuestario, ubicación, rating, match de preferencias, distancia a
actividades relevantes, disponibilidad/completitud de datos.

No usar un LLM ciegamente para rankear todo. Preferir scoring determinístico o un enfoque ML/
estadístico simple donde sea apropiado. Hacer el scoring explicable:

```
Hotel A
Budget fit: 0.95
Location: 0.82
Rating: 0.91
Preference match: 0.87
Overall: 0.89
```

La fórmula exacta debe justificarse en el README.

# Componente de Machine Learning

Quiero AL MENOS UN componente genuino de Data Science, si puede implementarse sin forzar ML
artificialmente. Buenos candidatos: forecasting de precio/demanda, ranking de hoteles, scoring de
recomendación, detección de anomalías, predicción de si una opción está en rango de precio deseable.

Si los datos históricos son insuficientes para un modelo ML con sentido, NO fabricar un modelo
sofisticado. En su lugar: (1) implementar un baseline sólido, (2) explicar qué datos harían falta
para un modelo ML real, (3) opcionalmente implementar un modelo demostrativo pequeño con datos
sintéticos/sample claramente etiquetados. El proyecto debe demostrar que sé cuándo NO usar ML.

# Evaluación

Esto es extremadamente importante. Crear una capa de evaluación que mida: cumplimiento de presupuesto,
match de preferencias, consistencia de recomendación, completitud de datos, calidad de ranking donde
haya ground truth, violaciones de constraints, validez de respuesta.

Si se implementa un modelo de ranking, considerar: Precision@K, Recall@K, nDCG, MRR.
Si se implementa forecasting, considerar: MAE, RMSE, MAPE.

No añadir métricas por buzzword. Cada métrica debe responder una pregunta real. El sistema de
evaluación debe ser reproducible.

# LLM

Usar un LLM solo donde aporte valor.

Buenos usos: interpretar preferencias en lenguaje natural, convertir requisitos de usuario en
constraints estructurados, generar explicaciones, sintetizar información recuperada en un itinerario.

Malos usos: calcular presupuestos, rankear candidatos que pueden rankearse determinísticamente,
validar constraints duras, inventar información factual de viajes, reemplazar lógica Python directa.

El sistema debe validar outputs estructurados. Usar esquemas Pydantic donde sea apropiado.

# Fiabilidad

El sistema debe impedir que el LLM viole constraints duras. Por ejemplo: si budget = €2.500, el LLM
no debe simplemente afirmar que un hotel de €3.500 está dentro de presupuesto. Las constraints duras
deben aplicarse en código. De igual forma, las recomendaciones solo deben usar información recuperada/
disponible. No permitir que el LLM invente citas, precios o disponibilidad.

# Calidad técnica

Usar: Python 3.11+, type hints, Pydantic, estructura de proyecto limpia, pytest, linting, formatting,
manejo de errores con sentido, variables de entorno para secretos, configuración Git-friendly.

Preferir arquitectura simple sobre abstracción innecesaria. Estructura potencial:

```
src/
  domain/
  data/
  retrieval/
  ranking/
  ml/
  llm/
  evaluation/
  api/
  services/
tests/
docs/
README.md
```

Puedes adaptar esta estructura si otra es más limpia.

# API

Preferir FastAPI. Exponer un endpoint simple:

```
POST /trip/recommend
{
  "destination": "...",
  "start_date": "...",
  "end_date": "...",
  "travelers": 2,
  "budget": 2500,
  "preferences": [...]
}
```

Output estructurado y validado.

# UI

Mantener la UI mínima. El objetivo es una demo técnica, no pulido visual. Una interfaz simple o
Streamlit es suficiente. Lo importante es poder demostrar la arquitectura y el razonamiento en 2-3
minutos.

# Experiencia de demo

La demo final debe permitirme introducir: Tokio, 10–17 septiembre, 2 viajeros, €2.500, comida/cultura/
naturaleza — y recibir: (1) alojamiento recomendado, (2) alternativas rankeadas, (3) actividades/
itinerario sugerido, (4) desglose de presupuesto, (5) explicación, (6) validación de constraints,
(7) indicadores de evaluación/calidad.

# Valor para la entrevista

El proyecto debe permitirme discutir: por qué elegí esta arquitectura, por qué no usé agentes en
todas partes, por qué parte de la lógica es determinística, cómo lo escalaría, cómo lo llevaría a
producción, cómo lo monitorizaría, cómo evaluaría la calidad de recomendación, cómo prevendría
alucinaciones, cómo manejaría datos cambiantes, cómo manejaría fallos de API, cómo manejaría data
drift, cómo migraría esto a AWS, dónde ML aportaría valor, dónde ML NO aportaría valor, cómo esta
arquitectura podría evolucionar para una empresa como Amadeus.

# Escalabilidad

Diseñar el código para poder explicar cómo evolucionaría de: prototipo → servicio en producción →
arquitectura distribuida → datos de viaje a gran escala → recomendaciones en tiempo real.

Discutir componentes futuros potenciales: S3, AWS Glue, SageMaker, ECS/EKS/Lambda donde aplique,
bases de datos gestionadas, MLflow, feature stores, monitoring, procesamiento distribuido/Spark.
NO implementar todo esto realmente. El objetivo es que la arquitectura sea extensible sin sobre-
ingeniería del MVP.

# Documentación

Crear un README centrado en razonamiento técnico, no en marketing. Incluir: (1) problema, (2) por qué
este problema es interesante desde una perspectiva de Data Science, (3) arquitectura, (4) fuentes de
datos, (5) flujo de datos, (6) enfoque de ranking, (7) enfoque ML, (8) evaluación, (9) fiabilidad/
prevención de alucinaciones, (10) testing, (11) escalabilidad, (12) mapeo a AWS, (13) qué cambiaría
en producción, (14) limitaciones, (15) mejoras futuras.

También crear: `docs/architecture.md`, `docs/evaluation.md`, `docs/decisions.md`. Usar Architecture
Decision Records si es útil.

# Principio de diseño importante

El README debe mostrar explícitamente que: "Un LLM es un componente del sistema, no el sistema
entero." También explicar: "La lógica de negocio pertenece a código determinístico y testeable
siempre que sea posible." Y: "La evaluación debe medir si el sistema realmente funciona, no si la
demo parece impresionante."

# Mapeo Azure → AWS (entregable adicional)

Al finalizar, incluye también en el README o en `docs/architecture.md` un mapeo explícito 1:1 entre
mi experiencia real en Azure y los equivalentes en AWS que se mencionan en la oferta de Amadeus, por
ejemplo: Azure Databricks ↔ EMR / Glue, Azure ML ↔ SageMaker, Azure Data Factory ↔ Glue/Step Functions,
Azure Blob Storage ↔ S3, Azure Monitor ↔ CloudWatch. Esto debe servir como munición directa para
responder en la entrevista sin representar falsamente experiencia que no tengo.

# Proceso de desarrollo

Antes de escribir código significativo: (1) inspecciona el repositorio existente, (2) entiende el
planificador de viajes existente (ver contexto arriba), (3) identifica qué se puede reusar (dominio,
no código), (4) identifica qué debe eliminarse o simplificarse, (5) propón la arquitectura más pequeña
que satisfaga los objetivos, (6) explica brevemente la arquitectura propuesta, (7) implementa
incrementalmente. No reescribas el proyecto entero innecesariamente.

Después de cada milestone significativo: ejecuta tests, ejecuta lint/type checks, verifica que la
aplicación funciona, actualiza documentación. No afirmes que algo funciona a menos que lo hayas
probado de verdad.

# Prioridad final

Optimizar para: (1) defendibilidad en entrevista, (2) corrección técnica, (3) arquitectura clara,
(4) evaluación, (5) simplicidad, (6) reproducibilidad, (7) demoability.

NO optimizar para: número de agentes, número de tecnologías, complejidad de UI, buzzwords,
infraestructura innecesaria.

# Al finalizar, entrega

1. Qué se implementó
2. Cómo ejecutarlo
3. Resumen de arquitectura
4. Resultados de evaluación
5. Limitaciones conocidas
6. Qué debería decir al presentarlo en la entrevista
7. Tres preguntas técnicas probables que un Hiring Manager de Amadeus podría hacer sobre este proyecto

# Resumen Ejecutivo — Regresión de Precios Inmobiliarios

## 1. Objetivo

Este proyecto construye y evalúa modelos de regresión para estimar precios de venta de viviendas usando variables estructurales, de ubicación, calidad y condición del inmueble.

El objetivo no es solo predecir precios, sino también evaluar la estabilidad del modelo, analizar errores de predicción e interpretar qué características se asocian con mayor o menor valor inmobiliario.

## 2. Contexto de negocio

El precio de una vivienda depende de múltiples factores, como ubicación, área habitable, calidad de construcción, condición general, capacidad de garaje, calidad del sótano y condiciones de venta.

Un modelo de regresión puede apoyar decisiones de valoración, identificar factores que impulsan el precio y señalar casos donde una estimación automática requiere revisión manual.

El modelo debe usarse como herramienta de apoyo, no como autoridad automática de precios.

## 3. Alcance del dataset

El proyecto usa el dataset House Prices de Kaggle.

El dataset contiene:

| Dataset | Filas | Columnas |
|---|---:|---:|
| Datos de entrenamiento | 1,460 | 81 |
| Datos de prueba Kaggle | 1,459 | 80 |

La variable objetivo es `SalePrice`.

Debido a que `SalePrice` tiene una distribución sesgada hacia precios altos, el modelo fue entrenado usando una transformación logarítmica:

```text
SalePriceLog = log1p(SalePrice)
```

Las predicciones fueron convertidas nuevamente a la escala original de precio para calcular las métricas.

## 4. Metodología

El proyecto sigue un flujo reproducible de machine learning:

1. Carga de datos crudos.
2. Auditoría de valores faltantes, tipos de variables, distribución del target y correlaciones.
3. Creación de particiones de entrenamiento y validación.
4. Tratamiento de valores faltantes con reglas específicas por tipo de variable.
5. Codificación de variables categóricas con one-hot encoding.
6. Entrenamiento de varios modelos de regresión.
7. Comparación mediante métricas de validación y validación cruzada.
8. Selección del modelo predictivo final.
9. Análisis de errores.
10. Interpretación de asociaciones usando coeficientes Ridge.

## 5. Modelos comparados

Se evaluaron los siguientes modelos:

| Modelo | Propósito |
|---|---|
| Baseline con mediana | Punto mínimo de comparación |
| Linear Regression | Modelo simple e interpretable |
| Ridge Regression | Modelo lineal regularizado |
| Lasso Regression | Modelo lineal regularizado con selección parcial de variables |
| Random Forest Regressor | Modelo no lineal de ensamble |

## 6. Resultados principales

### 6.1 Validación simple

En la partición inicial de validación, Linear Regression obtuvo el menor error:

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 15,074.87 | 22,906.51 | 0.9316 |
| Ridge Regression | 16,434.82 | 25,110.06 | 0.9178 |
| Lasso Regression | 16,328.30 | 25,122.65 | 0.9177 |
| Random Forest | 17,349.97 | 29,673.74 | 0.8852 |
| Baseline con mediana | 59,568.25 | 88,667.17 | -0.0250 |

Sin embargo, depender solo de una partición de validación sería débil, porque el dataset es pequeño y contiene muchas variables categóricas codificadas.

### 6.2 La validación cruzada cambió la decisión

La validación cruzada mostró que Linear Regression fue inestable entre folds.

Random Forest obtuvo el mejor RMSE promedio y mejor estabilidad:

| Modelo | RMSE promedio | Desviación RMSE | R² promedio |
|---|---:|---:|---:|
| Random Forest | 30,537.30 | 6,119.20 | 0.8508 |
| Ridge Regression | 43,337.81 | 36,656.64 | 0.5981 |
| Lasso Regression | 45,883.63 | 42,185.14 | 0.5240 |
| Linear Regression | 52,687.83 | 49,190.82 | 0.3637 |
| Baseline con mediana | 81,317.92 | 5,639.73 | -0.0565 |

## 7. Decisión del modelo final

El modelo predictivo final es **Random Forest**.

La decisión se basa en la estabilidad de validación cruzada, no solo en el mejor resultado de una partición simple.

Ridge Regression se usa por separado para interpretación, porque sus coeficientes regularizados son más fáciles de explicar que la estructura interna de Random Forest.

## 8. Análisis de errores del modelo final

El modelo final Random Forest obtuvo el siguiente desempeño en validación:

| Métrica | Valor |
|---|---:|
| MAE | 17,349.97 |
| RMSE | 29,673.74 |
| R² | 0.8852 |
| Error absoluto mediano | 9,980.23 |
| Error porcentual absoluto promedio | 10.19% |
| Error porcentual absoluto mediano | 6.29% |

La diferencia entre MAE y RMSE indica que un número pequeño de errores grandes eleva el error total.

Los errores más grandes se concentran en propiedades de mayor valor o características atípicas.

## 9. Interpretación de variables

Los coeficientes Ridge sugieren que los siguientes factores se asocian positivamente con el precio:

- Barrios premium como StoneBr, Crawfor, NridgHt y NoRidge.
- Calidad general del inmueble.
- Área habitable sobre el nivel del suelo.
- Excelente calidad de sótano.
- Capacidad de garaje.
- Excelente calidad de cocina.
- Funcionalidad normal.

Las asociaciones negativas incluyen:

- Barrios de menor valor.
- Condiciones de venta anormales.
- Algunas categorías de zonificación.
- Problemas funcionales.
- Características menos deseables del inmueble.

Estos coeficientes deben interpretarse como asociaciones, no como efectos causales. Algunas asociaciones pueden estar influenciadas por categorías raras y deben tratarse como señales direccionales, no como reglas definitivas de negocio.

## 10. Recomendaciones de negocio

1. Usar el modelo como herramienta de apoyo a precios, no como sistema automático de valoración.
2. Revisar manualmente predicciones con alto error, especialmente en propiedades de lujo o atípicas.
3. Usar Random Forest para predicción y coeficientes Ridge para comunicación con stakeholders.
4. Combinar el resultado del modelo con conocimiento local del mercado antes de tomar decisiones de precio.
5. Mejorar modelos futuros con mayor granularidad geográfica, detalles de remodelación, timing de mercado, ventas comparables e indicadores macroeconómicos.

## 11. Limitaciones

- El dataset es pequeño para un problema de regresión con muchas variables.
- El conjunto de prueba de Kaggle no incluye precios reales, por lo que la validación depende de particiones internas y validación cruzada.
- Las variables categóricas con one-hot encoding pueden volver inestables a los modelos lineales.
- El modelo no incluye coordenadas geográficas exactas, calidad de colegios, tasas de interés, demanda local ni ventas comparables.
- La interpretación del modelo es asociativa, no causal.
- Algunas categorías raras pueden distorsionar la interpretación de ciertos coeficientes.

## 12. Próximos pasos

- Afinar hiperparámetros de Random Forest.
- Comparar con Gradient Boosting o XGBoost.
- Crear variables derivadas como antigüedad de vivienda, antigüedad de remodelación y área total.
- Construir una API de predicción en un futuro proyecto de despliegue.
- Crear un dashboard para explicar factores de precio a usuarios de negocio.
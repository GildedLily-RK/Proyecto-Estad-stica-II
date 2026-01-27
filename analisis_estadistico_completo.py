"""
PROYECTO ESTADÍSTICA II - UMSA
Análisis de Métodos y Hábitos de Estudio

Este script replica TODOS los cálculos de las secciones 8 y 9 del documento:
- Sección 8: Resultados y Análisis (descriptivos e inferenciales)
- Sección 9: Desarrollo de los Objetivos (pruebas avanzadas)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CARGA Y LIMPIEZA DE DATOS
# ============================================================================

print("="*80)
print("ANÁLISIS ESTADÍSTICO - MÉTODOS Y HÁBITOS DE ESTUDIO UMSA")
print("="*80)
print()

# Cargar datos
df = pd.read_excel('RespuestasTest.xlsx')
print(f"Total de respuestas cargadas: {len(df)}")
print()

# Renombrar columnas para facilitar el trabajo
df.columns = [
    'marca_temporal', 'puntuacion', 'edad', 'genero', 'semestre',
    'metodo_estudio', 'horas_estudio', 'frecuencia_estudio', 'nota_final',
    'aprobo_todas', 'rendimiento_comparado', 'metodo_efectivo',
    'nota_antes_cambio', 'metodo_anterior', 'mejora_tras_cambio',
    'usa_agenda', 'usa_herramientas_digitales', 'por_que_recomienda',
    'cambio_metodo', 'nota_antes_cambio_2'
]

# ============================================================================
# SECCIÓN 8.1: CARACTERÍSTICAS DE LA MUESTRA
# ============================================================================

print("="*80)
print("8.1 CARACTERÍSTICAS DE LA MUESTRA")
print("="*80)
print()

# Género
print("GÉNERO:")
genero_freq = df['genero'].value_counts()
genero_pct = df['genero'].value_counts(normalize=True) * 100
print(f"  Femenino: {genero_freq.get('Femenino', genero_freq.get('F', 0))} ({genero_pct.get('Femenino', genero_pct.get('F', 0)):.1f}%)")
print(f"  Masculino: {genero_freq.get('Masculino', genero_freq.get('M', 0))} ({genero_pct.get('Masculino', genero_pct.get('M', 0)):.1f}%)")
print()

# Edad
print("EDAD:")
print(f"  Media: {df['edad'].mean():.1f} años")
print(f"  Mediana: {df['edad'].median():.0f} años")
print(f"  Desviación Estándar: {df['edad'].std():.1f} años")
print()

# Distribución por edad
df['rango_edad'] = pd.cut(df['edad'], bins=[17, 20, 23, 100], labels=['18-20', '21-23', '24+'])
edad_dist = df['rango_edad'].value_counts()
edad_pct = df['rango_edad'].value_counts(normalize=True) * 100
print("  Distribución por rangos:")
for rango in ['18-20', '21-23', '24+']:
    if rango in edad_dist.index:
        print(f"    {rango} años: {edad_dist[rango]} ({edad_pct[rango]:.1f}%)")
print()

# Semestre
print("SEMESTRE:")
semestre_freq = df['semestre'].value_counts()
semestre_pct = df['semestre'].value_counts(normalize=True) * 100
print(f"  Total 3° y 4° semestre: {semestre_freq.get('3', 0) + semestre_freq.get('4', 0)} "
      f"({(semestre_pct.get('3', 0) + semestre_pct.get('4', 0)):.1f}%)")
print()

# ============================================================================
# SECCIÓN 8.2: ESTADÍSTICA DESCRIPTIVA DE HÁBITOS Y TÉCNICAS
# ============================================================================

print("="*80)
print("8.2 ESTADÍSTICA DESCRIPTIVA DE HÁBITOS Y TÉCNICAS DE ESTUDIO")
print("="*80)
print()

# Tabla de técnicas de estudio
print("TÉCNICAS DE ESTUDIO:")
print("-" * 80)
print(f"{'Técnica':<35} {'Frecuencia':<12} {'Porcentaje':<15} {'Efectividad'}")
print("-" * 80)

metodo_freq = df['metodo_estudio'].value_counts()
metodo_pct = df['metodo_estudio'].value_counts(normalize=True) * 100

for metodo in metodo_freq.index:
    freq = metodo_freq[metodo]
    pct = metodo_pct[metodo]
    # Calcular efectividad promedio para ese método
    efectividad = df[df['metodo_estudio'] == metodo]['metodo_efectivo'].mean()
    print(f"{metodo:<35} {freq:<12} {pct:>6.1f}%         {efectividad:.1f}/5")

print("-" * 80)
print()

# ============================================================================
# SECCIÓN 8.3: ESTADÍSTICAS DE TIEMPO DE ESTUDIO
# ============================================================================

print("="*80)
print("8.3 ESTADÍSTICAS DE TIEMPO DE ESTUDIO")
print("="*80)
print()

# Convertir horas de estudio a numérico
def convertir_horas(valor):
    if pd.isna(valor):
        return np.nan
    valor_str = str(valor).strip()
    # Buscar patrones comunes
    if '-' in valor_str:
        partes = valor_str.split('-')
        try:
            return (float(partes[0]) + float(partes[1])) / 2
        except:
            return np.nan
    try:
        return float(valor_str.split()[0])
    except:
        return np.nan

df['horas_numericas'] = df['horas_estudio'].apply(convertir_horas)

print("HORAS SEMANALES DE ESTUDIO:")
print(f"  Media: {df['horas_numericas'].mean():.1f} horas")
print(f"  Mediana: {df['horas_numericas'].median():.1f} horas")
print(f"  Desviación Estándar: {df['horas_numericas'].std():.1f} horas")
print(f"  Rango: {df['horas_numericas'].min():.0f} a {df['horas_numericas'].max():.0f} horas")
print()

# Moda de horas
horas_moda = df['horas_estudio'].mode()[0] if len(df['horas_estudio'].mode()) > 0 else 'N/A'
horas_moda_count = (df['horas_estudio'] == horas_moda).sum()
print(f"  Moda: {horas_moda} ({horas_moda_count} casos)")
print()

print("FRECUENCIA DE ESTUDIO (escala 1-5):")
print(f"  Media: {df['frecuencia_estudio'].mean():.1f}")
print(f"  Mediana: {df['frecuencia_estudio'].median():.1f}")
print(f"  Desviación Estándar: {df['frecuencia_estudio'].std():.1f}")
print(f"  Rango: {df['frecuencia_estudio'].min():.0f} a {df['frecuencia_estudio'].max():.0f}")
frecuencia_moda = df['frecuencia_estudio'].mode()[0]
frecuencia_moda_count = (df['frecuencia_estudio'] == frecuencia_moda).sum()
print(f"  Moda: {frecuencia_moda} ({frecuencia_moda_count} casos)")
print()

# ============================================================================
# SECCIÓN 8.4: ANÁLISIS DE RENDIMIENTO ACADÉMICO
# ============================================================================

print("="*80)
print("8.4 ANÁLISIS DE RENDIMIENTO ACADÉMICO")
print("="*80)
print()

# Convertir notas a numérico
def convertir_nota(valor):
    if pd.isna(valor):
        return np.nan
    try:
        return float(str(valor).strip())
    except:
        return np.nan

df['nota_numerica'] = df['nota_final'].apply(convertir_nota)

# Clasificar notas por rangos
def clasificar_nota(nota):
    if pd.isna(nota):
        return 'Sin dato'
    if nota >= 81:
        return '81-100'
    elif nota >= 66:
        return '66-80'
    elif nota >= 51:
        return '51-65'
    else:
        return '<51'

df['rango_nota'] = df['nota_numerica'].apply(clasificar_nota)

print("DISTRIBUCIÓN DE NOTAS:")
print("-" * 60)
print(f"{'Rango de Notas':<20} {'Frecuencia':<15} {'Porcentaje'}")
print("-" * 60)

rangos_orden = ['81-100', '66-80', '51-65', '<51']
for rango in rangos_orden:
    freq = (df['rango_nota'] == rango).sum()
    pct = (freq / len(df)) * 100
    print(f"{rango:<20} {freq:<15} {pct:>6.1f}%")

print("-" * 60)
print()

# Aprobación de materias
print("APROBACIÓN DE MATERIAS:")
print("-" * 60)

aprobo_freq = df['aprobo_todas'].value_counts()
aprobo_pct = df['aprobo_todas'].value_counts(normalize=True) * 100

# Calcular notas promedio por situación
df['aprobo_binario'] = df['aprobo_todas'].apply(lambda x: 'Aprobó todas' if 'Si' in str(x) or 'Sí' in str(x) else 'No aprobó todas')

for situacion in ['Aprobó todas', 'No aprobó todas']:
    count = (df['aprobo_binario'] == situacion).sum()
    pct = (count / len(df)) * 100
    nota_promedio = df[df['aprobo_binario'] == situacion]['nota_numerica'].mean()
    print(f"{situacion:<25} {count:<10} {pct:>6.1f}%    Nota promedio: {nota_promedio:.1f}")

print("-" * 60)
print()

# ============================================================================
# SECCIÓN 8.5: ANÁLISIS INFERENCIAL
# ============================================================================

print("="*80)
print("8.5 ANÁLISIS INFERENCIAL")
print("="*80)
print()

# ============================================================================
# 8.5.1: PRUEBA DE HIPÓTESIS PARA UNA PROPORCIÓN
# ============================================================================

print("-" * 80)
print("8.5.1 PRUEBA DE HIPÓTESIS PARA UNA PROPORCIÓN")
print("-" * 80)
print()

# Definir técnicas activas
tecnicas_activas = ['Ejercicios prácticos', 'Técnica Pomodoro', 'Tecnica Feyman']
df['usa_tecnica_activa'] = df['metodo_estudio'].apply(lambda x: any(t in str(x) for t in tecnicas_activas))

n = len(df)
x = df['usa_tecnica_activa'].sum()
p_hat = x / n
p0 = 0.5

print(f"Hipótesis:")
print(f"  H₀: p = 0.5 (50% usan técnicas activas)")
print(f"  H₁: p > 0.5 (más del 50% usan técnicas activas)")
print()
print(f"Datos:")
print(f"  n = {n}")
print(f"  x = {x}")
print(f"  p̂ = {p_hat:.3f}")
print()

# Calcular estadístico Z
z_stat = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
p_value = 1 - stats.norm.cdf(z_stat)

print(f"Estadístico de prueba:")
print(f"  Z = {z_stat:.2f}")
print(f"  Valor-p = {p_value:.3f}")
print()
print(f"Conclusión (α = 0.05):")
if p_value < 0.05:
    print(f"  Se RECHAZA H₀. Hay evidencia de que más del 50% usan técnicas activas.")
else:
    print(f"  NO se rechaza H₀. No hay evidencia suficiente.")
print()

# ============================================================================
# 8.5.2: INTERVALO DE CONFIANZA PARA PROPORCIÓN
# ============================================================================

print("-" * 80)
print("8.5.2 INTERVALO DE CONFIANZA PARA PROPORCIÓN")
print("-" * 80)
print()

# Proporción de estudiantes que aprueban todas las materias
n_ic = len(df)
x_ic = (df['aprobo_binario'] == 'Aprobó todas').sum()
p_hat_ic = x_ic / n_ic

# Intervalo de confianza al 95%
z_alpha = 1.96
margen_error = z_alpha * np.sqrt(p_hat_ic * (1 - p_hat_ic) / n_ic)
ic_inferior = p_hat_ic - margen_error
ic_superior = p_hat_ic + margen_error

print(f"Proporción de estudiantes que aprueban todas las materias:")
print(f"  n = {n_ic}")
print(f"  x = {x_ic}")
print(f"  p̂ = {p_hat_ic:.3f}")
print()
print(f"Intervalo de Confianza al 95%:")
print(f"  IC = [{ic_inferior:.3f}, {ic_superior:.3f}]")
print(f"  IC = [{ic_inferior*100:.1f}%, {ic_superior*100:.1f}%]")
print()
print(f"Interpretación:")
print(f"  Con 95% de confianza, se estima que entre {ic_inferior*100:.1f}% y {ic_superior*100:.1f}%")
print(f"  de los estudiantes de la UMSA aprueban todas sus materias.")
print()

# ============================================================================
# 8.5.3: PRUEBA CHI-CUADRADO DE INDEPENDENCIA
# ============================================================================

print("-" * 80)
print("8.5.3 PRUEBA CHI-CUADRADO DE INDEPENDENCIA")
print("-" * 80)
print()

# Crear tabla de contingencia: Uso de agenda vs Aprobación
print("Análisis: Asociación entre uso de agenda y aprobación de materias")
print()

df['usa_agenda_binario'] = df['usa_agenda'].apply(lambda x: 'Usa agenda' if 'Si' in str(x) or 'Sí' in str(x) else 'No usa agenda')

tabla_contingencia = pd.crosstab(df['usa_agenda_binario'], df['aprobo_binario'])
print("Tabla de contingencia observada:")
print(tabla_contingencia)
print()

# Realizar prueba Chi-cuadrado
chi2, p_val_chi, dof, expected = chi2_contingency(tabla_contingencia)

print(f"Resultados de la prueba:")
print(f"  χ² = {chi2:.2f}")
print(f"  gl = {dof}")
print(f"  Valor-p = {p_val_chi:.3f}")
print()
print(f"Conclusión (α = 0.05):")
if p_val_chi < 0.05:
    print(f"  Existe asociación estadísticamente significativa.")
else:
    print(f"  NO existe asociación estadísticamente significativa entre el uso de agenda")
    print(f"  y la aprobación de materias.")
print()

# ============================================================================
# SECCIÓN 9: DESARROLLO DE LOS OBJETIVOS
# ============================================================================

print("="*80)
print("9. DESARROLLO DE LOS OBJETIVOS")
print("="*80)
print()

# ============================================================================
# 9.0.1: PRUEBA t DE STUDENT PARA MUESTRAS INDEPENDIENTES
# ============================================================================

print("-" * 80)
print("9.0.1 PRUEBA t DE STUDENT PARA MUESTRAS INDEPENDIENTES")
print("-" * 80)
print()

print("Comparación: Horas de estudio entre quienes aprueban vs no aprueban todas las materias")
print()

# Separar grupos
grupo_aprobo = df[df['aprobo_binario'] == 'Aprobó todas']['horas_numericas'].dropna()
grupo_no_aprobo = df[df['aprobo_binario'] == 'No aprobó todas']['horas_numericas'].dropna()

# Estadísticos descriptivos
print(f"Grupo A (Aprueban todas):")
print(f"  n = {len(grupo_aprobo)}")
print(f"  Media = {grupo_aprobo.mean():.1f} horas")
print(f"  DE = {grupo_aprobo.std():.1f}")
print()
print(f"Grupo B (No aprueban todas):")
print(f"  n = {len(grupo_no_aprobo)}")
print(f"  Media = {grupo_no_aprobo.mean():.1f} horas")
print(f"  DE = {grupo_no_aprobo.std():.1f}")
print()

# Prueba t de Student
t_stat, p_val_t = ttest_ind(grupo_aprobo, grupo_no_aprobo)
gl_t = len(grupo_aprobo) + len(grupo_no_aprobo) - 2

print(f"Resultados de la prueba t:")
print(f"  t = {t_stat:.2f}")
print(f"  gl = {gl_t}")
print(f"  Valor-p = {p_val_t:.3f}")
print()
print(f"Conclusión (α = 0.05):")
if p_val_t < 0.05:
    print(f"  Existe diferencia estadísticamente significativa en las horas de estudio.")
else:
    print(f"  NO existe diferencia estadísticamente significativa en las horas de estudio")
    print(f"  entre quienes aprueban y no aprueban todas las materias.")
print()

# ============================================================================
# 9.1: ANÁLISIS DE REGRESIÓN LOGÍSTICA
# ============================================================================

print("-" * 80)
print("9.1 ANÁLISIS DE REGRESIÓN LOGÍSTICA")
print("-" * 80)
print()

print("Modelo para predecir aprobación de todas las materias")
print()

# Preparar variables
df_modelo = df.copy()
df_modelo['Y_aprobo'] = (df_modelo['aprobo_binario'] == 'Aprobó todas').astype(int)
df_modelo['X_horas'] = df_modelo['horas_numericas']
df_modelo['X_agenda'] = df_modelo['usa_agenda_binario'].apply(lambda x: 1 if x == 'Usa agenda' else 0)
df_modelo['X_digitales'] = df_modelo['usa_herramientas_digitales'].apply(lambda x: 1 if 'Si' in str(x) or 'Sí' in str(x) else 0)
df_modelo['X_tecnica_activa'] = df_modelo['usa_tecnica_activa'].astype(int)

# Eliminar filas con valores faltantes
df_modelo = df_modelo[['Y_aprobo', 'X_horas', 'X_agenda', 'X_digitales', 'X_tecnica_activa']].dropna()

print(f"Variables predictoras:")
print(f"  1) Horas de estudio semanales (numérica)")
print(f"  2) Uso de agenda (binaria: 1=Sí, 0=No)")
print(f"  3) Uso de herramientas digitales (binaria: 1=Sí, 0=No)")
print(f"  4) Técnica activa vs. pasiva (binaria: 1=Activa, 0=Pasiva)")
print()

# Preparar X e y
X = df_modelo[['X_horas', 'X_agenda', 'X_digitales', 'X_tecnica_activa']]
y = df_modelo['Y_aprobo']

print(f"Tamaño de muestra para el modelo: n = {len(df_modelo)}")
print()

# Entrenar modelo
modelo = LogisticRegression(random_state=42, max_iter=1000)
modelo.fit(X, y)

# Obtener coeficientes
intercepto = modelo.intercept_[0]
coeficientes = modelo.coef_[0]

print(f"Resultados del modelo:")
print(f"  Intercepto: {intercepto:.2f}")
print()
print(f"Coeficientes:")
nombres_vars = ['Horas estudio', 'Uso agenda', 'Herramientas digitales', 'Técnica activa']
for nombre, coef in zip(nombres_vars, coeficientes):
    odds_ratio = np.exp(coef)
    print(f"  {nombre}: {coef:.2f} (OR = {odds_ratio:.2f})")
print()

print(f"Interpretación:")
print(f"  - Por cada hora adicional de estudio, la probabilidad de aprobar")
print(f"    todas las materias aumenta {(np.exp(coeficientes[0]) - 1) * 100:.0f}%")
print(f"  - Usar agenda aumenta la probabilidad en {(np.exp(coeficientes[1]) - 1) * 100:.0f}%")
print(f"  - Usar herramientas digitales modifica la probabilidad en {(np.exp(coeficientes[2]) - 1) * 100:.0f}%")
print(f"  - Las técnicas activas aumentan la probabilidad en {(np.exp(coeficientes[3]) - 1) * 100:.0f}%")
print()

# Calcular pseudo R²
from sklearn.metrics import log_loss
y_pred_proba = modelo.predict_proba(X)[:, 1]
null_model_loss = log_loss(y, [y.mean()] * len(y))
model_loss = log_loss(y, y_pred_proba)
pseudo_r2 = 1 - (model_loss / null_model_loss)

print(f"Pseudo R²: {pseudo_r2:.2f}")
print()

# ============================================================================
# 9.2: ANÁLISIS CUALITATIVO DE RECOMENDACIONES
# ============================================================================

print("-" * 80)
print("9.2 ANÁLISIS CUALITATIVO DE RECOMENDACIONES")
print("-" * 80)
print()

print("Temas emergentes de respuestas abiertas")
print("(¿Por qué recomendarías tu técnica?)")
print()

# Análisis de palabras clave
respuestas = df['por_que_recomienda'].dropna()
total_respuestas = len(respuestas)

# Definir categorías
categorias = {
    'Concentración y enfoque': ['concentr', 'enfoc', 'atención', 'distrac'],
    'Retención y memoria': ['record', 'memori', 'retener', 'retención'],
    'Práctica y aplicación': ['prácti', 'ejerci', 'aplicar', 'hacer'],
    'Organización': ['organiz', 'planific', 'estructur', 'orden']
}

print("Categorías identificadas:")
for categoria, palabras_clave in categorias.items():
    count = sum(respuestas.str.lower().str.contains('|'.join(palabras_clave), na=False))
    porcentaje = (count / total_respuestas) * 100
    print(f"  {categoria}: {porcentaje:.0f}%")
print()

# ============================================================================
# 9.3: SÍNTESIS DE HALLAZGOS PRINCIPALES
# ============================================================================

print("-" * 80)
print("9.3 SÍNTESIS DE HALLAZGOS PRINCIPALES")
print("-" * 80)
print()

metodo_principal = df['metodo_estudio'].mode()[0]
pct_metodo_principal = (df['metodo_estudio'] == metodo_principal).sum() / len(df) * 100

rango_principal = df['rango_nota'].mode()[0]
pct_rango_principal = (df['rango_nota'] == rango_principal).sum() / len(df) * 100

pct_aprobo = (df['aprobo_binario'] == 'Aprobó todas').sum() / len(df) * 100

mejora_count = df['rendimiento_comparado'].str.contains('mejor', case=False, na=False).sum()
pct_mejora = (mejora_count / len(df)) * 100

print(f"✓ Perfil predominante:")
print(f"    Mujer, {df['edad'].mean():.0f} años, {df['semestre'].mode()[0]}° semestre")
print()
print(f"✓ Técnica preferida:")
print(f"    {metodo_principal} ({pct_metodo_principal:.1f}%)")
print()
print(f"✓ Tiempo de estudio:")
print(f"    {df['horas_numericas'].mean():.1f} horas semanales en promedio")
print()
print(f"✓ Rendimiento:")
print(f"    {pct_rango_principal:.1f}% obtienen notas en rango {rango_principal}")
print()
print(f"✓ Aprobación:")
print(f"    Solo {pct_aprobo:.1f}% aprueban todas las materias")
print()
print(f"✓ Percepción:")
print(f"    {pct_mejora:.1f}% perciben mejoría respecto a semestres anteriores")
print()
print(f"✓ Factores asociados:")
print(f"    Uso de agenda y técnicas activas muestran tendencia positiva")
print()

print("="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)

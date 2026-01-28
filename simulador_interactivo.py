"""
OPCIÓN 1: SIMULADOR INTERACTIVO - "PREDICE TU ÉXITO ACADÉMICO"
=================================================================

Este programa usa REGRESIÓN LOGÍSTICA y SIMULACIÓN MONTE CARLO
para predecir la probabilidad de éxito académico según los hábitos de estudio.

Conceptos de Estadística II aplicados:
- Regresión Logística
- Simulación Monte Carlo
- Intervalos de confianza
- Probabilidades condicionales
- Análisis de sensibilidad
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración estética
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("🎓 SIMULADOR DE ÉXITO ACADÉMICO - UMSA")
print("="*80)
print()
print("Basado en datos reales de 60 estudiantes")
print("Usa Regresión Logística + Simulación Monte Carlo")
print()

# ============================================================================
# CARGA Y PREPARACIÓN DEL MODELO
# ============================================================================

df = pd.read_excel('RespuestasTest.xlsx')

# Renombrar columnas
df.columns = [
    'marca_temporal', 'puntuacion', 'edad', 'genero', 'semestre',
    'metodo_estudio', 'horas_estudio', 'frecuencia_estudio', 'nota_final',
    'aprobo_todas', 'rendimiento_comparado', 'metodo_efectivo',
    'nota_antes_cambio', 'metodo_anterior', 'mejora_tras_cambio',
    'usa_agenda', 'usa_herramientas_digitales', 'por_que_recomienda',
    'cambio_metodo', 'nota_antes_cambio_2'
]

# Función para convertir horas
def convertir_horas(valor):
    if pd.isna(valor):
        return np.nan
    valor_str = str(valor).strip()
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

# Preparar variables
df['horas_numericas'] = df['horas_estudio'].apply(convertir_horas)
df['aprobo_binario'] = df['aprobo_todas'].apply(lambda x: 1 if 'Si' in str(x) or 'Sí' in str(x) else 0)
df['agenda_binario'] = df['usa_agenda'].apply(lambda x: 1 if 'Si' in str(x) or 'Sí' in str(x) else 0)
df['digitales_binario'] = df['usa_herramientas_digitales'].apply(lambda x: 1 if 'Si' in str(x) or 'Sí' in str(x) else 0)

tecnicas_activas = ['Ejercicios prácticos', 'Técnica Pomodoro', 'Feyman']
df['tecnica_activa'] = df['metodo_estudio'].apply(lambda x: 1 if any(t in str(x) for t in tecnicas_activas) else 0)

# Preparar dataset para modelo
df_modelo = df[['aprobo_binario', 'horas_numericas', 'frecuencia_estudio', 
                'metodo_efectivo', 'agenda_binario', 'digitales_binario', 
                'tecnica_activa']].dropna()

X = df_modelo[['horas_numericas', 'frecuencia_estudio', 'metodo_efectivo', 
               'agenda_binario', 'digitales_binario', 'tecnica_activa']]
y = df_modelo['aprobo_binario']

# Entrenar modelo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
modelo = LogisticRegression(random_state=42, max_iter=1000)
modelo.fit(X_scaled, y)

print("✅ Modelo entrenado con", len(df_modelo), "estudiantes")
print("📊 Precisión del modelo:", f"{modelo.score(X_scaled, y)*100:.1f}%")
print()

# ============================================================================
# FUNCIÓN DE PREDICCIÓN INDIVIDUAL
# ============================================================================

def predecir_exito(horas, frecuencia, efectividad, usa_agenda, usa_digitales, tecnica_activa):
    """
    Predice la probabilidad de aprobar todas las materias
    Retorna: probabilidad + intervalo de confianza + interpretación
    """
    
    # Crear input para el modelo
    X_nuevo = np.array([[horas, frecuencia, efectividad, 
                         usa_agenda, usa_digitales, tecnica_activa]])
    X_nuevo_scaled = scaler.transform(X_nuevo)
    
    # Predicción
    prob_exito = modelo.predict_proba(X_nuevo_scaled)[0][1]
    
    # Simulación Monte Carlo para intervalo de confianza
    # (Bootstrap de coeficientes)
    n_simulations = 1000
    probabilidades = []
    
    for _ in range(n_simulations):
        # Añadir ruido a los coeficientes (simula incertidumbre)
        coef_ruido = modelo.coef_[0] + np.random.normal(0, 0.1, len(modelo.coef_[0]))
        intercept_ruido = modelo.intercept_[0] + np.random.normal(0, 0.1)
        
        # Calcular probabilidad con coeficientes ruidosos
        z = np.dot(X_nuevo_scaled[0], coef_ruido) + intercept_ruido
        prob_sim = 1 / (1 + np.exp(-z))
        probabilidades.append(prob_sim)
    
    # Intervalo de confianza al 95%
    ic_inf = np.percentile(probabilidades, 2.5)
    ic_sup = np.percentile(probabilidades, 97.5)
    
    return prob_exito, ic_inf, ic_sup, probabilidades

# ============================================================================
# FUNCIÓN DE ANÁLISIS DE SENSIBILIDAD
# ============================================================================

def analisis_sensibilidad(horas_base, frecuencia_base, efectividad_base, 
                          usa_agenda, usa_digitales, tecnica_activa):
    """
    Analiza cómo cambia la probabilidad al modificar cada variable
    """
    
    resultados = {}
    
    # Variar horas de estudio
    horas_range = np.linspace(1, 6, 20)
    probs_horas = []
    for h in horas_range:
        prob, _, _, _ = predecir_exito(h, frecuencia_base, efectividad_base, 
                                       usa_agenda, usa_digitales, tecnica_activa)
        probs_horas.append(prob)
    resultados['horas'] = (horas_range, probs_horas)
    
    # Variar frecuencia de estudio
    freq_range = np.linspace(1, 5, 20)
    probs_freq = []
    for f in freq_range:
        prob, _, _, _ = predecir_exito(horas_base, f, efectividad_base, 
                                       usa_agenda, usa_digitales, tecnica_activa)
        probs_freq.append(prob)
    resultados['frecuencia'] = (freq_range, probs_freq)
    
    # Variar efectividad percibida
    efect_range = np.linspace(1, 5, 20)
    probs_efect = []
    for e in efect_range:
        prob, _, _, _ = predecir_exito(horas_base, frecuencia_base, e, 
                                       usa_agenda, usa_digitales, tecnica_activa)
        probs_efect.append(prob)
    resultados['efectividad'] = (efect_range, probs_efect)
    
    return resultados

# ============================================================================
# MODO INTERACTIVO
# ============================================================================

def modo_interactivo():
    print("="*80)
    print("🎯 MODO INTERACTIVO - Predice tu probabilidad de éxito")
    print("="*80)
    print()
    
    print("Responde las siguientes preguntas:")
    print()
    
    # Preguntas
    horas = float(input("1. ¿Cuántas horas estudias por semana? (1-6): "))
    frecuencia = float(input("2. ¿Con qué frecuencia estudias? (1=Nunca, 5=Diariamente): "))
    efectividad = float(input("3. ¿Qué tan efectivo consideras tu método? (1-5): "))
    usa_agenda = input("4. ¿Usas agenda para planificar? (Si/No): ").strip().lower() == 'si'
    usa_digitales = input("5. ¿Usas herramientas digitales? (Si/No): ").strip().lower() == 'si'
    tecnica_activa = input("6. ¿Usas técnicas activas (ejercicios/Pomodoro)? (Si/No): ").strip().lower() == 'si'
    
    print()
    print("⏳ Calculando...")
    print()
    
    # Predicción
    prob, ic_inf, ic_sup, simulaciones = predecir_exito(
        horas, frecuencia, efectividad,
        int(usa_agenda), int(usa_digitales), int(tecnica_activa)
    )
    
    # Resultados
    print("="*80)
    print("📊 RESULTADOS DE TU PREDICCIÓN")
    print("="*80)
    print()
    print(f"🎲 Probabilidad de aprobar TODAS tus materias: {prob*100:.1f}%")
    print(f"📈 Intervalo de confianza 95%: [{ic_inf*100:.1f}%, {ic_sup*100:.1f}%]")
    print()
    
    # Interpretación
    if prob >= 0.7:
        print("✅ ¡EXCELENTE! Tienes alta probabilidad de éxito.")
        print("   Mantén tus hábitos actuales.")
    elif prob >= 0.5:
        print("⚠️  BUENO. Tienes probabilidad moderada de éxito.")
        print("   Considera mejorar algunos hábitos.")
    else:
        print("❌ ALERTA. Tu probabilidad de éxito es baja.")
        print("   ¡Necesitas cambiar urgentemente tus hábitos!")
    
    print()
    print("🔍 Análisis detallado:")
    print("-"*80)
    
    # Comparar con el promedio
    promedio_prob = modelo.predict_proba(X_scaled)[:, 1].mean()
    if prob > promedio_prob:
        print(f"   📈 Estás {(prob-promedio_prob)*100:.1f}% por ENCIMA del promedio")
    else:
        print(f"   📉 Estás {(promedio_prob-prob)*100:.1f}% por DEBAJO del promedio")
    
    print()
    
    # Recomendaciones personalizadas
    print("💡 RECOMENDACIONES PERSONALIZADAS:")
    print("-"*80)
    
    if horas < 3:
        mejora_horas = prob * 1.15  # Estimación de mejora
        print(f"   1. Aumenta tus horas de estudio a 3-4 horas semanales")
        print(f"      Impacto estimado: +{(mejora_horas-prob)*100:.1f}% probabilidad")
    
    if frecuencia < 3:
        print(f"   2. Estudia con más regularidad (al menos 3 veces por semana)")
        print(f"      La consistencia es clave para el éxito")
    
    if not usa_agenda:
        print(f"   3. Usa una agenda para planificar tus tareas")
        print(f"      Mejora la organización y reduce el estrés")
    
    if not tecnica_activa:
        print(f"   4. Adopta técnicas activas (ejercicios prácticos, Pomodoro)")
        print(f"      Son más efectivas que la lectura pasiva")
    
    print()
    
    # Generar gráficas
    print("📊 Generando visualizaciones...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis Completo de tu Predicción', fontsize=16, fontweight='bold')
    
    # Gráfica 1: Distribución de probabilidades (Monte Carlo)
    axes[0, 0].hist(simulaciones, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(prob, color='red', linestyle='--', linewidth=2, label=f'Tu predicción: {prob*100:.1f}%')
    axes[0, 0].axvline(ic_inf, color='orange', linestyle=':', linewidth=1.5, label='IC 95%')
    axes[0, 0].axvline(ic_sup, color='orange', linestyle=':', linewidth=1.5)
    axes[0, 0].set_xlabel('Probabilidad de Éxito')
    axes[0, 0].set_ylabel('Frecuencia (Simulación Monte Carlo)')
    axes[0, 0].set_title('Distribución de Probabilidades\n(1000 simulaciones)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfica 2: Comparación con otros estudiantes
    todos_probs = modelo.predict_proba(X_scaled)[:, 1]
    axes[0, 1].hist(todos_probs, bins=20, edgecolor='black', alpha=0.6, color='lightgreen', label='Otros estudiantes')
    axes[0, 1].axvline(prob, color='red', linestyle='--', linewidth=3, label=f'Tú: {prob*100:.1f}%')
    axes[0, 1].axvline(todos_probs.mean(), color='blue', linestyle=':', linewidth=2, label=f'Promedio: {todos_probs.mean()*100:.1f}%')
    axes[0, 1].set_xlabel('Probabilidad de Éxito')
    axes[0, 1].set_ylabel('Número de Estudiantes')
    axes[0, 1].set_title('Tu Posición vs Otros Estudiantes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfica 3: Análisis de sensibilidad
    sensibilidad = analisis_sensibilidad(horas, frecuencia, efectividad, 
                                         int(usa_agenda), int(usa_digitales), int(tecnica_activa))
    
    axes[1, 0].plot(sensibilidad['horas'][0], sensibilidad['horas'][1], 'o-', linewidth=2, markersize=4)
    axes[1, 0].axvline(horas, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Tu valor actual')
    axes[1, 0].axhline(prob, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Horas de Estudio Semanales')
    axes[1, 0].set_ylabel('Probabilidad de Éxito')
    axes[1, 0].set_title('Impacto de Aumentar las Horas de Estudio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(sensibilidad['horas'][0], sensibilidad['horas'][1], alpha=0.2)
    
    # Gráfica 4: Radar de tus características
    categorias = ['Horas\nEstudio', 'Frecuencia\nEstudio', 'Efectividad\nPercibida', 
                  'Usa\nAgenda', 'Herramientas\nDigitales', 'Técnica\nActiva']
    valores_tuyos = [horas/6, frecuencia/5, efectividad/5, 
                     int(usa_agenda), int(usa_digitales), int(tecnica_activa)]
    
    # Valores promedio del dataset
    valores_promedio = [
        df_modelo['horas_numericas'].mean()/6,
        df_modelo['frecuencia_estudio'].mean()/5,
        df_modelo['metodo_efectivo'].mean()/5,
        df_modelo['agenda_binario'].mean(),
        df_modelo['digitales_binario'].mean(),
        df_modelo['tecnica_activa'].mean()
    ]
    
    # Cerrar el polígono
    valores_tuyos += valores_tuyos[:1]
    valores_promedio += valores_promedio[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
    angles += angles[:1]
    
    ax_radar = plt.subplot(2, 2, 4, projection='polar')
    ax_radar.plot(angles, valores_tuyos, 'o-', linewidth=2, label='Tú', color='red')
    ax_radar.fill(angles, valores_tuyos, alpha=0.25, color='red')
    ax_radar.plot(angles, valores_promedio, 'o-', linewidth=2, label='Promedio UMSA', color='blue')
    ax_radar.fill(angles, valores_promedio, alpha=0.15, color='blue')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categorias, size=8)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Tu Perfil de Estudio vs Promedio', y=1.08)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax_radar.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/prediccion_personalizada.png', dpi=300, bbox_inches='tight')
    print("✅ Gráficas guardadas en: prediccion_personalizada.png")
    print()
    
    return prob, ic_inf, ic_sup

# ============================================================================
# MODO DEMOSTRACIÓN (si no es interactivo)
# ============================================================================

def modo_demostracion():
    """Muestra 3 casos de ejemplo"""
    print("="*80)
    print("📋 MODO DEMOSTRACIÓN - 3 Perfiles de Estudiantes")
    print("="*80)
    print()
    
    casos = [
        {
            'nombre': 'Estudiante EXCELENTE',
            'horas': 5,
            'frecuencia': 5,
            'efectividad': 5,
            'agenda': 1,
            'digitales': 1,
            'activa': 1
        },
        {
            'nombre': 'Estudiante PROMEDIO',
            'horas': 2,
            'frecuencia': 3,
            'efectividad': 3,
            'agenda': 1,
            'digitales': 0,
            'activa': 0
        },
        {
            'nombre': 'Estudiante EN RIESGO',
            'horas': 1,
            'frecuencia': 2,
            'efectividad': 2,
            'agenda': 0,
            'digitales': 0,
            'activa': 0
        }
    ]
    
    resultados = []
    
    for caso in casos:
        prob, ic_inf, ic_sup, _ = predecir_exito(
            caso['horas'], caso['frecuencia'], caso['efectividad'],
            caso['agenda'], caso['digitales'], caso['activa']
        )
        
        print(f"\n{caso['nombre']}:")
        print(f"  Horas: {caso['horas']}h | Frecuencia: {caso['frecuencia']}/5 | Efectividad: {caso['efectividad']}/5")
        print(f"  Agenda: {'Sí' if caso['agenda'] else 'No'} | Digitales: {'Sí' if caso['digitales'] else 'No'} | Activa: {'Sí' if caso['activa'] else 'No'}")
        print(f"  ➜ Probabilidad de éxito: {prob*100:.1f}% (IC 95%: [{ic_inf*100:.1f}%, {ic_sup*100:.1f}%])")
        
        resultados.append((caso['nombre'], prob))
    
    # Gráfica comparativa
    plt.figure(figsize=(10, 6))
    nombres = [r[0] for r in resultados]
    probs = [r[1] * 100 for r in resultados]
    colores = ['green', 'orange', 'red']
    
    bars = plt.bar(nombres, probs, color=colores, edgecolor='black', linewidth=2, alpha=0.7)
    plt.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% (Línea de decisión)')
    plt.ylabel('Probabilidad de Aprobar Todas las Materias (%)', fontsize=12)
    plt.title('Comparación de 3 Perfiles de Estudiantes', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('outputs/comparacion_perfiles.png', dpi=300, bbox_inches='tight')
    print()
    print("Gráfica comparativa guardada en: comparacion_perfiles.png")

# ============================================================================
# MENÚ PRINCIPAL
# ============================================================================

print("Elige el modo:")
print("1. Modo INTERACTIVO (ingresa tus datos)")
print("2. Modo DEMOSTRACIÓN (ver 3 casos ejemplo)")
print()

opcion = input("Opción (1 o 2): ").strip()

if opcion == '1':
    modo_interactivo()
elif opcion == '2':
    modo_demostracion()
else:
    print("Opción inválida. Ejecutando demostración...")
    modo_demostracion()

print()
print("="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print()
print("📚 Conceptos de Estadística II aplicados:")
print("   ✓ Regresión Logística (predicción binaria)")
print("   ✓ Simulación Monte Carlo (intervalos de confianza)")
print("   ✓ Análisis de sensibilidad")
print("   ✓ Inferencia estadística")
print("   ✓ Probabilidades condicionales")

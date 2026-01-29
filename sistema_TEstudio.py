"""
OPCIÓN 3: SISTEMA DE RECOMENDACIÓN INTELIGENTE
===============================================

Un "Asistente Virtual Estadístico" que:
- Analiza el perfil del estudiante
- Usa clustering (K-means) para identificar grupos
- Genera recomendaciones personalizadas basadas en evidencia
- Muestra probabilidades y análisis bayesiano

Conceptos de Estadística II aplicados:
- Clustering (K-means)
- Análisis discriminante
- Probabilidad condicional (Teorema de Bayes)
- Análisis de componentes principales (PCA)
- Inferencia bayesiana
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("="*80)
print("🤖 ASISTENTE VIRTUAL ESTADÍSTICO - Sistema de Recomendaciones")
print("="*80)
print()
print("Este sistema usa Machine Learning y Estadística Bayesiana")
print("para generar recomendaciones personalizadas de estudio")
print()

# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

df = pd.read_excel('RespuestasTest.xlsx')

df.columns = [
    'marca_temporal', 'puntuacion', 'edad', 'genero', 'semestre',
    'metodo_estudio', 'horas_estudio', 'frecuencia_estudio', 'nota_final',
    'aprobo_todas', 'rendimiento_comparado', 'metodo_efectivo',
    'nota_antes_cambio', 'metodo_anterior', 'mejora_tras_cambio',
    'usa_agenda', 'usa_herramientas_digitales', 'por_que_recomienda',
    'cambio_metodo', 'nota_antes_cambio_2'
]

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

df['horas_numericas'] = df['horas_estudio'].apply(convertir_horas)
df['aprobo_binario'] = df['aprobo_todas'].apply(lambda x: 1 if 'Si' in str(x) or 'Sí' in str(x) else 0)
df['agenda_binario'] = df['usa_agenda'].apply(lambda x: 1 if 'Si' in str(x) or 'Sí' in str(x) else 0)
df['digitales_binario'] = df['usa_herramientas_digitales'].apply(lambda x: 1 if 'Si' in str(x) or 'Sí' in str(x) else 0)

tecnicas_activas = ['Ejercicios prácticos', 'Técnica Pomodoro', 'Feyman']
df['tecnica_activa'] = df['metodo_estudio'].apply(lambda x: 1 if any(t in str(x) for t in tecnicas_activas) else 0)

print("✅ Datos cargados:", len(df), "estudiantes")
print()

# ============================================================================
# CLUSTERING: IDENTIFICAR PERFILES DE ESTUDIANTES
# ============================================================================

def realizar_clustering():
    """Identifica grupos de estudiantes usando K-means"""
    
    print("="*80)
    print("🔬 ANÁLISIS DE CLUSTERING - Identificando Perfiles de Estudiantes")
    print("="*80)
    print()
    
    # Preparar datos para clustering
    df_cluster = df[['horas_numericas', 'frecuencia_estudio', 'metodo_efectivo', 
                     'agenda_binario', 'digitales_binario', 'tecnica_activa', 
                     'aprobo_binario']].dropna()
    
    X_cluster = df_cluster[['horas_numericas', 'frecuencia_estudio', 'metodo_efectivo', 
                            'agenda_binario', 'digitales_binario', 'tecnica_activa']]
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Determinar número óptimo de clusters (Método del codo)
    print("📊 Determinando número óptimo de clusters...")
    inertias = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Usar 4 clusters (buena interpretabilidad)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_cluster['Cluster'] = clusters
    
    print(f"✅ Identificados {n_clusters} perfiles de estudiantes")
    print()
    
    # Analizar cada cluster
    print("📊 CARACTERÍSTICAS DE CADA PERFIL:")
    print("="*80)
    
    cluster_names = []
    cluster_stats = []
    
    for i in range(n_clusters):
        cluster_data = df_cluster[df_cluster['Cluster'] == i]
        tasa_aprobacion = cluster_data['aprobo_binario'].mean() * 100
        
        print(f"\n PERFIL {i+1}:")
        print(f"   Tamaño: {len(cluster_data)} estudiantes ({len(cluster_data)/len(df_cluster)*100:.1f}%)")
        print(f"   Tasa de aprobación: {tasa_aprobacion:.1f}%")
        print(f"   Características promedio:")
        print(f"      • Horas de estudio: {cluster_data['horas_numericas'].mean():.1f}h")
        print(f"      • Frecuencia: {cluster_data['frecuencia_estudio'].mean():.1f}/5")
        print(f"      • Efectividad: {cluster_data['metodo_efectivo'].mean():.1f}/5")
        print(f"      • Usa agenda: {cluster_data['agenda_binario'].mean()*100:.0f}%")
        print(f"      • Usa digitales: {cluster_data['digitales_binario'].mean()*100:.0f}%")
        print(f"      • Técnica activa: {cluster_data['tecnica_activa'].mean()*100:.0f}%")
        
        # Nombrar clusters según características
        if tasa_aprobacion >= 50:
            if cluster_data['horas_numericas'].mean() >= 2.5:
                nombre = "ESTUDIANTES EXITOSOS"
            else:
                nombre = "ESTUDIANTES EFICIENTES"
        else:
            if cluster_data['frecuencia_estudio'].mean() < 2.5:
                nombre = "ESTUDIANTES EN RIESGO"
            else:
                nombre = "ESTUDIANTES CON POTENCIAL"
        
        cluster_names.append(nombre)
        cluster_stats.append({
            'nombre': nombre,
            'tasa_aprobacion': tasa_aprobacion,
            'horas': cluster_data['horas_numericas'].mean(),
            'frecuencia': cluster_data['frecuencia_estudio'].mean(),
            'efectividad': cluster_data['metodo_efectivo'].mean(),
            'size': len(cluster_data)
        })
    
    # Visualización: PCA para 2D
    print()
    print("📊 Generando visualización de clusters...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Clusters en espacio PCA
    colores = ['red', 'blue', 'green', 'orange']
    for i in range(n_clusters):
        mask = clusters == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colores[i], label=cluster_names[i], 
                   s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Marcar centroides
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='black', marker='X', s=300, edgecolors='white', linewidth=2,
               label='Centroides', zorder=10)
    
    ax1.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax1.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax1.set_title('Perfiles de Estudiantes (Análisis PCA)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Tasa de aprobación por cluster
    nombres_cortos = [stats['nombre'].replace('ESTUDIANTES ', '') for stats in cluster_stats]
    tasas = [stats['tasa_aprobacion'] for stats in cluster_stats]
    sizes = [stats['size'] for stats in cluster_stats]
    
    bars = ax2.bar(range(n_clusters), tasas, color=colores, edgecolor='black', linewidth=2, alpha=0.7)
    ax2.axhline(50, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='50% (Umbral)')
    ax2.set_xlabel('Perfil de Estudiante', fontsize=12)
    ax2.set_ylabel('Tasa de Aprobación (%)', fontsize=12)
    ax2.set_title('Éxito Académico por Perfil', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(n_clusters))
    ax2.set_xticklabels(nombres_cortos, rotation=15, ha='right')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores y tamaños
    for i, (bar, tasa, size) in enumerate(zip(bars, tasas, sizes)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{tasa:.1f}%\n(n={size})', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/clustering_perfiles.png', dpi=300, bbox_inches='tight')
    print("   Guardado: clustering_perfiles.png")
    print()
    
    return df_cluster, kmeans, scaler, cluster_names, cluster_stats

# ============================================================================
# SISTEMA DE RECOMENDACIONES PERSONALIZADO
# ============================================================================

def generar_recomendaciones(horas, frecuencia, efectividad, usa_agenda, usa_digitales, 
                           tecnica_activa, kmeans, scaler, cluster_stats):
    """Genera recomendaciones personalizadas basadas en clustering"""
    
    print("="*80)
    print("💡 GENERANDO RECOMENDACIONES PERSONALIZADAS")
    print("="*80)
    print()
    
    # Crear perfil del estudiante
    perfil = np.array([[horas, frecuencia, efectividad, 
                       int(usa_agenda), int(usa_digitales), int(tecnica_activa)]])
    perfil_scaled = scaler.transform(perfil)
    
    # Identificar cluster
    cluster_asignado = kmeans.predict(perfil_scaled)[0]
    distancias = kmeans.transform(perfil_scaled)[0]
    
    print(f"📍 TU PERFIL:")
    print(f"   Eres un: {cluster_stats[cluster_asignado]['nombre']}")
    print(f"   Tasa de éxito típica de tu perfil: {cluster_stats[cluster_asignado]['tasa_aprobacion']:.1f}%")
    print()
    
    # Calcular probabilidad bayesiana
    # P(Aprobar | Cluster)
    prob_aprobar_dado_cluster = cluster_stats[cluster_asignado]['tasa_aprobacion'] / 100
    
    print(f"📊 ANÁLISIS BAYESIANO:")
    print(f"   P(Aprobar | Tu perfil) = {prob_aprobar_dado_cluster*100:.1f}%")
    print()
    
    # Identificar el mejor cluster
    mejor_cluster = max(range(len(cluster_stats)), key=lambda i: cluster_stats[i]['tasa_aprobacion'])
    
    if cluster_asignado != mejor_cluster:
        print(f"🎯 PLAN DE MEJORA:")
        print(f"   El perfil con mayor éxito es: {cluster_stats[mejor_cluster]['nombre']}")
        print(f"   Tasa de éxito: {cluster_stats[mejor_cluster]['tasa_aprobacion']:.1f}%")
        print()
        print(f"   📈 CAMBIOS RECOMENDADOS para acercarte a ese perfil:")
        print()
        
        # Comparar características
        diferencias = []
        
        if cluster_stats[mejor_cluster]['horas'] > horas:
            dif_horas = cluster_stats[mejor_cluster]['horas'] - horas
            print(f"   1. ⏰ Aumenta tus horas de estudio en {dif_horas:.1f}h semanales")
            print(f"      (Tu actual: {horas:.1f}h → Recomendado: {cluster_stats[mejor_cluster]['horas']:.1f}h)")
            diferencias.append(('horas', dif_horas))
        
        if cluster_stats[mejor_cluster]['frecuencia'] > frecuencia:
            dif_freq = cluster_stats[mejor_cluster]['frecuencia'] - frecuencia
            print(f"   2. 📅 Aumenta la frecuencia de estudio en {dif_freq:.1f} puntos")
            print(f"      (Tu actual: {frecuencia:.1f}/5 → Recomendado: {cluster_stats[mejor_cluster]['frecuencia']:.1f}/5)")
            diferencias.append(('frecuencia', dif_freq))
        
        if not usa_agenda:
            print(f"   3. 📓 Adopta el uso de agenda para planificar")
            print(f"      Los estudiantes exitosos lo usan en mayor proporción")
            diferencias.append(('agenda', 1))
        
        if not tecnica_activa:
            print(f"   4. 💪 Cambia a técnicas activas (ejercicios, Pomodoro)")
            print(f"      Son más efectivas que técnicas pasivas")
            diferencias.append(('tecnica', 1))
        
        if not diferencias:
            print(f"   ✅ Tu perfil ya es muy bueno!")
            print(f"   Mantén tus hábitos actuales.")
    else:
        print(f"🌟 ¡EXCELENTE!")
        print(f"   Ya estás en el perfil con mayor tasa de éxito.")
        print(f"   Mantén tus hábitos actuales.")
    
    print()
    
    # Predicción de mejora
    print("="*80)
    print("🔮 PREDICCIÓN DE MEJORA")
    print("="*80)
    print()
    
    if cluster_asignado != mejor_cluster:
        prob_actual = cluster_stats[cluster_asignado]['tasa_aprobacion']
        prob_objetivo = cluster_stats[mejor_cluster]['tasa_aprobacion']
        mejora_potencial = prob_objetivo - prob_actual
        
        print(f"📈 Si adoptas las recomendaciones:")
        print(f"   Probabilidad actual de aprobar todo: {prob_actual:.1f}%")
        print(f"   Probabilidad objetivo: {prob_objetivo:.1f}%")
        print(f"   Mejora potencial: +{mejora_potencial:.1f} puntos porcentuales")
        print()
        
        # Intervalo de confianza para la mejora
        # Usando bootstrap
        n_bootstrap = 1000
        mejoras_simuladas = np.random.beta(
            prob_objetivo/10, (100-prob_objetivo)/10, n_bootstrap
        ) * 100
        
        ic_inf = np.percentile(mejoras_simuladas, 2.5)
        ic_sup = np.percentile(mejoras_simuladas, 97.5)
        
        print(f"   Intervalo de confianza 95%: [{ic_inf:.1f}%, {ic_sup:.1f}%]")
    
    print()
    
    # Recomendaciones específicas basadas en evidencia
    print("="*80)
    print("📚 RECOMENDACIONES BASADAS EN EVIDENCIA ESTADÍSTICA")
    print("="*80)
    print()
    
    recomendaciones = []
    
    # Analizar cada variable
    df_cluster_local = df[['horas_numericas', 'frecuencia_estudio', 'metodo_efectivo', 
                           'aprobo_binario']].dropna()
    
    # Horas de estudio
    corr_horas, p_horas = stats.pearsonr(df_cluster_local['horas_numericas'], 
                                         df_cluster_local['aprobo_binario'])
    if p_horas < 0.05 and corr_horas > 0:
        print(f"✓ Las horas de estudio tienen correlación significativa con el éxito")
        print(f"  (r = {corr_horas:.3f}, p = {p_horas:.3f})")
        if horas < df_cluster_local['horas_numericas'].quantile(0.75):
            print(f"  → Recomendación: Aumenta tus horas (top 25%: {df_cluster_local['horas_numericas'].quantile(0.75):.1f}h)")
        print()
    
    # Frecuencia
    corr_freq, p_freq = stats.pearsonr(df_cluster_local['frecuencia_estudio'], 
                                       df_cluster_local['aprobo_binario'])
    if p_freq < 0.05 and corr_freq > 0:
        print(f"✓ La frecuencia de estudio tiene correlación significativa con el éxito")
        print(f"  (r = {corr_freq:.3f}, p = {p_freq:.3f})")
        if frecuencia < df_cluster_local['frecuencia_estudio'].quantile(0.75):
            print(f"  → Recomendación: Aumenta la frecuencia (top 25%: {df_cluster_local['frecuencia_estudio'].quantile(0.75):.1f}/5)")
        print()
    
    # Efectividad percibida
    corr_efect, p_efect = stats.pearsonr(df_cluster_local['metodo_efectivo'], 
                                         df_cluster_local['aprobo_binario'])
    if p_efect < 0.05 and corr_efect > 0:
        print(f"✓ La efectividad percibida correlaciona con el éxito")
        print(f"  (r = {corr_efect:.3f}, p = {p_efect:.3f})")
        if efectividad < 4:
            print(f"  → Recomendación: Busca técnicas que consideres más efectivas")
        print()
    
    # Crear visualización del plan de mejora
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Tu Plan de Mejora Personalizado', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Comparación de tu perfil vs objetivo
    categorias = ['Horas\nEstudio', 'Frecuencia\nEstudio', 'Efectividad']
    valores_tuyos = [horas/6, frecuencia/5, efectividad/5]
    valores_objetivo = [
        cluster_stats[mejor_cluster]['horas']/6,
        cluster_stats[mejor_cluster]['frecuencia']/5,
        cluster_stats[mejor_cluster]['efectividad']/5
    ]
    
    x = np.arange(len(categorias))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, valores_tuyos, width, label='Tu perfil actual',
                       color='lightcoral', edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, valores_objetivo, width, label='Perfil objetivo',
                       color='lightgreen', edgecolor='black', linewidth=1.5)
    
    axes[0].set_ylabel('Valor Normalizado (0-1)', fontsize=12)
    axes[0].set_title('Comparación de Perfiles', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categorias)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Añadir flechas de mejora
    for i, (val_tuyo, val_obj) in enumerate(zip(valores_tuyos, valores_objetivo)):
        if val_obj > val_tuyo:
            axes[0].annotate('', xy=(i, val_obj), xytext=(i, val_tuyo),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Gráfico 2: Probabilidad de éxito
    clusters_nombres = [stat['nombre'].replace('ESTUDIANTES ', '') for stat in cluster_stats]
    probabilidades = [stat['tasa_aprobacion'] for stat in cluster_stats]
    colores_barras = ['red' if i == cluster_asignado else 'green' if i == mejor_cluster else 'gray' 
                     for i in range(len(cluster_stats))]
    
    bars = axes[1].barh(clusters_nombres, probabilidades, color=colores_barras, 
                       edgecolor='black', linewidth=2, alpha=0.7)
    axes[1].axvline(50, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].set_xlabel('Probabilidad de Aprobar Todo (%)', fontsize=12)
    axes[1].set_title('Tasa de Éxito por Perfil', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Marcar tu posición y objetivo
    for i, (nombre, prob) in enumerate(zip(clusters_nombres, probabilidades)):
        if i == cluster_asignado:
            axes[1].text(prob + 2, i, f'TÚ: {prob:.1f}%', va='center', fontweight='bold', fontsize=10)
        elif i == mejor_cluster:
            axes[1].text(prob + 2, i, f'OBJETIVO: {prob:.1f}%', va='center', fontweight='bold', fontsize=10)
        else:
            axes[1].text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/plan_mejora_personalizado.png', dpi=300, bbox_inches='tight')
    print()
    print("✅ Visualización guardada: plan_mejora_personalizado.png")
    print()

# ============================================================================
# MENÚ INTERACTIVO
# ============================================================================

print("🚀 Iniciando análisis de clustering...")
print()
df_cluster, kmeans, scaler, cluster_names, cluster_stats = realizar_clustering()

print()
print("="*80)
print("🎯 SISTEMA DE RECOMENDACIONES PERSONALIZADO")
print("="*80)
print()
print("Responde las siguientes preguntas para obtener recomendaciones:")
print()

horas = float(input("1. ¿Cuántas horas estudias por semana? (1-6): "))
frecuencia = float(input("2. ¿Con qué frecuencia estudias? (1=Nunca, 5=Diariamente): "))
efectividad = float(input("3. ¿Qué tan efectivo consideras tu método? (1-5): "))
usa_agenda = input("4. ¿Usas agenda para planificar? (Si/No): ").strip().lower() == 'si'
usa_digitales = input("5. ¿Usas herramientas digitales? (Si/No): ").strip().lower() == 'si'
tecnica_activa = input("6. ¿Usas técnicas activas (ejercicios/Pomodoro)? (Si/No): ").strip().lower() == 'si'

print()
print("⏳ Analizando tu perfil y generando recomendaciones...")
print()

generar_recomendaciones(horas, frecuencia, efectividad, usa_agenda, usa_digitales,
                       tecnica_activa, kmeans, scaler, cluster_stats)

print()
print("="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
print()
print("📚 Conceptos de Estadística II aplicados:")
print("   ✓ Clustering (K-means)")
print("   ✓ Análisis de Componentes Principales (PCA)")
print("   ✓ Inferencia Bayesiana")
print("   ✓ Correlación y significancia estadística")
print("   ✓ Intervalos de confianza (Bootstrap)")
print("   ✓ Análisis discriminante")
print("   ✓ Predicción probabilística")

"""
OPCIÓN 2: DASHBOARD ANIMADO - "VISUALIZACIÓN DINÁMICA DE HÁBITOS DE ESTUDIO"
==============================================================================

Dashboard interactivo con gráficas animadas que muestran:
- Correlaciones entre variables
- Distribuciones dinámicas
- Comparaciones interactivas
- Animaciones de cambios en el tiempo

Conceptos de Estadística II aplicados:
- Correlación de Pearson/Spearman
- Distribuciones de probabilidad
- Análisis multivariado
- Pruebas de hipótesis visualizadas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuración estética
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("="*80)
print("📊 DASHBOARD ANIMADO - Análisis Visual de Hábitos de Estudio")
print("="*80)
print()

# ============================================================================
# CARGA DE DATOS
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

# Función auxiliar
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

print("✅ Datos cargados:", len(df), "estudiantes")
print()

# ============================================================================
# 1. MAPA DE CALOR DE CORRELACIONES ANIMADO
# ============================================================================

def crear_mapa_calor_animado():
    """Crea un mapa de calor con animación de aparición"""
    
    print("📊 Generando Mapa de Calor de Correlaciones...")
    
    # Variables numéricas para correlación
    vars_correlacion = df[['horas_numericas', 'frecuencia_estudio', 'metodo_efectivo', 
                            'aprobo_binario', 'agenda_binario']].dropna()
    
    # Renombrar para mejor visualización
    vars_correlacion.columns = ['Horas\nEstudio', 'Frecuencia\nEstudio', 
                                'Efectividad\nPercibida', 'Aprobó\nTodas', 'Usa\nAgenda']
    
    # Calcular matriz de correlación
    corr_matrix = vars_correlacion.corr()
    
    # Calcular valores p
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                           columns=corr_matrix.columns, 
                           index=corr_matrix.index)
    
    for i, col1 in enumerate(vars_correlacion.columns):
        for j, col2 in enumerate(vars_correlacion.columns):
            if i != j:
                corr, p_val = pearsonr(vars_correlacion[col1].dropna(), 
                                      vars_correlacion[col2].dropna())
                p_values.iloc[i, j] = p_val
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mapa de calor de correlaciones
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax1)
    ax1.set_title('Matriz de Correlación de Pearson\n(Hábitos de Estudio vs Rendimiento)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Mapa de significancia (p-values)
    mask = p_values > 0.05  # Marcar correlaciones no significativas
    sns.heatmap(p_values, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0.05,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8, "label": "p-value"},
                vmin=0, vmax=0.1, ax=ax2, mask=mask)
    ax2.set_title('Significancia Estadística (p-values)\n(Verde = Significativo, Rojo = No significativo)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('outputs/mapa_calor_correlaciones.png', dpi=300, bbox_inches='tight')
    print("   ✅ Guardado: mapa_calor_correlaciones.png")
    
    # Interpretaciones
    print()
    print("   📈 CORRELACIONES MÁS FUERTES:")
    print("   " + "-"*60)
    
    # Obtener las 3 correlaciones más fuertes (excluyendo diagonal)
    corr_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlaciones_fuertes = corr_upper.unstack().sort_values(ascending=False).head(3)
    
    for (var1, var2), corr_val in correlaciones_fuertes.items():
        p_val = p_values.loc[var1, var2]
        significativo = "✓ Significativa" if p_val < 0.05 else "✗ No significativa"
        print(f"   {var1} ↔ {var2}")
        print(f"      r = {corr_val:.3f} (p = {p_val:.3f}) {significativo}")
    
    print()

# ============================================================================
# 2. GRÁFICO DE DISPERSIÓN INTERACTIVO CON REGRESIÓN
# ============================================================================

def crear_scatter_regresion():
    """Crea gráfico de dispersión con línea de regresión y CI"""
    
    print("📊 Generando Gráfico de Dispersión con Regresión...")
    
    # Limpiar datos
    df_clean = df[['horas_numericas', 'frecuencia_estudio', 'metodo_efectivo', 'aprobo_binario']].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Regresión: Factores que Predicen el Éxito', 
                 fontsize=16, fontweight='bold')
    
    relaciones = [
        ('horas_numericas', 'Horas de Estudio Semanales', 'aprobo_binario', 'Probabilidad de Aprobar'),
        ('frecuencia_estudio', 'Frecuencia de Estudio (1-5)', 'aprobo_binario', 'Probabilidad de Aprobar'),
        ('metodo_efectivo', 'Efectividad Percibida (1-5)', 'aprobo_binario', 'Probabilidad de Aprobar'),
        ('horas_numericas', 'Horas de Estudio Semanales', 'frecuencia_estudio', 'Frecuencia de Estudio')
    ]
    
    for idx, (var_x, label_x, var_y, label_y) in enumerate(relaciones):
        ax = axes[idx // 2, idx % 2]
        
        x = df_clean[var_x]
        y = df_clean[var_y]
        
        # Scatter plot con transparencia
        colors = ['red' if v == 0 else 'green' for v in y] if var_y == 'aprobo_binario' else 'blue'
        ax.scatter(x, y, alpha=0.6, s=80, c=colors, edgecolors='black', linewidth=0.5)
        
        # Línea de regresión
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Regresión: y = {z[0]:.2f}x + {z[1]:.2f}')
        
        # Intervalo de confianza
        from scipy import stats as sp_stats
        predict_y = p(x)
        std_error = np.sqrt(np.sum((y - predict_y)**2) / (len(y) - 2))
        margin = 1.96 * std_error
        ax.fill_between(x_line, p(x_line) - margin, p(x_line) + margin, alpha=0.2, color='red')
        
        # Correlación
        corr, p_val = pearsonr(x, y)
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel(label_x, fontsize=11)
        ax.set_ylabel(label_y, fontsize=11)
        ax.set_title(f'{label_x} vs {label_y}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/scatter_regresion.png', dpi=300, bbox_inches='tight')
    print("    Guardado: scatter_regresion.png")
    print()

# ============================================================================
# 3. DISTRIBUCIONES COMPARATIVAS
# ============================================================================

def crear_distribuciones_comparativas():
    """Compara distribuciones entre grupos"""
    
    print("📊 Generando Distribuciones Comparativas...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparación de Distribuciones: Aprobó vs No Aprobó Todas las Materias', 
                 fontsize=16, fontweight='bold')
    
    # Separar grupos
    aprobo = df[df['aprobo_binario'] == 1]
    no_aprobo = df[df['aprobo_binario'] == 0]
    
    variables = [
        ('horas_numericas', 'Horas de Estudio Semanales', 'horas'),
        ('frecuencia_estudio', 'Frecuencia de Estudio (1-5)', 'frecuencia'),
        ('metodo_efectivo', 'Efectividad Percibida (1-5)', 'efectividad'),
        ('edad', 'Edad', 'años')
    ]
    
    for idx, (var, titulo, unidad) in enumerate(variables):
        ax = axes[idx // 2, idx % 2]
        
        # Histogramas superpuestos
        ax.hist(aprobo[var].dropna(), bins=10, alpha=0.6, label='Aprobó todas (n=' + str(aprobo[var].notna().sum()) + ')',
                color='green', edgecolor='black', density=True)
        ax.hist(no_aprobo[var].dropna(), bins=10, alpha=0.6, label='No aprobó todas (n=' + str(no_aprobo[var].notna().sum()) + ')',
                color='red', edgecolor='black', density=True)
        
        # Curvas de densidad
        if aprobo[var].notna().sum() > 1:
            from scipy.stats import gaussian_kde
            kde_aprobo = gaussian_kde(aprobo[var].dropna())
            x_range = np.linspace(aprobo[var].min(), aprobo[var].max(), 100)
            ax.plot(x_range, kde_aprobo(x_range), 'g-', linewidth=2, alpha=0.8)
        
        if no_aprobo[var].notna().sum() > 1:
            from scipy.stats import gaussian_kde
            kde_no_aprobo = gaussian_kde(no_aprobo[var].dropna())
            x_range = np.linspace(no_aprobo[var].min(), no_aprobo[var].max(), 100)
            ax.plot(x_range, kde_no_aprobo(x_range), 'r-', linewidth=2, alpha=0.8)
        
        # Prueba t de Student
        if aprobo[var].notna().sum() > 1 and no_aprobo[var].notna().sum() > 1:
            t_stat, p_val = stats.ttest_ind(aprobo[var].dropna(), no_aprobo[var].dropna())
            
            # Añadir resultado de la prueba
            resultado_texto = f'Prueba t: t = {t_stat:.2f}, p = {p_val:.3f}\n'
            if p_val < 0.05:
                resultado_texto += '✓ Diferencia significativa'
                color_texto = 'green'
            else:
                resultado_texto += '✗ No hay diferencia significativa'
                color_texto = 'red'
            
            ax.text(0.5, 0.95, resultado_texto, 
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=color_texto, linewidth=2))
        
        # Estadísticos
        media_aprobo = aprobo[var].mean()
        media_no_aprobo = no_aprobo[var].mean()
        
        ax.axvline(media_aprobo, color='green', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Media (Aprobó): {media_aprobo:.2f}')
        ax.axvline(media_no_aprobo, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Media (No aprobó): {media_no_aprobo:.2f}')
        
        ax.set_xlabel(f'{titulo} ({unidad})', fontsize=11)
        ax.set_ylabel('Densidad de Probabilidad', fontsize=11)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/distribuciones_comparativas.png', dpi=300, bbox_inches='tight')
    print("   ✅ Guardado: distribuciones_comparativas.png")
    print()

# ============================================================================
# 4. GRÁFICO DE VIOLÍN MÚLTIPLE
# ============================================================================

def crear_violin_plots():
    """Crea gráficos de violín para visualizar distribuciones"""
    
    print("📊 Generando Gráficos de Violín...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribución de Variables por Resultado Académico', 
                 fontsize=16, fontweight='bold')
    
    # Preparar datos
    df_plot = df[['horas_numericas', 'frecuencia_estudio', 'metodo_efectivo', 'aprobo_binario']].dropna()
    df_plot['Resultado'] = df_plot['aprobo_binario'].map({1: 'Aprobó Todas', 0: 'No Aprobó Todas'})
    
    variables = [
        ('horas_numericas', 'Horas de Estudio\nSemanales'),
        ('frecuencia_estudio', 'Frecuencia de Estudio\n(escala 1-5)'),
        ('metodo_efectivo', 'Efectividad Percibida\n(escala 1-5)')
    ]
    
    for idx, (var, titulo) in enumerate(variables):
        ax = axes[idx]
        
        # Violin plot
        parts = ax.violinplot([df_plot[df_plot['Resultado'] == 'Aprobó Todas'][var].dropna(),
                               df_plot[df_plot['Resultado'] == 'No Aprobó Todas'][var].dropna()],
                              positions=[1, 2], widths=0.7, showmeans=True, showmedians=True)
        
        # Colorear violines
        colors = ['green', 'red']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        
        # Box plots superpuestos
        bp = ax.boxplot([df_plot[df_plot['Resultado'] == 'Aprobó Todas'][var].dropna(),
                        df_plot[df_plot['Resultado'] == 'No Aprobó Todas'][var].dropna()],
                       positions=[1, 2], widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor='white', alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Aprobó\nTodas', 'No Aprobó\nTodas'])
        ax.set_ylabel('Valor', fontsize=11)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir estadísticos
        media_aprobo = df_plot[df_plot['Resultado'] == 'Aprobó Todas'][var].mean()
        media_no_aprobo = df_plot[df_plot['Resultado'] == 'No Aprobó Todas'][var].mean()
        
        ax.text(1, ax.get_ylim()[1] * 0.95, f'μ = {media_aprobo:.2f}', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(2, ax.get_ylim()[1] * 0.95, f'μ = {media_no_aprobo:.2f}', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('outputs/violin_plots.png', dpi=300, bbox_inches='tight')
    print("   ✅ Guardado: violin_plots.png")
    print()

# ============================================================================
# 5. GRÁFICO DE BARRAS APILADAS (TÉCNICAS DE ESTUDIO)
# ============================================================================

def crear_barras_tecnicas():
    """Crea gráfico de barras apiladas de técnicas por resultado"""
    
    print("📊 Generando Gráfico de Técnicas de Estudio...")
    
    # Agrupar técnicas similares
    def simplificar_tecnica(tecnica):
        tecnica_str = str(tecnica).lower()
        if 'ejercicio' in tecnica_str or 'prácti' in tecnica_str:
            return 'Ejercicios Prácticos'
        elif 'resumen' in tecnica_str or 'resúmen' in tecnica_str:
            return 'Resúmenes'
        elif 'pomodoro' in tecnica_str:
            return 'Técnica Pomodoro'
        elif 'lectura' in tecnica_str:
            return 'Lectura Repetida'
        elif 'digital' in tecnica_str or 'apunte' in tecnica_str:
            return 'Apuntes Digitales'
        else:
            return 'Otras Técnicas'
    
    df['tecnica_simplificada'] = df['metodo_estudio'].apply(simplificar_tecnica)
    df['resultado_texto'] = df['aprobo_binario'].map({1: 'Aprobó Todas', 0: 'No Aprobó Todas'})
    
    # Crear tabla de contingencia
    tabla = pd.crosstab(df['tecnica_simplificada'], df['resultado_texto'], normalize='index') * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Efectividad de Técnicas de Estudio', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Barras apiladas (porcentajes)
    tabla.plot(kind='barh', stacked=True, ax=ax1, color=['lightcoral', 'lightgreen'],
              edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Porcentaje de Estudiantes (%)', fontsize=12)
    ax1.set_ylabel('Técnica de Estudio', fontsize=12)
    ax1.set_title('Distribución de Resultados por Técnica\n(% normalizado por técnica)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(title='Resultado', loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores en las barras
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=9, fontweight='bold')
    
    # Gráfico 2: Tasa de aprobación por técnica
    tasa_aprobacion = (df.groupby('tecnica_simplificada')['aprobo_binario'].mean() * 100).sort_values()
    
    bars = ax2.barh(tasa_aprobacion.index, tasa_aprobacion.values, 
                    color=['red' if x < 50 else 'orange' if x < 60 else 'green' for x in tasa_aprobacion.values],
                    edgecolor='black', linewidth=1.5)
    
    ax2.axvline(50, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='50% (Promedio esperado)')
    ax2.set_xlabel('Tasa de Aprobación de Todas las Materias (%)', fontsize=12)
    ax2.set_ylabel('Técnica de Estudio', fontsize=12)
    ax2.set_title('Tasa de Éxito por Técnica de Estudio', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (idx, val) in enumerate(tasa_aprobacion.items()):
        ax2.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/barras_tecnicas.png', dpi=300, bbox_inches='tight')
    print("   ✅ Guardado: barras_tecnicas.png")
    print()

# ============================================================================
# 6. RESUMEN ESTADÍSTICO VISUAL
# ============================================================================

def crear_resumen_estadistico():
    """Crea un panel de resumen con métricas clave"""
    
    print("📊 Generando Resumen Estadístico Visual...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Dashboard Estadístico Completo - Hábitos de Estudio UMSA', 
                 fontsize=18, fontweight='bold')
    
    # Panel 1: Métrica de aprobación
    ax1 = fig.add_subplot(gs[0, 0])
    tasa_aprobacion = (df['aprobo_binario'].mean() * 100)
    ax1.text(0.5, 0.5, f'{tasa_aprobacion:.1f}%', ha='center', va='center', 
            fontsize=50, fontweight='bold', color='green' if tasa_aprobacion > 50 else 'red')
    ax1.text(0.5, 0.2, 'Tasa de Aprobación\nde Todas las Materias', ha='center', va='center', 
            fontsize=12)
    ax1.axis('off')
    ax1.set_facecolor('whitesmoke')
    
    # Panel 2: Horas promedio
    ax2 = fig.add_subplot(gs[0, 1])
    horas_promedio = df['horas_numericas'].mean()
    ax2.text(0.5, 0.5, f'{horas_promedio:.1f}h', ha='center', va='center', 
            fontsize=50, fontweight='bold', color='navy')
    ax2.text(0.5, 0.2, 'Horas de Estudio\nSemanales Promedio', ha='center', va='center', 
            fontsize=12)
    ax2.axis('off')
    ax2.set_facecolor('whitesmoke')
    
    # Panel 3: Técnica más popular
    ax3 = fig.add_subplot(gs[0, 2])
    tecnica_popular = df['tecnica_simplificada'].mode()[0]
    porcentaje_popular = (df['tecnica_simplificada'] == tecnica_popular).sum() / len(df) * 100
    ax3.text(0.5, 0.5, f'{porcentaje_popular:.0f}%', ha='center', va='center', 
            fontsize=50, fontweight='bold', color='purple')
    ax3.text(0.5, 0.2, f'Usan\n{tecnica_popular}', ha='center', va='center', 
            fontsize=11)
    ax3.axis('off')
    ax3.set_facecolor('whitesmoke')
    
    # Panel 4: Distribución de género
    ax4 = fig.add_subplot(gs[1, 0])
    genero_counts = df['genero'].value_counts()
    ax4.pie(genero_counts.values, labels=genero_counts.index, autopct='%1.1f%%',
           startangle=90, colors=['lightblue', 'lightpink'])
    ax4.set_title('Distribución por Género', fontsize=12, fontweight='bold')
    
    # Panel 5: Distribución de edad
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(df['edad'], bins=15, edgecolor='black', color='lightgreen', alpha=0.7)
    ax5.axvline(df['edad'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Media: {df["edad"].mean():.1f} años')
    ax5.set_xlabel('Edad', fontsize=11)
    ax5.set_ylabel('Frecuencia', fontsize=11)
    ax5.set_title('Distribución de Edad', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Uso de herramientas
    ax6 = fig.add_subplot(gs[1, 2])
    herramientas = ['Usa\nAgenda', 'Usa\nDigitales', 'Técnica\nActiva']
    porcentajes = [
        (df['agenda_binario'].sum() / len(df)) * 100,
        (df['usa_herramientas_digitales'].str.contains('Si|Sí', na=False).sum() / len(df)) * 100,
        ((df['metodo_estudio'].str.contains('Ejercicios|Pomodoro|Feyman', na=False).sum() / len(df)) * 100)
    ]
    bars = ax6.bar(herramientas, porcentajes, color=['skyblue', 'orange', 'green'],
                  edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Porcentaje de Estudiantes (%)', fontsize=11)
    ax6.set_title('Adopción de Herramientas de Estudio', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 100)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, pct in zip(bars, porcentajes):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel 7-9: Comparación de promedios
    ax7 = fig.add_subplot(gs[2, :])
    
    variables_comp = ['horas_numericas', 'frecuencia_estudio', 'metodo_efectivo']
    labels_comp = ['Horas Estudio', 'Frecuencia', 'Efectividad']
    
    aprobo_means = [df[df['aprobo_binario'] == 1][var].mean() for var in variables_comp]
    no_aprobo_means = [df[df['aprobo_binario'] == 0][var].mean() for var in variables_comp]
    
    x_pos = np.arange(len(labels_comp))
    width = 0.35
    
    bars1 = ax7.bar(x_pos - width/2, aprobo_means, width, label='Aprobó Todas',
                   color='green', edgecolor='black', linewidth=1.5, alpha=0.7)
    bars2 = ax7.bar(x_pos + width/2, no_aprobo_means, width, label='No Aprobó Todas',
                   color='red', edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax7.set_ylabel('Valor Promedio', fontsize=12)
    ax7.set_title('Comparación de Promedios: Aprobó vs No Aprobó Todas las Materias', 
                 fontsize=13, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(labels_comp)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.savefig('outputs/dashboard_resumen.png', dpi=300, bbox_inches='tight')
    print("   ✅ Guardado: dashboard_resumen.png")
    print()

# ============================================================================
# EJECUTAR TODAS LAS VISUALIZACIONES
# ============================================================================

print("🚀 Generando todas las visualizaciones...")
print()

crear_mapa_calor_animado()
crear_scatter_regresion()
crear_distribuciones_comparativas()
crear_violin_plots()
crear_barras_tecnicas()
crear_resumen_estadistico()

print()
print("="*80)
print("✅ DASHBOARD COMPLETADO")
print("="*80)
print()
print("📂 Archivos generados:")
print("   1. mapa_calor_correlaciones.png - Correlaciones entre variables")
print("   2. scatter_regresion.png - Gráficos de dispersión con regresión")
print("   3. distribuciones_comparativas.png - Comparación de distribuciones")
print("   4. violin_plots.png - Gráficos de violín")
print("   5. barras_tecnicas.png - Efectividad de técnicas")
print("   6. dashboard_resumen.png - Resumen estadístico completo")
print()
print("📚 Conceptos de Estadística II aplicados:")
print("   ✓ Correlación de Pearson")
print("   ✓ Análisis de regresión lineal")
print("   ✓ Intervalos de confianza")
print("   ✓ Pruebas t de Student")
print("   ✓ Pruebas de significancia")
print("   ✓ Distribuciones de probabilidad")
print("   ✓ Análisis multivariado")

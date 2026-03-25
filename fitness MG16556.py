import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Datos comunes
concentraciones_antibiotico = [0, 0.5, 1, 2, 3, 5, 7,10] #Concentracion en µg/mL (0, 100, 200, 400, 600, 1000, 1400, 20000)

# Datos para cada día
days_data = {
    # Day 2
    2: {
        "IC0": {
            "replicas": [
                [0.693, 0.131, 0.143, 0.128, 0.124, 0.142, 0.103, 0.11],
                [0.69, 0.152, 0.143, 0.149, 0.149, 0.137, 0.131, 0.098],
                [0.684, 0.162, 0.149, 0.164, 0.132, 0.154, 0.147, 0.107]
            ]
        },
        "IC25": {
            "replicas": [
                [0.685, 0.136, 0.146, 0.157, 0.141, 0.14, 0.111, 0.109],
                [0.696, 0.149, 0.149, 0.154, 0.15, 0.118, 0.111, 0.112],
                [0.532, 0.135, 0.159, 0.154, 0.152, 0.113, 0.117, 0.116]
            ]
        },
        "IC50": {
            "replicas": [
                [0.509, 0.272, 0.255, 0.152, 0.163, 0.14, 0.124, 0.102],
                [0.818, 0.162, 0.239, 0.179, 0.146, 0.467, 0.113, 0.108],
                [0.68, 0.119, 0.121, 0.102, 0.117, 0.124, 0.115, 0.091]
            ]
        }
    },
    # Day 4
    4: {
        "IC0": {
            "replicas": [
                [0.662, 0.142, 0.138, 0.138, 0.12, 0.13, 0.116, 0.131],
                [0.775, 0.182, 0.163, 0.164, 0.145, 0.154, 0.141, 0.141],
                [0.768, 0.201, 0.154, 0.151, 0.167, 0.18, 0.12, 0.119]
            ]
        },
        "IC25": {
            "replicas": [
                [0.683, 0.161, 0.166, 0.184, 0.182, 0.146, 0.134, 0.115],
                [0.663, 0.16, 0.176, 0.165, 0.166, 0.151, 0.13, 0.133],
                [0.694, 0.178, 0.181, 0.193, 0.166, 0.143, 0.139, 0.114]
            ]
        },
        "IC50": {
            "replicas": [
                [0.643, 0.654, 0.678, 0.161, 0.141, 0.144, 0.105, 0.122],
                [0.643, 0.573, 0.669, 0.237, 0.173, 0.125, 0.132, 0.123],
                [0.64, 0.509, 0.615, 0.131, 0.107, 0.087, 0.112, 0.097]
            ]
        }
    },
    # Day 6
    6: {
        "IC0": {
            "replicas": [
                [0.617, 0.099, 0.115, 0.111, 0.092, 0.1, 0.106, 0.11],
                [0.636, 0.102, 0.12, 0.1, 0.09, 0.118, 0.117, 0.117],
                [0.59, 0.124, 0.123, 0.094, 0.109, 0.092, 0.115, 0.118]
            ]
        },
        "IC25": {
            "replicas": [
                [0.657, 0.342, 0.11, 0.1, 0.099, 0.106, 0.112, 0.098],
                [0.615, 0.317, 0.117, 0.118, 0.09, 0.106, 0.112, 0.092],
                [0.53, 0.398, 0.098, 0.113, 0.095, 0.117, 0.117, 0.112]
            ]
        },
        "IC50": {
            "replicas": [
                [0.499, 0.373, 0.452, 0.459, 0.117, 0.101, 0.113, 0.1],
                [0.468, 0.416, 0.465, 0.404, 0.105, 0.126, 0.117, 0.108],
                [0.531, 0.52, 0.436, 0.506, 0.138, 0.109, 0.119, 0.108]
            ]
        }
    },
    # Day 8
    8: {
        "IC0": {
            "replicas": [
                [0.506, 0.423, 0.122, 0.115, 0.095, 0.132, 0.115, 0.113],
                [0.51, 0.406, 0.137, 0.119, 0.099, 0.124, 0.117, 0.094],
                [0.503, 0.394, 0.117, 0.123, 0.115, 0.117, 0.116, 0.115]
            ]
        },
        "IC25": {
            "replicas": [
                [0.489, 0.451, 0.365, 0.386, 0.124, 0.127, 0.116, 0.113],
                [0.524, 0.458, 0.385, 0.11, 0.127, 0.108, 0.106, 0.106],
                [0.512, 0.504, 0.431, 0.435, 0.125, 0.128, 0.124, 0.099]
            ]
        },
        "IC50": {
            "replicas": [
                [0.482, 0.494, 0.474, 0.475, 0.482, 0.137, 0.096, 0.108],
                [0.518, 0.443, 0.524, 0.512, 0.453, 0.148, 0.126, 0.119],
                [0.598, 0.512, 0.542, 0.568, 0.546, 0.148, 0.127, 0.111]
            ]
        }
    },
    # Day 10
    10: {
        "IC0": {
            "replicas": [
                [0.59, 0.604, 0.631, 0.133, 0.128, 0.128, 0.116, 0.12],
                [0.594, 0.582, 0.535, 0.105, 0.1, 0.104, 0.103, 0.099],
                [0.594, 0.598, 0.509, 0.103, 0.129, 0.1, 0.102, 0.107]
            ]
        },
        "IC25": {
            "replicas": [
                [0.613, 0.589, 0.617, 0.458, 0.12, 0.136, 0.121, 0.117],
                [0.666, 0.589, 0.699, 0.456, 0.105, 0.136, 0.116, 0.139],
                [0.613, 0.713, 0.618, 0.474, 0.101, 0.126, 0.095, 0.111]
            ]
        },
        "IC50": {
            "replicas": [
                [0.608, 0.666, 0.632, 0.615, 0.556, 0.126, 0.268, 0.141],
                [0.581, 0.63, 0.613, 0.581, 0.566, 0.127, 0.173, 0.145],
                [0.427, 0.542, 0.54, 0.513, 0.54, 0.117, 0.143, 0.161]
            ]
        }
    }
}

# Procesar datos (sin cambios)
rows = []
for day, day_data in days_data.items():
    for conc_idx, conc in enumerate(concentraciones_antibiotico):
        for tipo in ["IC0", "IC25", "IC50"]:
            for replica in day_data[tipo]["replicas"]:
                rows.append({
                    "Dia": day,
                    "Concentracion": conc,
                    "Tipo": tipo,
                    "DO": replica[conc_idx]
                })
df = pd.DataFrame(rows)

# Función para calcular fitness (¡modificada!)
def calculate_fitness(df, tipo, day, threshold=0.300):
    """Calcula el fitness solo si la DO promedio >= threshold en al menos una concentración."""
    subset = df[(df['Tipo'] == tipo) & (df['Dia'] == day)]
    mean_do = subset.groupby('Concentracion')['DO'].mean().reset_index()
    mean_do = mean_do.sort_values('Concentracion')
    
    # Verificar si hay crecimiento significativo
    if mean_do['DO'].max() < threshold:
        return 0.0  # No hay crecimiento
    
    # Calcular AUC solo si hay crecimiento
    auc = np.trapz(mean_do['DO'], mean_do['Concentracion'])
    max_growth = mean_do[mean_do['Concentracion'] == 0]['DO'].values[0]
    normalized_auc = auc / (max_growth * max(mean_do['Concentracion']))
    
    return normalized_auc

# Calcular fitness para todas las condiciones
fitness_results = []
for day in days_data.keys():
    for tipo in ["IC0", "IC25", "IC50"]:
        fitness = calculate_fitness(df, tipo, day, threshold=0.300)
        fitness_results.append({
            'Dia': day,
            'Tipo': tipo,
            'Fitness': fitness
        })

fitness_df = pd.DataFrame(fitness_results)

# Filtrar datos con fitness > 0 para el gráfico
fitness_filtered = fitness_df[fitness_df['Fitness'] > 0]

# Configuración del gráfico
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Gráfico de evolución del fitness (solo datos con crecimiento)
plot = sns.lineplot(
    data=fitness_filtered,
    x='Dia',
    y='Fitness',
    hue='Tipo',
    style='Tipo',
    markers=True,
    dashes=False,
    markersize=10,
    linewidth=2.5
)

# Personalización
plt.title('Fitness de MG1655', fontsize=16, pad=20)
plt.xlabel('Tiempo (Días)', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.xticks(sorted(days_data.keys()))

# Mostrar tabla de resultados
print("\nTabla de Fitness (DO ≥ 0.250):")
print(fitness_df.pivot(index='Dia', columns='Tipo', values='Fitness').round(4))

plt.show()
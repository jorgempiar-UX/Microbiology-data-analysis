import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# Datos comunes
concentraciones_antibiotico = [0, 0.3, 0.6, 1, 1.3, 1.6, 2, 2.3] #Concentracion en µg/mL ()

# Datos para cada día
days_data = {
    # Day 2
    2: {
        "IC0": {
            "replicas": [
                [0.523, 0.515, 0.435, 0.433, 0.184, 0.151, 0.127, 0.118],
                [0.586, 0.537, 0.599, 0.463, 0.146, 0.112, 0.122, 0.106],
                [0.602, 0.56, 0.613, 0.45, 0.128, 0.11, 0.13, 0.099]
            ]
        },
        "IC25": {
            "replicas": [
                [0.595, 0.589, 0.526, 0.62, 0.166, 0.162, 0.161, 0.115],
                [0.587, 0.561, 0.596, 0.552, 0.175, 0.139, 0.162, 0.131],
                [0.671, 0.532, 0.574, 0.661, 0.171, 0.165, 0.157, 0.117]
            ]
        },
        "IC50": {
            "replicas": [
                [0.596, 0.64, 0.517, 0.649, 0.581, 0.569, 0.164, 0.127],
                [0.647, 0.6, 0.501, 0.629, 0.542, 0.592, 0.141, 0.126],
                [0.567, 0.617, 0.508, 0.598, 0.588, 0.546, 0.148, 0.153]
            ]
        }
    },
    # Day 4
    4: {
        "IC0": {
            "replicas": [
                [0.789, 0.684, 0.637, 0.112, 0.153, 0.192, 0.198, 0.135],
                [0.823, 0.778, 0.733, 0.508, 0.23, 0.113, 0.232, 0.198],
                [0.856, 0.8, 0.78, 0.601, 0.242, 0.109, 0.239, 0.217]
            ]
        },
        "IC25": {
            "replicas": [
                [0.957, 0.753, 0.682, 0.845, 0.272, 0.252, 0.208, 0.185],
                [0.993, 0.856, 0.998, 0.886, 0.305, 0.294, 0.244, 0.197],
                [0.971, 0.853, 0.493, 0.834, 0.306, 0.276, 0.208, 0.182]
            ]
        },
        "IC50": {
            "replicas": [
                [0.898, 0.784, 0.786, 0.721, 0.809, 0.776, 0.215, 0.208],
                [0.878, 0.77, 0.636, 0.771, 0.81, 0.771, 0.219, 0.223],
                [0.881, 0.762, 0.527, 0.711, 0.778, 0.773, 0.198, 0.182]
            ]
        }
    },
    # Day 6
    6: {
        "IC0": {
            "replicas": [
                [0.762, 0.692, 0.926, 0.654, 0.167, 0.124, 0.15, 0.134],
                [0.86, 0.655, 0.907, 0.652, 0.225, 0.203, 0.17, 0.146],
                [0.883, 0.713, 0.723, 0.726, 0.233, 0.116, 0.17, 0.154]
            ]
        },
        "IC25": {
            "replicas": [
                [0.9, 0.693, 0.677, 0.616, 0.213, 0.186, 0.184, 0.163],
                [0.913, 0.746, 0.674, 0.646, 0.201, 0.194, 0.152, 0.165],
                [0.824, 0.585, 0.537, 0.681, 0.203, 0.198, 0.174, 0.156]
            ]
        },
        "IC50": {
            "replicas": [
                [0.851, 0.802, 0.629, 0.908, 0.654, 0.66, 0.191, 0.18],
                [0.815, 0.576, 0.571, 1.001, 0.864, 0.758, 0.17, 0.146],
                [0.785, 0.615, 0.592, 0.644, 0.672, 0.575, 0.184, 0.18]
            ]
        }
    },
    # Day 8
    8: {
        "IC0": {
            "replicas": [
                [0.783, 0.7, 0.776, 0.726, 0.27, 0.12, 0.162, 0.134],
                [0.925, 0.744, 0.928, 0.792, 0.286, 0.18, 0.256, 0.167],
                [0.945, 0.787, 0.87, 0.624, 0.262, 0.113, 0.222, 0.164]
            ]
        },
        "IC25": {
            "replicas": [
                [0.914, 0.873, 0.87, 0.24, 0.259, 0.274, 0.239, 0.187],
                [0.765, 0.927, 0.782, 0.23, 0.295, 0.265, 0.261, 0.178],
                [0.846, 0.822, 0.848, 0.25, 0.244, 0.28, 0.25, 0.249]
            ]
        },
        "IC50": {
            "replicas": [
                [0.869, 0.816, 0.784, 0.816, 0.759, 0.518, 0.25, 0.178],
                [0.835, 0.76, 0.748, 0.823, 0.796, 0.521, 0.232, 0.194],
                [0.831, 0.727, 1.086, 0.815, 0.755, 0.558, 0.193, 0.171]
            ]
        }
    },
    # Day 10
    10: {
        "IC0": {
            "replicas": [
                [0.77, 0.503, 0.637, 0.515, 0.11, 0.122, 0.116, 0.129],
                [0.811, 0.505, 0.528, 0.302, 0.121, 0.113, 0.136, 0.152],
                [0.728, 0.512, 0.564, 0.476, 0.12, 0.202, 0.139, 0.166]
            ]
        },
        "IC25": {
            "replicas": [
                [0.765, 0.653, 0.572, 0.144, 0.135, 0.138, 0.135, 0.11],
                [0.831, 0.516, 0.541, 0.159, 0.112, 0.128, 0.108, 0.135],
                [0.696, 0.583, 0.6, 0.123, 0.133, 0.105, 0.109, 0.129]
            ]
        },
        "IC50": {
            "replicas": [
                [0.652, 0.502, 0.548, 0.686, 0.493, 0.537, 0.16, 0.169],
                [0.832, 0.578, 0.593, 0.806, 0.573, 0.605, 0.184, 0.158],
                [0.832, 0.656, 0.531, 0.646, 0.618, 0.756, 0.176, 0.143]
            ]
        }
    }
}

# Procesar datos
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

# Crear columna combinada
df["Dia-Tipo"] = df["Dia"].astype(str) + "-" + df["Tipo"]

# Generar paleta de colores
unique_combinations = sorted(df["Dia-Tipo"].unique())
palette = sns.husl_palette(len(unique_combinations))
palette_dict = dict(zip(unique_combinations, palette))

# Estilos de línea
line_styles = {"IC0": "", "IC25": (2, 2), "IC50": (4, 1, 1, 1)}


#FIGURA: Separada por tipo (IC0, IC25, IC50)
sns.set_style("whitegrid")


# Mapea las concentraciones a posiciones equidistantes
concentraciones_ordenadas = sorted(concentraciones_antibiotico)
posiciones_x = range(len(concentraciones_ordenadas))  # 0, 1, 2, ..., 7

# Crea un diccionario para reemplazar los valores en el DataFrame
mapeo_concentraciones = {conc: pos for conc, pos in zip(concentraciones_ordenadas, posiciones_x)}
df["Concentracion_pos"] = df["Concentracion"].map(mapeo_concentraciones)

plot = sns.relplot(
    data=df,
    x="Concentracion_pos",
    y="DO",
    hue="Dia",
    col="Tipo",
    kind="line",
    palette="husl",
    height=5,
    aspect=0.8,
    facet_kws={'sharey': True, 'sharex': True},
    errorbar=('ci', 95),
    estimator='mean',
    linewidth=2,
    markers=True,

    style="Dia",  # Diferentes estilos de marcador por día
    markersize=8,  # Tamaño de los marcadores
    dashes=False,  # Deshabilita líneas discontinuas 
)


axes = plot.axes.flat
for ax in axes:
    ax.set_xticks(posiciones_x)  
    ax.set_xticklabels(concentraciones_ordenadas, rotation=45)  
    ax.set_xlabel("Concentración de Antibiótico")  

# Configuración de la segunda figura
plot.set_titles("{col_name}")
plot.fig.suptitle("Curvas Dosis-Respuesta a IC0, IC25 e IC50 para BLATEM-1", y=0.99, fontsize=16)
plot.set_axis_labels("Concentración de Antibiótico (MIC)", "Densidad Óptica (DO)")


axes = plot.axes.flat  # Lista de todos los subplots
for i, ax in enumerate(axes):
    if i == len(axes) - 1:  #
        ax.legend(
            title="Día",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=8,
            frameon=True
        )
    else:
        if ax.get_legend() is not None:  
            ax.get_legend().remove()  
plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.show()

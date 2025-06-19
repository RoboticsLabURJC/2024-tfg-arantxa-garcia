import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Cargar datos
with open("prediction_timeline.json", "r") as f:
    data = json.load(f)

# Categorías y etiquetas
categories = ["actions", "gaze", "phone"]
category_labels = {"actions": "Acción", "gaze": "Mirada", "phone": "Móvil"}

# Función para interpretar valores de 'phone'
def interpret_phone(value):
    if value == 1:
        return "Using phone"
    return None

# Eventos por categoría y etiqueta
events = {cat: {} for cat in categories}
current = {cat: None for cat in categories}
start_time = {cat: None for cat in categories}

for entry in data:
    timestamp = entry["timestamp"]
    for cat in categories:
        value = entry.get(cat)
        if cat == "phone":
            value = interpret_phone(value)
            # Para phone, tratamos el valor como antes
            if value != current[cat]:
                if current[cat] is not None:
                    label = current[cat]
                    if label not in events[cat]:
                        events[cat][label] = []
                    events[cat][label].append((start_time[cat], timestamp))
                current[cat] = value
                start_time[cat] = timestamp
        else:
            # Para actions y gaze, manejamos listas de valores
            current_values = set(value) if value else set()
            previous_values = set(current[cat]) if current[cat] is not None else set()
            
            # Si hay cambios en los valores
            if current_values != previous_values:
                # Finalizar los intervalos anteriores
                if current[cat] is not None:
                    for label in previous_values:
                        if label not in events[cat]:
                            events[cat][label] = []
                        events[cat][label].append((start_time[cat], timestamp))
                
                # Iniciar nuevos intervalos
                current[cat] = list(current_values) if current_values else None
                start_time[cat] = timestamp

# Añadir últimos bloques
for cat in categories:
    if cat == "phone":
        label = current[cat]
        if label is not None and start_time[cat] != data[-1]["timestamp"]:
            if label not in events[cat]:
                events[cat][label] = []
            events[cat][label].append((start_time[cat], data[-1]["timestamp"]))
    else:
        if current[cat] is not None:
            for label in current[cat]:
                if label not in events[cat]:
                    events[cat][label] = []
                events[cat][label].append((start_time[cat], data[-1]["timestamp"]))

# Asegurar que la categoría 'phone' tenga la etiqueta "Using phone" incluso sin datos
if "Using phone" not in events["phone"]:
    events["phone"]["Using phone"] = []

# Crear gráfico
fig, ax = plt.subplots(figsize=(16, 10))  # Vertical

yticks = []
yticklabels = []
colors = {}
color_palette = plt.cm.tab20.colors
color_idx = 0
y = 0
separation = 1  # Espacio entre categorías

for cat in categories:
    if y > 0:
        ax.axhline(y - 0.5, color='gray', linestyle='dotted')  # Línea discontinua
    
    # Mostrar todas las etiquetas definidas, incluso si no tienen intervalos
    for label in events[cat].keys():
        if label not in colors:
            colors[label] = color_palette[color_idx % len(color_palette)]
            color_idx += 1
        
        # Dibujar los intervalos si existen
        intervals = events[cat][label]
        for start, end in intervals:
            ax.barh(y, end - start, left=start, height=0.8, color=colors[label])
        
        # Añadir la etiqueta al gráfico aunque no tenga intervalos
        yticks.append(y)
        yticklabels.append(label)
        y += 1
    
    y += separation  # Separación entre categorías

# Estética
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlabel("Tiempo (s)")
ax.set_title("Análisis temporal por etiqueta")
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

# Leyenda
legend_patches = [mpatches.Patch(color=col, label=lbl) for lbl, col in colors.items()]
ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Etiquetas")

plt.tight_layout()
plt.show()
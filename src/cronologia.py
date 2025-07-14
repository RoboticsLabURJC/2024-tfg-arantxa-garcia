import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("prediction_timeline.json", "r") as f:
    data = json.load(f)

categories = ["actions", "gaze", "phone"]

def interpret_phone(value):
    if value == 1:
        return "Using phone"
    return None

events = {cat: {} for cat in categories}
current = {cat: None for cat in categories}
start_time = {cat: None for cat in categories}

for entry in data:
    timestamp = entry["timestamp"]
    for cat in categories:
        value = entry.get(cat)
        if cat == "phone":
            value = interpret_phone(value)
            if value != current[cat]:
                if current[cat] is not None:
                    label = current[cat]
                    if label not in events[cat]:
                        events[cat][label] = []
                    events[cat][label].append((start_time[cat], timestamp))
                current[cat] = value
                start_time[cat] = timestamp
        else:
            current_values = set(value) if value else set()
            previous_values = set(current[cat]) if current[cat] is not None else set()
            if current_values != previous_values:
                if current[cat] is not None:
                    for label in previous_values:
                        if label not in events[cat]:
                            events[cat][label] = []
                        events[cat][label].append((start_time[cat], timestamp))
                current[cat] = list(current_values) if current_values else None
                start_time[cat] = timestamp

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

if "Using phone" not in events["phone"]:
    events["phone"]["Using phone"] = []

fig, ax = plt.subplots(figsize=(16, 10))

yticks = []
yticklabels = []
colors = {}
color_palette = plt.cm.tab20.colors
color_idx = 0
y = 0
separation = 1

for cat in categories:
    if y > 0:
        ax.axhline(y - 0.5, color='gray', linestyle='dotted')
    for label in events[cat].keys():
        if label not in colors:
            colors[label] = color_palette[color_idx % len(color_palette)]
            color_idx += 1
        intervals = events[cat][label]
        for start, end in intervals:
            ax.barh(y, end - start, left=start, height=0.8, color=colors[label])
        yticks.append(y)
        yticklabels.append(label)
        y += 1
    y += separation

alert_start_time = None

for i, entry in enumerate(data):
    t = entry["timestamp"]
    actions = entry.get("actions", [])
    phone = interpret_phone(entry.get("phone", None))

    # Colores suaves
    green_color = (0.7, 1, 0.7, 0.5)      # light green
    yellow_color = (1, 1, 0.6, 0.3)     # light yellow
    red_color = (1, 0.6, 0.6, 0.2)       # light red
    phone_color = (1, 0.4, 0.4, 0.3)      # light red for phone

    fondo_color = green_color

    if phone == "Using phone":
        fondo_color = phone_color
        alert_start_time = None 
    else:
        found_driver = any(a.startswith("driver_action") for a in actions)
        found_hands = any(a in ["hands_on_wheel/only_left", "hands_on_wheel/only_right"] for a in actions)

        if found_driver or found_hands:
            if alert_start_time is None:
                alert_start_time = t  
            elapsed_time = t - alert_start_time

            if elapsed_time < 3:
                fondo_color = yellow_color
            else:
                fondo_color = red_color
        else:
            alert_start_time = None
            fondo_color = green_color

    if i < len(data) - 1:
        next_t = data[i + 1]["timestamp"]
    else:
        next_t = data[-1]["timestamp"] + 1

    ax.axvspan(t, next_t, facecolor=fondo_color, zorder=-1)

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlabel("Tiempo (s)")
ax.set_title("Análisis temporal por etiqueta")
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

min_time = data[0]["timestamp"]
max_time = data[-1]["timestamp"]
ticks = list(range(int(min_time), int(max_time) + 10, 10))
ax.set_xticks(ticks)

legend_patches = [mpatches.Patch(color=col, label=lbl) for lbl, col in colors.items()]
ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Etiquetas")

plt.tight_layout()
plt.show()

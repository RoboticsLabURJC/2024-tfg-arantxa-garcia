"""

Este script se encarga de entrenar un modelo RandomForest con los datos balanceados y de reconstruir
los videos prediciendo las acciones de cada frame. En este script luego predice sobre los datos
de una sesion con datos desbalanceados.

NO NECESITA ARGUMENTOS

"""

import json
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from datetime import datetime
import random
import math

def tf(to_tf, original, axis):
    return to_tf[axis] - original[axis]

def calc_ang(p1, p2, center):
    u, v = np.array(p1) - np.array(center), np.array(p2) - np.array(center)
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clip to avoid domain errors

def convert_to_multilabel(df):
    label_map = {
        "hands_using_wheel/both": "both_hands",
        "hands_using_wheel/only_left": "left_hand",
        "hands_using_wheel/only_right": "right_hand",
        "driver_actions/radio": "radio",
        "driver_actions/drinking": "drinking",
        "driver_actions/reach_side": "reach_side"
    }
    df['label'] = df['label'].map(label_map)
    multilabels = pd.get_dummies(df['label'])
    
    # Asegurar que todas las columnas posibles existan (como valores numéricos)
    for label in label_map.values():
        if label not in multilabels:
            multilabels[label] = 0
        else:
            multilabels[label] = multilabels[label].astype(int)  # Convertir a int
    
    # Ordenar columnas para consistencia
    multilabels = multilabels[list(label_map.values())]
    
    # Cuando alguna de las últimas tres acciones es 1, establecer left_hand a 1
    last_three = ['radio', 'drinking', 'reach_side']
    mask = multilabels[last_three].any(axis=1)
    multilabels['left_hand'] = np.where(mask, 1, multilabels['left_hand'])
    
    return pd.concat([df.drop(columns='label'), multilabels], axis=1)
def truncate(f, n):
    return np.floor(f * 10**n) / 10**n

def compute_multilabel_confusion_matrix(Y_true, Y_pred, class_labels, set_name="train", threshold=0.5):
    """
    Versión simplificada para combinaciones específicas entre left_hand y otras acciones
    """
    # Convertir a arrays numpy
    y_true = Y_true.values if hasattr(Y_true, 'values') else Y_true
    y_pred = (Y_pred >= threshold).astype(int)
    
    # Definir combinaciones relevantes (left_hand con drinking, reach_side o radio)
    relevant_combinations = [
        [1, 0, 0, 0, 0, 0],  # both_hands solo
        [0, 1, 0, 0, 0, 0],  # left_hand solo
        [0, 1, 0, 1, 0, 0],  # left_hand + radio
        [0, 1, 0, 0, 1, 0],  # left_hand + drinking
        [0, 1, 0, 0, 0, 1],  # left_hand + reach_side
        [0, 0, 1, 0, 0, 0]   # right_hand solo
    ]
    
    # Nombres legibles para las combinaciones
    combo_names = {
        (1, 0, 0, 0, 0, 0): "both_hands",
        (0, 1, 0, 0, 0, 0): "left_hand",
        (0, 1, 0, 1, 0, 0): "left_hand + radio",
        (0, 1, 0, 0, 1, 0): "left_hand + drinking",
        (0, 1, 0, 0, 0, 1): "left_hand + reach_side",
        (0, 0, 1, 0, 0, 0): "right_hand"
    }
    
    # Inicializar matriz
    size = len(relevant_combinations)
    conf_matrix = np.zeros((size, size), dtype=int)
    
    # Mapeo de combinaciones a índices
    combo_to_idx = {tuple(combo): idx for idx, combo in enumerate(relevant_combinations)}
    
    # Llenar matriz
    for true_row, pred_row in zip(y_true, y_pred):
        true_combo = tuple(true_row)
        pred_combo = tuple(pred_row)
        
        # Solo considerar combinaciones relevantes
        if true_combo in combo_to_idx and pred_combo in combo_to_idx:
            i = combo_to_idx[true_combo]
            j = combo_to_idx[pred_combo]
            conf_matrix[i, j] += 1
    
    # Convertir a DataFrame con nombres legibles
    index_labels = [combo_names[tuple(combo)] for combo in relevant_combinations]
    conf_matrix = pd.DataFrame(conf_matrix, index=index_labels, columns=index_labels)
    
    # Normalizar por filas
    norm_conf = conf_matrix.div(conf_matrix.sum(axis=1), axis=0).fillna(0)
    
    # Visualización
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_conf, annot=True, fmt=".2f", cmap="Blues", 
                cbar_kws={'label': 'Porcentaje'})
    plt.title(f"Matriz de Confusión - {set_name.capitalize()}")
    plt.xlabel("Predicciones")
    plt.ylabel("Reales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return conf_matrix

def compute_confusion_matrix(Y_true, Y_pred, class_labels, set_name="train", threshold=0.5):
    preds_bin = (Y_pred >= threshold).astype(int)
    num_classes = len(class_labels)
    conf_matrix = np.zeros((num_classes, num_classes))

    no_preds = yes_preds = 0
    for true, pred in zip(Y_true.values, preds_bin):
        true_idx = np.where(true == 1)[0]
        pred_idx = np.where(pred == 1)[0]

        if pred_idx.size == 0:
            no_preds += 1
        else:
            yes_preds += 1
            for t in true_idx:
                for p in pred_idx:
                    conf_matrix[t, p] += 1

    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_conf = np.divide(conf_matrix, row_sums, where=row_sums != 0)
    norm_conf = truncate(norm_conf.T, 2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(norm_conf, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Etiqueta Real")
    plt.ylabel("Predicción")
    plt.title(f"Matriz de Confusión - {set_name.capitalize()}")
    fname = f"{set_name}_confusion_matrix_{datetime.now():%Y-%m-%d_%H-%M-%S}.png"
    plt.savefig(fname)
    plt.show()

    correct = np.diag(conf_matrix)
    per_class_total = conf_matrix.sum(axis=1)
    accuracy = correct / per_class_total * 100

    print(f"Porcentajes de aciertos por clase: {accuracy}")

    plt.figure(figsize=(8, 5))
    plt.bar(class_labels, accuracy, color='blue')
    plt.title(f"Aciertos por clase ({set_name})")
    plt.ylabel("Porcentaje")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fname.replace("matrix", "bar"))
    plt.show()

    print(f"Ejemplos sin predicciones en {set_name}: {no_preds}")
    print(f"Ejemplos con predicciones en {set_name}: {yes_preds}")


class RandomForest:
    def __init__(self, json_both, json_onlyleft, json_onlyright, json_radio, json_drinking, json_reachside):
        self.both_j = json_both
        self.onlyleft_j = json_onlyleft
        self.onlyright_j = json_onlyright
        self.radio_j = json_radio
        self.drinking_j = json_drinking
        self.reachside_j = json_reachside

        self.both = []
        self.onlyleft = []
        self.onlyright = []
        self.radio = []
        self.drinking = []
        self.reachside = []

        self.rows = []

    # Función para cargar un archivo JSON con manejo de codificación
    def load_json(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {file_path}. Trying alternative encoding...")
            with open(file_path, 'r', encoding='latin-1') as f:
                return json.load(f)

    def open_jsons(self):

        # Usar la función para cargar los archivos
        self.both = self.load_json(self.both_j)
        self.onlyleft = self.load_json(self.onlyleft_j)
        self.onlyright = self.load_json(self.onlyright_j)
        self.radio = self.load_json(self.radio_j)
        self.drinking = self.load_json(self.drinking_j)
        self.reachside = self.load_json(self.reachside_j)

    def generate_random_number(self, prin_num, rang_low=0.015, rang_high=0.015):
        return random.uniform(prin_num - rang_low, prin_num + rang_high)

    def add_gaussian_noise(self, value, mean=0, std=0.01):
        return value + np.random.normal(mean, std)
    
    def generate_random_angle(self):
        angle_range = 7
        return math.radians(random.uniform(-angle_range, angle_range))

    def rotate_point(self, x, y, cx, cy, angle_rad, flag):
        x -= cx
        y -= cy

        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

        if flag == 0:
            return new_x + cx
        elif flag == 1:
            return new_y + cy

    def prepare_data(self, data, add_noise):

        for item in data:

            right_elbow = item['pose']['pose'][1]
            cx, cy, cz = right_elbow

            rotation = self.generate_random_angle()

            features = {
                "center_left_x": tf(item['pose']['pose'][50], right_elbow, 0),
                "center_left_y": tf(item['pose']['pose'][50], right_elbow, 1),
                "center_right_x": tf(item['pose']['pose'][51], right_elbow, 0),
                "center_right_y": tf(item['pose']['pose'][51], right_elbow, 1),
                "pose_0_x": tf(item['pose']['pose'][0], right_elbow, 0),
                "pose_0_y": tf(item['pose']['pose'][0], right_elbow, 1),
                "pose_1_x": tf(item['pose']['pose'][1], right_elbow, 0),
                "pose_1_y": tf(item['pose']['pose'][1], right_elbow, 1),
                "pose_2_x": tf(item['pose']['pose'][2], right_elbow, 0),
                "pose_2_y": tf(item['pose']['pose'][2], right_elbow, 1),
                "pose_3_x": tf(item['pose']['pose'][3], right_elbow, 0),
                "pose_3_y": tf(item['pose']['pose'][3], right_elbow, 1),
                "pose_4_x": tf(item['pose']['pose'][4], right_elbow, 0),
                "pose_4_y": tf(item['pose']['pose'][4], right_elbow, 1),
                "pose_5_x": tf(item['pose']['pose'][5], right_elbow, 0),
                "pose_5_y": tf(item['pose']['pose'][5], right_elbow, 1),
                "elbow_right": calc_ang(item['pose']['pose'][1], item['pose']['pose'][5], item['pose']['pose'][3]),
                "elbow_left": calc_ang(item['pose']['pose'][0], item['pose']['pose'][4], item['pose']['pose'][2]),
                "wrist_right": calc_ang(item['pose']['pose'][3], item['pose']['pose'][51], item['pose']['pose'][5]),
                "wrist_left": calc_ang(item['pose']['pose'][2], item['pose']['pose'][50], item['pose']['pose'][4]),
                
                "label": item['type'] # cambiar el label
            }

            features_tras = {
                "center_left_x": self.rotate_point(tf(item['pose']['pose'][50], right_elbow, 0), tf(item['pose']['pose'][50], right_elbow, 1), cx, cy, rotation, 0),
                "center_left_y": self.rotate_point(tf(item['pose']['pose'][50], right_elbow, 0), tf(item['pose']['pose'][50], right_elbow, 1), cx, cy, rotation, 1),
                "center_right_x": self.rotate_point(tf(item['pose']['pose'][51], right_elbow, 0), tf(item['pose']['pose'][51], right_elbow, 1), cx, cy, rotation, 0),
                "center_right_y": self.rotate_point(tf(item['pose']['pose'][51], right_elbow, 0), tf(item['pose']['pose'][51], right_elbow, 1), cx, cy, rotation, 1),
                "pose_0_x": self.rotate_point(tf(item['pose']['pose'][0], right_elbow, 0), tf(item['pose']['pose'][0], right_elbow, 1), cx, cy, rotation, 0),
                "pose_0_y": self.rotate_point(tf(item['pose']['pose'][0], right_elbow, 0), tf(item['pose']['pose'][0], right_elbow, 1), cx, cy, rotation, 1),
                "pose_1_x": self.rotate_point(tf(item['pose']['pose'][1], right_elbow, 0), tf(item['pose']['pose'][1], right_elbow, 1), cx, cy, rotation, 0),
                "pose_1_y": self.rotate_point(tf(item['pose']['pose'][1], right_elbow, 0), tf(item['pose']['pose'][1], right_elbow, 1), cx, cy, rotation, 1),
                "pose_2_x": self.rotate_point(tf(item['pose']['pose'][2], right_elbow, 0), tf(item['pose']['pose'][2], right_elbow, 1), cx, cy, rotation, 0),
                "pose_2_y": self.rotate_point(tf(item['pose']['pose'][2], right_elbow, 0), tf(item['pose']['pose'][2], right_elbow, 1), cx, cy, rotation, 1),
                "pose_3_x": self.rotate_point(tf(item['pose']['pose'][3], right_elbow, 0), tf(item['pose']['pose'][3], right_elbow, 1), cx, cy, rotation, 0),
                "pose_3_y": self.rotate_point(tf(item['pose']['pose'][3], right_elbow, 0), tf(item['pose']['pose'][3], right_elbow, 1), cx, cy, rotation, 1),
                "pose_4_x": self.rotate_point(tf(item['pose']['pose'][4], right_elbow, 0), tf(item['pose']['pose'][4], right_elbow, 1), cx, cy, rotation, 0),
                "pose_4_y": self.rotate_point(tf(item['pose']['pose'][4], right_elbow, 0), tf(item['pose']['pose'][4], right_elbow, 1), cx, cy, rotation, 1),
                "pose_5_x": self.rotate_point(tf(item['pose']['pose'][5], right_elbow, 0), tf(item['pose']['pose'][5], right_elbow, 1), cx, cy, rotation, 0),
                "pose_5_y": self.rotate_point(tf(item['pose']['pose'][5], right_elbow, 0), tf(item['pose']['pose'][5], right_elbow, 1), cx, cy, rotation, 1),
                "elbow_right": calc_ang(item['pose']['pose'][1], item['pose']['pose'][5], item['pose']['pose'][3]),
                "elbow_left": calc_ang(item['pose']['pose'][0], item['pose']['pose'][4], item['pose']['pose'][2]),
                "wrist_right": calc_ang(item['pose']['pose'][3], item['pose']['pose'][51], item['pose']['pose'][5]),
                "wrist_left": calc_ang(item['pose']['pose'][2], item['pose']['pose'][50], item['pose']['pose'][4]),
               
                "label": item['type'] # cambiar el label
            }

            features_gauss = {
                "center_left_x": self.add_gaussian_noise(tf(item['pose']['pose'][50], right_elbow, 0)),
                "center_left_y": self.add_gaussian_noise(tf(item['pose']['pose'][50], right_elbow, 1)),
                "center_right_x": self.add_gaussian_noise(tf(item['pose']['pose'][51], right_elbow, 0)),
                "center_right_y": self.add_gaussian_noise(tf(item['pose']['pose'][51], right_elbow, 1)),
                "pose_0_x": self.add_gaussian_noise(tf(item['pose']['pose'][0], right_elbow, 0)),
                "pose_0_y": self.add_gaussian_noise(tf(item['pose']['pose'][0], right_elbow, 1)),
                "pose_1_x": self.add_gaussian_noise(tf(item['pose']['pose'][1], right_elbow, 0)),
                "pose_1_y": self.add_gaussian_noise(tf(item['pose']['pose'][1], right_elbow, 1)),
                "pose_2_x": self.add_gaussian_noise(tf(item['pose']['pose'][2], right_elbow, 0)),
                "pose_2_y": self.add_gaussian_noise(tf(item['pose']['pose'][2], right_elbow, 1)),
                "pose_3_x": self.add_gaussian_noise(tf(item['pose']['pose'][3], right_elbow, 0)),
                "pose_3_y": self.add_gaussian_noise(tf(item['pose']['pose'][3], right_elbow, 1)),
                "pose_4_x": self.add_gaussian_noise(tf(item['pose']['pose'][4], right_elbow, 0)),
                "pose_4_y": self.add_gaussian_noise(tf(item['pose']['pose'][4], right_elbow, 1)),
                "pose_5_x": self.add_gaussian_noise(tf(item['pose']['pose'][5], right_elbow, 0)),
                "pose_5_y": self.add_gaussian_noise(tf(item['pose']['pose'][5], right_elbow, 1)),
                "elbow_right": self.add_gaussian_noise(calc_ang(item['pose']['pose'][1], item['pose']['pose'][5], item['pose']['pose'][3])),
                "elbow_left": self.add_gaussian_noise(calc_ang(item['pose']['pose'][0], item['pose']['pose'][4], item['pose']['pose'][2])),
                "wrist_right": self.add_gaussian_noise(calc_ang(item['pose']['pose'][3], item['pose']['pose'][51], item['pose']['pose'][5])),
                "wrist_left": self.add_gaussian_noise(calc_ang(item['pose']['pose'][2], item['pose']['pose'][50], item['pose']['pose'][4])),
                
                "label": item['type'] # cambiar el label
            }
            
            self.rows.append(features)
            if add_noise:
                self.rows.append(features_tras)
                self.rows.append(features_gauss)


def train_actions_model(json_files_train, json_files_test):
    # Cargar datos de entrenamiento
    RandomForest_performer_train = RandomForest(*json_files_train)
    RandomForest_performer_train.open_jsons()
    for data in [RandomForest_performer_train.both, RandomForest_performer_train.onlyleft, RandomForest_performer_train.onlyright, 
                    RandomForest_performer_train.radio, RandomForest_performer_train.drinking, RandomForest_performer_train.reachside]:
        if data == RandomForest_performer_train.onlyleft:
            RandomForest_performer_train.prepare_data(data, add_noise=False)
        else:
            RandomForest_performer_train.prepare_data(data, add_noise=True)

    dataset_train = pd.DataFrame(RandomForest_performer_train.rows)
    dataset_train = convert_to_multilabel(dataset_train)

    X_train = dataset_train.iloc[:, :-6]  # todo menos etiquetas
    Y_train = dataset_train.iloc[:, -6:]  # etiquetas

    # Cargar datos de prueba
    RandomForest_performer_test = RandomForest(*json_files_test)
    RandomForest_performer_test.open_jsons()
    for data in [RandomForest_performer_test.both, RandomForest_performer_test.onlyleft, RandomForest_performer_test.onlyright, 
                    RandomForest_performer_test.radio, RandomForest_performer_test.drinking, RandomForest_performer_test.reachside]:
        if data == RandomForest_performer_test.onlyleft:
            RandomForest_performer_test.prepare_data(data, add_noise=False)
        else:
            RandomForest_performer_test.prepare_data(data, add_noise=True)

    dataset_test = pd.DataFrame(RandomForest_performer_test.rows)
    dataset_test = convert_to_multilabel(dataset_test)

    X_test = dataset_test.iloc[:, :-6]  # todo menos etiquetas
    Y_test = dataset_test.iloc[:, -6:]  # etiquetas

    # Imprimir distribución de clases
    train_class_counts = Y_train.sum(axis=0)
    test_class_counts = Y_test.sum(axis=0)
    
    print("Distribución de clases en el conjunto de entrenamiento:")
    print(train_class_counts)
    print("\nDistribución de clases en el conjunto de prueba:")
    print(test_class_counts)

    # Convertir etiquetas en valores binarios
    Y_train_bin = np.array(list(Y_train))
    Y_test_bin = np.array(list(Y_test))

    # Entrenamiento del modelo
    model = OneVsRestClassifier(RandomForestClassifier(random_state=1, class_weight='balanced', n_estimators=3))#, n_jobs=-1))
    model.fit(X_train, Y_train)


    # Predicciones
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Etiquetas de las clases
    class_labels = ['Both', 'Left', 'Right', 'Radio', 'Drinking', 'Reachside']

    # # 1. Calcular matriz de confusión para cada clase (sin normalizar)
    # mcm = multilabel_confusion_matrix(Y_test, test_preds)
    
    # # 2. Normalizar por filas (para ver porcentajes)
    # mcm_normalized = []
    # for i, matrix in enumerate(mcm):
    #     # Normalizar cada matriz dividiendo por la suma de la fila (axis=1)
    #     row_sums = matrix.sum(axis=1, keepdims=True)
    #     normalized_matrix = matrix / row_sums  # Evitar división por cero
    #     mcm_normalized.append(normalized_matrix)
    
    # # 3. Visualización (con porcentajes)
    # plt.figure(figsize=(20, 15))
    # for i, (matrix, label) in enumerate(zip(mcm_normalized, class_labels)):
    #     plt.subplot(2, 3, i+1)
    #     sns.heatmap(
    #         matrix, 
    #         annot=True, 
    #         fmt=".2%",  # Formato de porcentaje
    #         cmap='Blues', 
    #         vmin=0, 
    #         vmax=1,
    #         xticklabels=['No ' + label, label], 
    #         yticklabels=['No ' + label, label]
    #     )
    #     plt.title(f'Matriz de Confusión Normalizada ({label})')
    #     plt.ylabel('Real')
    #     plt.xlabel('Predicho')
    # plt.tight_layout()
    # plt.show()
    # # Calcular matriz de confusión
    compute_multilabel_confusion_matrix(Y_train, train_preds, class_labels=list(Y_train.columns), set_name="train")
    compute_multilabel_confusion_matrix(Y_test, test_preds, class_labels=list(Y_train.columns), set_name="test")
    
    # Devolver el modelo entrenado
    return model


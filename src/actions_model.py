"""
NO ARGUMENTS NEEDED
"""

import json
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import random
import math

def tf(to_tf, original, axis):
    return to_tf[axis] - original[axis]

def calc_ang(p1, p2, center):
    u, v = np.array(p1) - np.array(center), np.array(p2) - np.array(center)
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  

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
    
    for label in label_map.values():
        if label not in multilabels:
            multilabels[label] = 0
        else:
            multilabels[label] = multilabels[label].astype(int) 
    
    multilabels = multilabels[list(label_map.values())]
    
    last_three = ['radio', 'drinking', 'reach_side']
    mask = multilabels[last_three].any(axis=1)
    multilabels['left_hand'] = np.where(mask, 1, multilabels['left_hand'])
    
    return pd.concat([df.drop(columns='label'), multilabels], axis=1)
def truncate(f, n):
    return np.floor(f * 10**n) / 10**n

def compute_multilabel_confusion_matrix(Y_true, Y_pred, class_labels, set_name="train", threshold=0.5):
    
    y_true = Y_true.values if hasattr(Y_true, 'values') else Y_true
    y_pred = (Y_pred >= threshold).astype(int)
    
    relevant_combinations = [
        [1, 0, 0, 0, 0, 0],  # both_hands solo
        [0, 1, 0, 0, 0, 0],  # left_hand solo
        [0, 1, 0, 1, 0, 0],  # left_hand + radio
        [0, 1, 0, 0, 1, 0],  # left_hand + drinking
        [0, 1, 0, 0, 0, 1],  # left_hand + reach_side
        [0, 0, 1, 0, 0, 0]   # right_hand solo
    ]
    
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
    
    combo_to_idx = {tuple(combo): idx for idx, combo in enumerate(relevant_combinations)}
    
    for true_row, pred_row in zip(y_true, y_pred):
        true_combo = tuple(true_row)
        pred_combo = tuple(pred_row)
        
        if true_combo in combo_to_idx and pred_combo in combo_to_idx:
            i = combo_to_idx[true_combo]
            j = combo_to_idx[pred_combo]
            conf_matrix[i, j] += 1
    
    index_labels = [combo_names[tuple(combo)] for combo in relevant_combinations]
    conf_matrix = pd.DataFrame(conf_matrix, index=index_labels, columns=index_labels)
    
    norm_conf = conf_matrix.div(conf_matrix.sum(axis=1), axis=0).fillna(0)
    
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

    def load_json(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {file_path}. Trying alternative encoding...")
            with open(file_path, 'r', encoding='latin-1') as f:
                return json.load(f)

    def open_jsons(self):

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
                
                "label": item['type'] 
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
               
                "label": item['type']
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
                
                "label": item['type']
            }
            
            self.rows.append(features)
            if add_noise:
                self.rows.append(features_tras)
                self.rows.append(features_gauss)


def train_actions_model(json_files_train, json_files_test):
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

    X_train = dataset_train.iloc[:, :-6]  # all but labels
    Y_train = dataset_train.iloc[:, -6:]  # labels

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

    X_test = dataset_test.iloc[:, :-6]  # all but labels
    Y_test = dataset_test.iloc[:, -6:]  # labels

    train_class_counts = Y_train.sum(axis=0)
    test_class_counts = Y_test.sum(axis=0)
    
    print("Distribución de clases en el conjunto de entrenamiento:")
    print(train_class_counts)
    print("\nDistribución de clases en el conjunto de prueba:")
    print(test_class_counts)

    model = OneVsRestClassifier(RandomForestClassifier(random_state=1, class_weight='balanced', n_estimators=3))#, n_jobs=-1))
    model.fit(X_train, Y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    compute_multilabel_confusion_matrix(Y_train, train_preds, class_labels=list(Y_train.columns), set_name="train")
    compute_multilabel_confusion_matrix(Y_test, test_preds, class_labels=list(Y_train.columns), set_name="test")
    
    return model


"""

NO ARGUMENTS NEEDED

"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from datetime import datetime
import random
import math
from sklearn.metrics import confusion_matrix, classification_report

def tf(to_tf, original, flag):
        if(flag == 0):
            return (to_tf[0] - original[0]) #* scale
        elif(flag == 1):
            return (to_tf[1] - original[1]) #* scale

        return 0

def calc_ang(first_pnt, sec_pnt, middle_pnt):
    first_pnt = np.array(first_pnt)
    sec_pnt = np.array(sec_pnt)
    middle_pnt = np.array(middle_pnt)

    u = first_pnt - middle_pnt
    v = sec_pnt - middle_pnt

    dot_product = np.dot(u, v)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    angle_radians = np.arccos(dot_product / (norm_u * norm_v))

    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def convert_to_multilabel(df):

    label_mapping = {
        "gaze_zone/left_mirror": "left_mirror",
        "gaze_zone/right_mirror": "right_mirror",
        "gaze_zone/center_mirror": "center_mirror",
        "gaze_zone/left": "left",
        "gaze_zone/right": "right",
        "gaze_zone/front": "front",
        "gaze_zone/front_right": "front_right",
        "gaze_zone/steering_wheel": "steering_wheel",
    }
    df['label'] = df['label'].map(label_mapping)
    multilabels = pd.get_dummies(df['label']) 
    df = pd.concat([df, multilabels], axis=1)  
    df.drop(columns=['label'], inplace=True)  
    return df

def compute_confusion_matrix(Y_true, Y_pred, class_labels, set_name="train", threshold=0.5):
    n_labels = len(class_labels)
    global_conf_matrix = np.zeros((n_labels, n_labels))
    no_predictions = 0
    yes_predictions = 0

    binary_preds = (Y_pred >= threshold).astype(int)

    for i in range(Y_true.shape[0]):  
        true_labels = np.where(Y_true.iloc[i] == 1)[0]  
        pred_labels = np.where(binary_preds[i] == 1)[0] 

        if len(pred_labels) == 0:
            no_predictions += 1
        else:
            yes_predictions += 1

        for true in true_labels:
            for pred in pred_labels:
                global_conf_matrix[true, pred] += 1  

    row_sums = global_conf_matrix.sum(axis=1, keepdims=True)
    global_conf_matrix_normalized = np.divide(global_conf_matrix, row_sums, 
                                              where=row_sums != 0) 

    plt.figure(figsize=(10, 7))
    rounded_values = np.floor(global_conf_matrix_normalized.T * 100) / 100
    sns.heatmap(rounded_values, annot=True, fmt=".2f", cmap="Blues",

                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel("Etiqueta Real")
    plt.ylabel("Predicción")
    plt.title(f"Matriz de Confusión de {set_name.capitalize()}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{set_name}_confusion_matrix_{timestamp}.png"
    plt.savefig(filename)
    plt.show()

    correct_predictions = np.diag(global_conf_matrix)
    total_predictions_per_class = global_conf_matrix.sum(axis=1)
    correct_percentages = correct_predictions / total_predictions_per_class * 100

    print(f"Porcentajes de aciertos por clase: {correct_percentages}")

    plt.figure(figsize=(8, 5))
    plt.bar(class_labels, correct_percentages, color='blue')
    plt.xlabel("Clases")
    plt.ylabel("Porcentaje de aciertos")
    plt.title(f"Aciertos por clase en {set_name.capitalize()} (en porcentaje)")
    plt.xticks(rotation=45)

    filename_bar = f"{set_name}_correct_predictions_{timestamp}.png"
    plt.savefig(filename_bar)
    plt.show()

    print(f"Número de ejemplos sin predicciones en {set_name}: {no_predictions}")
    print(f"Número de ejemplos con predicciones en {set_name}: {yes_predictions}")

class RandomForest:
    def __init__(self, center_mirror, left_mirror, right_mirror, left, right, front, front_right, steering_wheel):
        self.center_mirror_j = center_mirror
        self.left_mirror_j = left_mirror
        self.right_mirror_j = right_mirror
        self.left_j = left
        self.right_j = right
        self.front_j = front
        self.front_right_j = front_right
        self.steering_wheel_j = steering_wheel

        self.center_mirror = []
        self.left_mirror = []
        self.right_mirror = []
        self.left = []
        self.right = []
        self.front = []
        self.front_right = []
        self.steering_wheel = []

        self.rows = []

    def open_jsons(self):

        try:
            with open(self.center_mirror_j, 'r', encoding='utf-8-sig') as f:
                self.center_mirror = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.center_mirror_j}. Trying alternative encoding...")
            with open(self.center_mirror_j, 'r', encoding='latin-1') as f:
                self.center_mirror = json.load(f)
        try:
            with open(self.left_mirror_j, 'r', encoding='utf-8-sig') as f:
                self.left_mirror = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.left_mirror_j}. Trying alternative encoding...")
            with open(self.left_mirror_j, 'r', encoding='latin-1') as f:
                self.left_mirror = json.load(f)
        try:
            with open(self.right_mirror_j, 'r', encoding='utf-8-sig') as f:
                self.right_mirror = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.right_mirror_j}. Trying alternative encoding...")
            with open(self.right_mirror_j, 'r', encoding='latin-1') as f:
                self.right_mirror = json.load(f)
        try:
            with open(self.left_j, 'r', encoding='utf-8-sig') as f:
                self.left = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.left_j}. Trying alternative encoding...")
            with open(self.left_j, 'r', encoding='latin-1') as f:
                self.left = json.load(f)
        try:
            with open(self.right_j, 'r', encoding='utf-8-sig') as f:
                self.right = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.right_j}. Trying alternative encoding...")
            with open(self.right_j, 'r', encoding='latin-1') as f:
                self.right = json.load(f)
        try:
            with open(self.front_j, 'r', encoding='utf-8-sig') as f:
                self.front = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.front_j}. Trying alternative encoding...")
            with open(self.front_j, 'r', encoding='latin-1') as f:
                self.front = json.load(f)
        try:
            with open(self.front_right_j, 'r', encoding='utf-8-sig') as f:
                self.front_right = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.front_right_j}. Trying alternative encoding...")
            with open(self.front_right_j, 'r', encoding='latin-1') as f:
                self.front_right = json.load(f)
        try:
            with open(self.steering_wheel_j, 'r', encoding='utf-8-sig') as f:
                self.steering_wheel = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.steering_wheel_j}. Trying alternative encoding...")
            with open(self.steering_wheel_j, 'r', encoding='latin-1') as f:
                self.steering_wheel = json.load(f)

    def generate_random_number(self, prin_num, rang_low=0.015, rang_high=0.015):
        return random.uniform(prin_num - rang_low, prin_num + rang_high)

    def add_gaussian_noise(self, value, mean=0, std=0.0001):
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

    def normalize_gaze(self, gaze, frame, flag):
        if flag == 0:
            return (gaze[0] - frame.shape[1] / 2) / (frame.shape[1] / 2)
        elif flag == 1:
            return (gaze[1] - frame.shape[0] / 2) / (frame.shape[0] / 2)

    def prepare_data(self, data, frame):

        for item in data:

            rotation = self.generate_random_angle()
            first_pnt = (item['face']['face'][0][0], item['face']['face'][0][1])
            nose = (item['face']['face'][1][0], item['face']['face'][1][1])

            features = {
                "face_0_x": tf(item['face']['face'][0], nose, 0),
                "face_0_y": tf(item['face']['face'][0], nose, 1),
                "face_1_x": tf(item['face']['face'][1], nose, 0),   
                "face_1_y": tf(item['face']['face'][1], nose, 1),
                "face_2_x": tf(item['face']['face'][2], nose, 0),                
                "face_2_y": tf(item['face']['face'][2], nose, 1),                
                "face_3_x": tf(item['face']['face'][3], nose, 0),                
                "face_3_y": tf(item['face']['face'][3], nose, 1),                
                "face_4_x": tf(item['face']['face'][4], nose, 0),                
                "face_4_y": tf(item['face']['face'][4], nose, 1),
                "face_5_x": tf(item['face']['face'][5], nose, 0),
                "face_5_y": tf(item['face']['face'][5], nose, 1),
                "face_6_x": tf(item['face']['face'][6], nose, 0),
                "face_6_y": tf(item['face']['face'][6], nose, 1),
                "face_7_x": tf(item['face']['face'][7], nose, 0),
                "face_7_y": tf(item['face']['face'][7], nose, 1),
                "face_8_x": tf(item['face']['face'][8], nose, 0),
                "face_8_y": tf(item['face']['face'][8], nose, 1),
                "face_9_x": tf(item['face']['face'][9], nose, 0),
                "face_9_y": tf(item['face']['face'][9], nose, 1),
                "face_10_x": tf(item['face']['face'][10], nose, 0),
                "face_10_y": tf(item['face']['face'][10], nose, 1),
                "face_11_x": tf(item['face']['face'][11], nose, 0),
                "face_11_y": tf(item['face']['face'][11], nose, 1),
                "face_12_x": tf(item['face']['face'][12], nose, 0),
                "face_12_y": tf(item['face']['face'][12], nose, 1),
                "face_13_x": tf(item['face']['face'][13], nose, 0),
                "face_13_y": tf(item['face']['face'][13], nose, 1),
                "face_14_x": tf(item['face']['face'][14], nose, 0),
                "face_14_y": tf(item['face']['face'][14], nose, 1),
                "face_15_x": tf(item['face']['face'][15], nose, 0),
                "face_15_y": tf(item['face']['face'][15], nose, 1),
                "face_16_x": tf(item['face']['face'][16], nose, 0),
                "face_16_y": tf(item['face']['face'][16], nose, 1),
                "face_17_x": tf(item['face']['face'][17], nose, 0),
                "face_17_y": tf(item['face']['face'][17], nose, 1),
                "face_18_x": tf(item['face']['face'][18], nose, 0),
                "face_18_y": tf(item['face']['face'][18], nose, 1),
                "face_19_x": tf(item['face']['face'][19], nose, 0),
                "face_19_y": tf(item['face']['face'][19], nose, 1),
                "face_20_x": tf(item['face']['face'][20], nose, 0),
                "face_20_y": tf(item['face']['face'][20], nose, 1),
                "face_21_x": tf(item['face']['face'][21], nose, 0),
                "face_21_y": tf(item['face']['face'][21], nose, 1),
                "face_22_x": tf(item['face']['face'][22], nose, 0),
                "face_22_y": tf(item['face']['face'][22], nose, 1),
                "face_23_x": tf(item['face']['face'][23], nose, 0),
                "face_23_y": tf(item['face']['face'][23], nose, 1),
                "face_24_x": tf(item['face']['face'][24], nose, 0),
                "face_24_y": tf(item['face']['face'][24], nose, 1),
                "face_25_x": tf(item['face']['face'][25], nose, 0),
                "face_25_y": tf(item['face']['face'][25], nose, 1),
                "face_26_x": tf(item['face']['face'][26], nose, 0),
                "face_26_y": tf(item['face']['face'][26], nose, 1),
                "gaze_0_x": self.normalize_gaze(item['face']['gaze'][0], frame, 0),
                "gaze_0_y": self.normalize_gaze(item['face']['gaze'][0], frame, 1),
                "gaze_1_x": self.normalize_gaze(item['face']['gaze'][1], frame, 0),
                "gaze_1_y": self.normalize_gaze(item['face']['gaze'][1], frame, 1), 

                "label": item['type'] 
            }

            features_gauss = {
                "face_0_x": self.add_gaussian_noise(tf(item['face']['face'][0], nose, 0)),
                "face_0_y": self.add_gaussian_noise(tf(item['face']['face'][0], nose, 1)),
                "face_1_x": self.add_gaussian_noise(tf(item['face']['face'][1], nose, 0)),
                "face_1_y": self.add_gaussian_noise(tf(item['face']['face'][1], nose, 1)),
                "face_2_x": self.add_gaussian_noise(tf(item['face']['face'][2], nose, 0)),
                "face_2_y": self.add_gaussian_noise(tf(item['face']['face'][2], nose, 1)),
                "face_3_x": self.add_gaussian_noise(tf(item['face']['face'][3], nose, 0)),
                "face_3_y": self.add_gaussian_noise(tf(item['face']['face'][3], nose, 1)),
                "face_4_x": self.add_gaussian_noise(tf(item['face']['face'][4], nose, 0)),
                "face_4_y": self.add_gaussian_noise(tf(item['face']['face'][4], nose, 1)),
                "face_5_x": self.add_gaussian_noise(tf(item['face']['face'][5], nose, 0)),
                "face_5_y": self.add_gaussian_noise(tf(item['face']['face'][5], nose, 1)),
                "face_6_x": self.add_gaussian_noise(tf(item['face']['face'][6], nose, 0)),
                "face_6_y": self.add_gaussian_noise(tf(item['face']['face'][6], nose, 1)),
                "face_7_x": self.add_gaussian_noise(tf(item['face']['face'][7], nose, 0)),
                "face_7_y": self.add_gaussian_noise(tf(item['face']['face'][7], nose, 1)),
                "face_8_x": self.add_gaussian_noise(tf(item['face']['face'][8], nose, 0)),
                "face_8_y": self.add_gaussian_noise(tf(item['face']['face'][8], nose, 1)),
                "face_9_x": self.add_gaussian_noise(tf(item['face']['face'][9], nose, 0)),
                "face_9_y": self.add_gaussian_noise(tf(item['face']['face'][9], nose, 1)),
                "face_10_x": self.add_gaussian_noise(tf(item['face']['face'][10], nose, 0)),
                "face_10_y": self.add_gaussian_noise(tf(item['face']['face'][10], nose, 1)),
                "face_11_x": self.add_gaussian_noise(tf(item['face']['face'][11], nose, 0)),
                "face_11_y": self.add_gaussian_noise(tf(item['face']['face'][11], nose, 1)),
                "face_12_x": self.add_gaussian_noise(tf(item['face']['face'][12], nose, 0)),
                "face_12_y": self.add_gaussian_noise(tf(item['face']['face'][12], nose, 1)),
                "face_13_x": self.add_gaussian_noise(tf(item['face']['face'][13], nose, 0)),
                "face_13_y": self.add_gaussian_noise(tf(item['face']['face'][13], nose, 1)),
                "face_14_x": self.add_gaussian_noise(tf(item['face']['face'][14], nose, 0)),
                "face_14_y": self.add_gaussian_noise(tf(item['face']['face'][14], nose, 1)),
                "face_15_x": self.add_gaussian_noise(tf(item['face']['face'][15], nose, 0)),
                "face_15_y": self.add_gaussian_noise(tf(item['face']['face'][15], nose, 1)),
                "face_16_x": self.add_gaussian_noise(tf(item['face']['face'][16], nose, 0)),
                "face_16_y": self.add_gaussian_noise(tf(item['face']['face'][16], nose, 1)),
                "face_17_x": self.add_gaussian_noise(tf(item['face']['face'][17], nose, 0)),
                "face_17_y": self.add_gaussian_noise(tf(item['face']['face'][17], nose, 1)),
                "face_18_x": self.add_gaussian_noise(tf(item['face']['face'][18], nose, 0)),
                "face_18_y": self.add_gaussian_noise(tf(item['face']['face'][18], nose, 1)),
                "face_19_x": self.add_gaussian_noise(tf(item['face']['face'][19], nose, 0)),
                "face_19_y": self.add_gaussian_noise(tf(item['face']['face'][19], nose, 1)),
                "face_20_x": self.add_gaussian_noise(tf(item['face']['face'][20], nose, 0)),
                "face_20_y": self.add_gaussian_noise(tf(item['face']['face'][20], nose, 1)),
                "face_21_x": self.add_gaussian_noise(tf(item['face']['face'][21], nose, 0)),
                "face_21_y": self.add_gaussian_noise(tf(item['face']['face'][21], nose, 1)),
                "face_22_x": self.add_gaussian_noise(tf(item['face']['face'][22], nose, 0)),
                "face_22_y": self.add_gaussian_noise(tf(item['face']['face'][22], nose, 1)),
                "face_23_x": self.add_gaussian_noise(tf(item['face']['face'][23], nose, 0)),
                "face_23_y": self.add_gaussian_noise(tf(item['face']['face'][23], nose, 1)),
                "face_24_x": self.add_gaussian_noise(tf(item['face']['face'][24], nose, 0)),
                "face_24_y": self.add_gaussian_noise(tf(item['face']['face'][24], nose, 1)),
                "face_25_x": self.add_gaussian_noise(tf(item['face']['face'][25], nose, 0)),
                "face_25_y": self.add_gaussian_noise(tf(item['face']['face'][25], nose, 1)),
                "face_26_x": self.add_gaussian_noise(tf(item['face']['face'][26], nose, 0)),
                "face_26_y": self.add_gaussian_noise(tf(item['face']['face'][26], nose, 1)),
                "gaze_0_x": self.add_gaussian_noise(self.normalize_gaze(item['face']['gaze'][0], frame, 0)), 
                "gaze_0_y": self.add_gaussian_noise(self.normalize_gaze(item['face']['gaze'][0], frame, 1)), 
                "gaze_1_x": self.add_gaussian_noise(self.normalize_gaze(item['face']['gaze'][1], frame, 0)), 
                "gaze_1_y": self.add_gaussian_noise(self.normalize_gaze(item['face']['gaze'][1], frame, 1)),     

                "label": item['type']
            }

            features_gauss_2 = {
                "face_0_x": self.add_gaussian_noise(tf(item['face']['face'][0], nose, 0)),
                "face_0_y": self.add_gaussian_noise(tf(item['face']['face'][0], nose, 1)),
                "face_1_x": self.add_gaussian_noise(tf(item['face']['face'][1], nose, 0)),
                "face_1_y": self.add_gaussian_noise(tf(item['face']['face'][1], nose, 1)),
                "face_2_x": self.add_gaussian_noise(tf(item['face']['face'][2], nose, 0)),
                "face_2_y": self.add_gaussian_noise(tf(item['face']['face'][2], nose, 1)),
                "face_3_x": self.add_gaussian_noise(tf(item['face']['face'][3], nose, 0)),
                "face_3_y": self.add_gaussian_noise(tf(item['face']['face'][3], nose, 1)),
                "face_4_x": self.add_gaussian_noise(tf(item['face']['face'][4], nose, 0)),
                "face_4_y": self.add_gaussian_noise(tf(item['face']['face'][4], nose, 1)),
                "face_5_x": self.add_gaussian_noise(tf(item['face']['face'][5], nose, 0)),
                "face_5_y": self.add_gaussian_noise(tf(item['face']['face'][5], nose, 1)),
                "face_6_x": self.add_gaussian_noise(tf(item['face']['face'][6], nose, 0)),
                "face_6_y": self.add_gaussian_noise(tf(item['face']['face'][6], nose, 1)),
                "face_7_x": self.add_gaussian_noise(tf(item['face']['face'][7], nose, 0)),
                "face_7_y": self.add_gaussian_noise(tf(item['face']['face'][7], nose, 1)),
                "face_8_x": self.add_gaussian_noise(tf(item['face']['face'][8], nose, 0)),
                "face_8_y": self.add_gaussian_noise(tf(item['face']['face'][8], nose, 1)),
                "face_9_x": self.add_gaussian_noise(tf(item['face']['face'][9], nose, 0)),
                "face_9_y": self.add_gaussian_noise(tf(item['face']['face'][9], nose, 1)),
                "face_10_x": self.add_gaussian_noise(tf(item['face']['face'][10], nose, 0)),
                "face_10_y": self.add_gaussian_noise(tf(item['face']['face'][10], nose, 1)),
                "face_11_x": self.add_gaussian_noise(tf(item['face']['face'][11], nose, 0)),
                "face_11_y": self.add_gaussian_noise(tf(item['face']['face'][11], nose, 1)),
                "face_12_x": self.add_gaussian_noise(tf(item['face']['face'][12], nose, 0)),
                "face_12_y": self.add_gaussian_noise(tf(item['face']['face'][12], nose, 1)),
                "face_13_x": self.add_gaussian_noise(tf(item['face']['face'][13], nose, 0)),
                "face_13_y": self.add_gaussian_noise(tf(item['face']['face'][13], nose, 1)),
                "face_14_x": self.add_gaussian_noise(tf(item['face']['face'][14], nose, 0)),
                "face_14_y": self.add_gaussian_noise(tf(item['face']['face'][14], nose, 1)),
                "face_15_x": self.add_gaussian_noise(tf(item['face']['face'][15], nose, 0)),
                "face_15_y": self.add_gaussian_noise(tf(item['face']['face'][15], nose, 1)),
                "face_16_x": self.add_gaussian_noise(tf(item['face']['face'][16], nose, 0)),
                "face_16_y": self.add_gaussian_noise(tf(item['face']['face'][16], nose, 1)),
                "face_17_x": self.add_gaussian_noise(tf(item['face']['face'][17], nose, 0)),
                "face_17_y": self.add_gaussian_noise(tf(item['face']['face'][17], nose, 1)),
                "face_18_x": self.add_gaussian_noise(tf(item['face']['face'][18], nose, 0)),
                "face_18_y": self.add_gaussian_noise(tf(item['face']['face'][18], nose, 1)),
                "face_19_x": self.add_gaussian_noise(tf(item['face']['face'][19], nose, 0)),
                "face_19_y": self.add_gaussian_noise(tf(item['face']['face'][19], nose, 1)),
                "face_20_x": self.add_gaussian_noise(tf(item['face']['face'][20], nose, 0)),
                "face_20_y": self.add_gaussian_noise(tf(item['face']['face'][20], nose, 1)),
                "face_21_x": self.add_gaussian_noise(tf(item['face']['face'][21], nose, 0)),
                "face_21_y": self.add_gaussian_noise(tf(item['face']['face'][21], nose, 1)),
                "face_22_x": self.add_gaussian_noise(tf(item['face']['face'][22], nose, 0)),
                "face_22_y": self.add_gaussian_noise(tf(item['face']['face'][22], nose, 1)),
                "face_23_x": self.add_gaussian_noise(tf(item['face']['face'][23], nose, 0)),
                "face_23_y": self.add_gaussian_noise(tf(item['face']['face'][23], nose, 1)),
                "face_24_x": self.add_gaussian_noise(tf(item['face']['face'][24], nose, 0)),
                "face_24_y": self.add_gaussian_noise(tf(item['face']['face'][24], nose, 1)),
                "face_25_x": self.add_gaussian_noise(tf(item['face']['face'][25], nose, 0)),
                "face_25_y": self.add_gaussian_noise(tf(item['face']['face'][25], nose, 1)),
                "face_26_x": self.add_gaussian_noise(tf(item['face']['face'][26], nose, 0)),
                "face_26_y": self.add_gaussian_noise(tf(item['face']['face'][26], nose, 1)),
                "gaze_0_x": self.add_gaussian_noise(self.normalize_gaze(item['face']['gaze'][0], frame, 0)), 
                "gaze_0_y": self.add_gaussian_noise(self.normalize_gaze(item['face']['gaze'][0], frame, 1)), 
                "gaze_1_x": self.add_gaussian_noise(self.normalize_gaze(item['face']['gaze'][1], frame, 0)), 
                "gaze_1_y": self.add_gaussian_noise(self.normalize_gaze(item['face']['gaze'][1], frame, 1)), 

                "label": item['type'] 
            }

            self.rows.append(features)
            self.rows.append(features_gauss)
            self.rows.append(features_gauss_2)

def train_gaze_model(json_files_1, json_files_2):
    RandomForest_performer = RandomForest(*json_files_1)
    RandomForest_performer.open_jsons()

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    for data in [RandomForest_performer.center_mirror, RandomForest_performer.left_mirror, RandomForest_performer.right_mirror,
                 RandomForest_performer.front, RandomForest_performer.left, RandomForest_performer.right,
                 RandomForest_performer.front_right, RandomForest_performer.steering_wheel]:
        RandomForest_performer.prepare_data(data, frame)

    dataset = pd.DataFrame(RandomForest_performer.rows)
    dataset = convert_to_multilabel(dataset)

    X_train = dataset.iloc[:, :-8]
    Y_train = dataset.iloc[:, -8:]

    RandomForest_performer_test = RandomForest(*json_files_2)
    RandomForest_performer_test.open_jsons()

    for data in [RandomForest_performer_test.center_mirror, RandomForest_performer_test.left_mirror, RandomForest_performer_test.right_mirror,
                 RandomForest_performer_test.front, RandomForest_performer_test.left, RandomForest_performer_test.right,
                 RandomForest_performer_test.front_right, RandomForest_performer_test.steering_wheel]:
        RandomForest_performer_test.prepare_data(data, frame)

    dataset_test = pd.DataFrame(RandomForest_performer_test.rows)
    dataset_test = convert_to_multilabel(dataset_test)

    X_test = dataset_test.iloc[:, :-8]
    Y_test = dataset_test.iloc[:, -8:]

    feature_columns = [f'face_{i}_{axis}' for i in range(27) for axis in ['x', 'y']] + ['gaze_x1', 'gaze_y1', 'gaze_x2', 'gaze_y2']
    X_train = pd.DataFrame(X_train, columns=feature_columns)
    X_test = pd.DataFrame(X_test, columns=feature_columns)

    Y_train_labels = Y_train.idxmax(axis=1)
    Y_test_labels = Y_test.idxmax(axis=1)

    class_labels = ['center_mirror', 'left_mirror', 'right_mirror', 'front', 'left', 'right', 'front_right', 'steering_wheel']

    print("Distribución de clases en el conjunto de entrenamiento:")
    print(Y_train_labels.value_counts())

    print("\nDistribución de clases en el conjunto de prueba:")
    print(Y_test_labels.value_counts())

    model = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_leaf=20, class_weight='balanced')  
    model.fit(X_train, Y_train_labels)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    cm_train = confusion_matrix(Y_train_labels, train_preds, labels=class_labels, normalize='true')
    print("\nMatriz de confusión train:")
    print(cm_train)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_train, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión train')
    plt.show()

    cm_test = confusion_matrix(Y_test_labels, test_preds, labels=class_labels, normalize='true')
    print("\nMatriz de confusión test:")
    print(cm_test)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_test, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión test')
    plt.show()

    print("\nReporte de clasificación:")
    print(classification_report(Y_test_labels, test_preds, target_names=class_labels))

    return model


"""

python reconstructor_balance.py <json_file> <data_path>

json_file: JSON file with the balanced data
data_path: Path where the video frames are located

It will print a video showing the hand, face, and pose frames for each balanced action.

"""

import cv2
import sys
import numpy as np
import json
import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import math

def tf(to_tf, original, flag):
        if(flag == 0):
            return (original[0])
        elif(flag == 1):
            return (original[1])

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
        "hands_using_wheel/both": "both_hands",
        "hands_using_wheel/only_left": "left_hand",
        "hands_using_wheel/only_right": "right_hand",
        "driver_actions/radio": "radio",
        "driver_actions/drinking": "drinking",
        "driver_actions/reach_side": "reach_side"
    }
    df['label'] = df['label'].map(label_mapping)
    multilabels = pd.get_dummies(df['label'])  
    df = pd.concat([df, multilabels], axis=1)  
    df.drop(columns=['label'], inplace=True)  
    return df

def generate_random_number(prin_num, rang_low=0.015, rang_high=0.015):
    return random.uniform(prin_num[0] - rang_low, prin_num[0] + rang_high), random.uniform(prin_num[1] - rang_low, prin_num[1] + rang_high), prin_num[2]

def add_gaussian_noise(value, mean=0, std=0.01):
    return value[0] + np.random.normal(mean, std), value[1] + np.random.normal(mean, std), value[2]

def generate_random_angle(angle_range=7):
    return math.radians(random.uniform(-angle_range, angle_range))

def rotate_point_around(x, y, cx, cy, angle_rad):
    x -= cx
    y -= cy

    new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

    return new_x + cx, new_y + cy

def prepare_data(item, image):

    right_elbow = item['pose']['pose'][1]

    rotation = generate_random_angle()

    features = [
        item['pose']['pose'][50], item['pose']['pose'][51], 
        item['pose']['pose'][0], item['pose']['pose'][1], 
        item['pose']['pose'][2], item['pose']['pose'][3], 
        item['pose']['pose'][4], item['pose']['pose'][5]
    ]

    features_tras = [
        rotate_point_around(p[0], p[1], right_elbow[0], right_elbow[1], rotation) + (p[2],)
        for p in features
    ]

    features_gauss = [add_gaussian_noise(item['pose']['pose'][50]), add_gaussian_noise(item['pose']['pose'][51]), add_gaussian_noise(item['pose']['pose'][0]), add_gaussian_noise(item['pose']['pose'][1]), add_gaussian_noise(item['pose']['pose'][2]), add_gaussian_noise(item['pose']['pose'][3]), add_gaussian_noise(item['pose']['pose'][4]), add_gaussian_noise(item['pose']['pose'][5])]

    plot_features_on_image(image, features, features_tras, features_gauss)

def draw_connections(frame, keypoints, connections, color):
    
    for idx1, idx2 in connections:
        x1, y1, x2, y2 = None, None, None, None
        
        for x, y, z in keypoints:
            if z == idx1:  
                x1 = int(x * frame.shape[1])
                y1 = int(y * frame.shape[0])
            elif z == idx2:  
                x2 = int(x * frame.shape[1])
                y2 = int(y * frame.shape[0])

        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)  
            
def plot_features_on_image(image, features, features_tras, features_gauss):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    else:
        print("Formato de imagen desconocido")

    for x, y, z in features:
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)  # Verde

    for x, y, z in features_tras:
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Rojo

    for x, y, z in features_gauss:
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 10, (255, 0, 0), -1)  # Azul

    connections = [
            (12, 24), (12, 11), (11, 23), (24, 23),
            (12, 14), (14, 16), (11, 13), (13, 15)
        ]

    draw_connections(image, features, connections, (0, 255, 0))
    draw_connections(image, features_tras, connections, (0, 0, 255))
    draw_connections(image, features_gauss, connections, (255, 0, 0))

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    
    plt.scatter([], [], c='green', label='Original', s=100)
    plt.scatter([], [], c='blue', label='Rotacion', s=100)
    plt.scatter([], [], c='red', label='Ruido Gauss', s=100)

    plt.legend()  
    plt.show()


class balanceReconstructor:
    def __init__(self, json_file, data_path):
        self.data_hands = []
        self.data_face = []
        self.data_pose = []
        self.data = []

        self.HANDS_CONNECTION = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (13, 17), (17, 18), (18, 19), (19, 20)
        ]

        self.FACE_CONNECTION = [ 
            (17, 291), (17, 61), (0, 61), (0, 291),
            (61, 4), (4, 291), (4, 48), (4, 278),
            (291, 426), (61, 206), (61, 50), (291, 280),
            (206, 48), (426, 278), (48, 50), (278, 280),
            (4, 107), (4, 336), (50, 145), (280, 374),
            (122, 107), (122, 145), (351, 336), (351, 374),
            (145, 130), (145, 133), (374, 359), (374, 362),
            (130, 159), (130, 46), (359, 386), (359, 276),
            (133, 159), (362, 386), (46, 105), (276, 334),
            (105, 107), (334, 336)          
        ]
        self.POSE_CONNECTION = [
            (12, 24), (12, 11), (11, 23), (24, 23),
            (12, 14), (14, 16), (11, 13), (13, 15)
        ]

        self.balanced_json = json_file
        self.data_path = data_path

    def open_json(self):
        print(f"Opening JSON file {self.balanced_json}")
        try:
            with open(self.balanced_json, 'r', encoding='utf-8-sig') as f:
                self.data = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.balanced_json}. Trying alternative encoding...")
            with open(self.balanced_json, 'r', encoding='latin-1') as f:
                self.data = json.load(f)

    def reconstruct(self):
        self.open_json()

        for action in self.data:
            json_session = action['json'].split('/')[1]

            frame = action['frame']

            image_face = json_session + "_" + str(frame) + "_" + "face.png"
            image_hands = json_session + "_" + str(frame) + "_" + "hands.png"
            image_pose = json_session + "_" + str(frame) + "_" + "pose.png"

            path_pose = self.data_path + action['type'] + "/" + image_pose
            path_hands = self.data_path + action['type'] + "/" + image_hands
            path_face = self.data_path + action['type'] + "/" + image_face

            face_im = cv2.imread(path_face)
            hands_im = cv2.imread(path_hands)
            pose_im = cv2.imread(path_pose)

            prepare_data(action, pose_im)

    def show_collage(self, face_im, hands_im, pose_im, action):
        frame1 = hands_im
        frame2 = pose_im
        frame3 = face_im

        if frame1 is None or frame2 is None or frame3 is None:
            print("Error al leer uno de los frames")
            return
        
        print("Action: ", action['type'])

        self.paint_frame(frame1, action['frame'], "hands", action)
        self.paint_frame(frame2, action['frame'], "pose", action)
        self.paint_frame(frame3, action['frame'], "face", action)

        height, width = frame1.shape[:2]
        new_size = (width // 2, height // 2)

        frame1 = cv2.resize(frame1, new_size)
        frame2 = cv2.resize(frame2, new_size)
        frame3 = cv2.resize(frame3, new_size)

        combined_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        combined_frame[0:height // 2, 0:width // 2] = frame1
        combined_frame[0:height // 2, width // 2:width] = frame2
        combined_frame[height // 2:height, 0:width // 2] = frame3

        cv2.putText(combined_frame, action['type'], (width // 2 + 10, height // 2 + 30 + 1 * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Combined Video', combined_frame)
        cv2.waitKey(1)

    def open_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el video {video_path}")
            return None
        cap.release()

    def paint_frame(self, frame, frame_number, json, data):
        print(data)
        if json == "hands":
            for x, y, z in data['hands']['hands']:
                x = int(x * frame.shape[1])
                y = int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for x, y in data['hands']['centers']:
                x = int(x * frame.shape[1])
                y = int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), 30)

            hand_right = data['hands']["hands"][:21]
            hand_left = data['hands']["hands"][21:]

            self.draw_connections(frame, hand_right, self.HANDS_CONNECTION)
            self.draw_connections(frame, hand_left, self.HANDS_CONNECTION)

        elif json == "pose":
            for x, y, z in data['pose']['pose']:
                x = int(x * frame.shape[1])
                y = int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                pose_data = data['pose']["pose"][:8]
                right_hand = data['pose']["pose"][8:29]
                left_hand = data['pose']["pose"][29:50]

                self.draw_connections(frame, pose_data, self.POSE_CONNECTION)
                self.draw_connections(frame, right_hand, self.HANDS_CONNECTION)
                self.draw_connections(frame, left_hand, self.HANDS_CONNECTION)

        elif json == "face":
            for x, y, indx in data['face']["face"]:
                x = int(x * frame.shape[1])
                y = int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for x, y in data['face']["gaze"]:
                x = int(x * frame.shape[1])
                y = int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                self.draw_connections(frame, data['face']["face"], self.FACE_CONNECTION)

    def draw_connections(self, frame, keypoints, connections):

        for idx1, idx2 in connections:
            x1, y1, x2, y2 = None, None, None, None
            for x, y, inx in keypoints:
                if inx == idx1:
                    x1 = int(x * frame.shape[1])
                    y1 = int(y * frame.shape[0])
                elif inx == idx2:
                    x2 = int(x * frame.shape[1])
                    y2 = int(y * frame.shape[0])

            if x1 and y1 and x2 and y2:

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python reconstructor_balance.py <json_file> <data_path>")
    else:
        vr = balanceReconstructor(sys.argv[1], sys.argv[2])
        vr.reconstruct()

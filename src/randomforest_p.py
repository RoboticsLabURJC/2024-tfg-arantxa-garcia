"""

Este script se encarga de entrenar un modelo RandomForest con los datos balanceados y de reconstruir
los videos prediciendo las acciones de cada frame. En este script luego predice sobre los datos
de una sesion con datos desbalanceados.

NO NECESITA ARGUMENTOS

"""

import sys
import json
import numpy as np
import pandas as pd
import time
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.RandomForest import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import os
from datetime import datetime

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

    def open_jsons(self):

        try:
            with open(self.both_j, 'r', encoding='utf-8-sig') as f:
                self.both = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.both_j}. Trying alternative encoding...")
            with open(self.both_j, 'r', encoding='latin-1') as f:
                self.both = json.load(f)

        try:
            with open(self.onlyleft_j, 'r', encoding='utf-8-sig') as f:
                self.onlyleft = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.onlyleft_j}. Trying alternative encoding...")
            with open(self.onlyleft_j, 'r', encoding='latin-1') as f:
                self.onlyleft = json.load(f)

        try:
            with open(self.onlyright_j, 'r', encoding='utf-8-sig') as f:
                self.onlyright = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.onlyright_j}. Trying alternative encoding...")
            with open(self.onlyright_j, 'r', encoding='latin-1') as f:
                self.onlyright = json.load(f)

        try:
            with open(self.radio_j, 'r', encoding='utf-8-sig') as f:
                self.radio = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.radio_j}. Trying alternative encoding...")
            with open(self.radio_j, 'r', encoding='latin-1') as f:
                self.radio = json.load(f)

        try:
            with open(self.drinking_j, 'r', encoding='utf-8-sig') as f:
                self.drinking = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.drinking_j}. Trying alternative encoding...")
            with open(self.drinking_j, 'r', encoding='latin-1') as f:
                self.drinking = json.load(f)

        try:
            with open(self.reachside_j, 'r', encoding='utf-8-sig') as f:
                self.reachside = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.reachside_j}. Trying alternative encoding...")
            with open(self.reachside_j, 'r', encoding='latin-1') as f:
                self.reachside = json.load(f)

    def tf(self, to_tf, original, flag):
        scale = original[0] * original[1]
        if(flag == 0):
            return (to_tf[0] - original[0]) * scale
        elif(flag == 1):
            return (to_tf[1] - original[1]) * scale

        return 0

    def prepare_data(self, data):

        for item in data:

            right_elbow = item['pose']['pose'][1]

            features = {
                # "center_left_x": item['pose']['pose'][50][0],
                # "center_left_y": item['pose']['pose'][50][1],
                # "center_right_x": item['pose']['pose'][51][0],
                # "center_right_y": item['pose']['pose'][51][1],
                # "pose_0_x": item['pose']['pose'][0][0],
                # "pose_0_y": item['pose']['pose'][0][1],
                # "pose_1_x": item['pose']['pose'][1][0],
                # "pose_1_y": item['pose']['pose'][1][1],
                # "pose_2_x": item['pose']['pose'][2][0],
                # "pose_2_y": item['pose']['pose'][2][1],
                # "pose_3_x": item['pose']['pose'][3][0],
                # "pose_3_y": item['pose']['pose'][3][1],
                # "pose_4_x": item['pose']['pose'][4][0],
                # "pose_4_y": item['pose']['pose'][4][1],
                # "pose_5_x": item['pose']['pose'][5][0],
                # "pose_5_y": item['pose']['pose'][5][1],
                # # "json": item['json'],
                # # "frame": item['frame'],
                # "label": item['type'] # cambiar el label
                "center_left_x": self.tf(item['pose']['pose'][50], right_elbow, 0),
                "center_left_y": self.tf(item['pose']['pose'][50], right_elbow, 1),
                "center_right_x": self.tf(item['pose']['pose'][51], right_elbow, 0),
                "center_right_y": self.tf(item['pose']['pose'][51], right_elbow, 1),
                "pose_0_x": self.tf(item['pose']['pose'][0], right_elbow, 0),
                "pose_0_y": self.tf(item['pose']['pose'][0], right_elbow, 1),
                "pose_1_x": self.tf(item['pose']['pose'][1], right_elbow, 0),
                "pose_1_y": self.tf(item['pose']['pose'][1], right_elbow, 1),
                "pose_2_x": self.tf(item['pose']['pose'][2], right_elbow, 0),
                "pose_2_y": self.tf(item['pose']['pose'][2], right_elbow, 1),
                "pose_3_x": self.tf(item['pose']['pose'][3], right_elbow, 0),
                "pose_3_y": self.tf(item['pose']['pose'][3], right_elbow, 1),
                "pose_4_x": self.tf(item['pose']['pose'][4], right_elbow, 0),
                "pose_4_y": self.tf(item['pose']['pose'][4], right_elbow, 1),
                "pose_5_x": self.tf(item['pose']['pose'][5], right_elbow, 0),
                "pose_5_y": self.tf(item['pose']['pose'][5], right_elbow, 1),
                # "json": item['json'],
                # "frame": item['frame'],
                "label": item['type'] # cambiar el label
            }

            # print(features)
            # print("--------------------------------------------------")
            
            self.rows.append(features)

class videoReconstructor:
    def __init__(self, json_1, json_2, json_3, json_4, video_1, video_2, video_3):
        self.files = [json_1, json_2, json_3, json_4]
        self.video_paths = [video_1, video_2, video_3]
        self.actions = self.load_actions_from_json()

        self.data_hands = []
        self.data_face = []
        self.data_pose = []

        self.counter_goods = 0
        self.counter_total = 0

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
            (12, 14), (14, 16), (11, 13), (13, 15),
        ]

    def tf(self, data_2_tf, point_of_ref, flag):
        scale = point_of_ref[0] * point_of_ref[1]
        if (flag == 0):
            # print((data_2_tf[0] - point_of_ref[0]) * scale)
            return (data_2_tf[0] - point_of_ref[0]) * scale
        elif (flag == 1):
            return (data_2_tf[1] - point_of_ref[1]) * scale
        
        return point_of_ref

    def prepare_prediction(self, data):

        right_elbow = data['pose'][1]

        features = {
            # "center_left_x": data['pose'][50][0],
            # "center_left_y": data['pose'][50][1],
            # "center_right_x": data['pose'][51][0],
            # "center_right_y": data['pose'][51][1],
            # "pose_0_x": data['pose'][0][0],
            # "pose_0_y": data['pose'][0][1],
            # "pose_1_x": data['pose'][1][0],
            # "pose_1_y": data['pose'][1][1],
            # "pose_2_x": data['pose'][2][0],
            # "pose_2_y": data['pose'][2][1],
            # "pose_3_x": data['pose'][3][0],
            # "pose_3_y": data['pose'][3][1],
            # "pose_4_x": data['pose'][4][0],
            # "pose_4_y": data['pose'][4][1],
            # "pose_5_x": data['pose'][5][0],
            # "pose_5_y": data['pose'][5][1]
            "center_left_x": self.tf(data['pose'][50], right_elbow, 0),
            "center_left_y": self.tf(data['pose'][50], right_elbow, 1),
            "center_right_x": self.tf(data['pose'][51], right_elbow, 0),
            "center_right_y": self.tf(data['pose'][51], right_elbow, 1),
            "pose_0_x": self.tf(data['pose'][0], right_elbow, 0),
            "pose_0_y": self.tf(data['pose'][0], right_elbow, 1),
            "pose_1_x": self.tf(data['pose'][1], right_elbow, 0),
            "pose_1_y": self.tf(data['pose'][1], right_elbow, 1),
            "pose_2_x": self.tf(data['pose'][2], right_elbow, 0),
            "pose_2_y": self.tf(data['pose'][2], right_elbow, 1),
            "pose_3_x": self.tf(data['pose'][3], right_elbow, 0),
            "pose_3_y": self.tf(data['pose'][3], right_elbow, 1),
            "pose_4_x": self.tf(data['pose'][4], right_elbow, 0),
            "pose_4_y": self.tf(data['pose'][4], right_elbow, 1),
            "pose_5_x": self.tf(data['pose'][5], right_elbow, 0),
            "pose_5_y": self.tf(data['pose'][5], right_elbow, 1)        }

        return features

    def get_files(self):
        try:
            if os.path.isdir(self.directory_path):
                self.files = [file for file in os.listdir(self.directory_path)
                                 if os.path.isfile(os.path.join(self.directory_path, file))]
            else:
                print(f"{self.directory_path} no es un directorio válido.")
        except Exception as e:
            print(f"Error al acceder al directorio: {e}")

    def load_actions_from_json(self):
        try:
            with open(self.files[0], 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.files[0]}. Trying alternative encoding...")
            with open(self.files[0], 'r', encoding='latin-1') as f:
                data = json.load(f)

        actions = {}

        for frame_id, frame_data in data["openlabel"]["actions"].items():
            if "type" in frame_data:
                for frame_interval in frame_data["frame_intervals"]:
                    frame_start = frame_interval["frame_start"]
                    frame_end = frame_interval["frame_end"]
                    for frame in range(frame_start, frame_end + 1):
                        if frame not in actions:
                            actions[frame] = []
                        if frame_data["type"] not in actions[frame]:
                            actions[frame].append(frame_data["type"])

        for frame_id, frame_data in data["openlabel"]["streams"].items():
            if "face_camera" in frame_id:
                self.face_sync = frame_data["stream_properties"]["sync"]["frame_shift"]
                print("Face sync: ", self.face_sync)
            elif "hands_camera" in frame_id:
                self.hands_sync = frame_data["stream_properties"]["sync"]["frame_shift"]
                print("Hands sync: ", self.hands_sync)
            elif "body_camera" in frame_id:
                self.pose_sync = frame_data["stream_properties"]["sync"]["frame_shift"]
                print("Pose sync: ", self.pose_sync)

        return actions

    def open_jsons(self):
        try:
            with open(self.files[1], 'r', encoding='utf-8-sig') as f:
                self.data_hands = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.files[1]}. Trying alternative encoding...")
            with open(self.files[1], 'r', encoding='latin-1') as f:
                self.data_hands = json.load(f)

        try:
            with open(self.files[2], 'r', encoding='utf-8-sig') as f:
                self.data_pose = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.files[2]}. Trying alternative encoding...")
            with open(self.files[2], 'r', encoding='latin-1') as f:
                self.data_pose = json.load(f)

        try:
            with open(self.files[3], 'r', encoding='utf-8-sig') as f:
                self.data_face = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.files[3]}. Trying alternative encoding...")
            with open(self.files[3], 'r', encoding='latin-1') as f:
                self.data_face = json.load(f)

    def reconstruct(self, video_paths):
        caps = [cv2.VideoCapture(video_path) for video_path in video_paths]


        if not all([cap.isOpened() for cap in caps]):
            print("No se pudo abrir uno o más videos.")
            return

        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = caps[0].get(cv2.CAP_PROP_FPS)

        # Reduce the size of each frame
        reduced_width = width // 2
        reduced_height = height // 2
        output_size = (reduced_width * 2, reduced_height * 2)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('combined_video.mp4', fourcc, fps, output_size)

        frame_number = 0
        video_started = [False, False, False]  # Flags to track when each video starts

        init_time = time.time()
        total_time = 0
        last_prediction = None
        consecutive_count = 0
        previous_prediction_s = ""

        needed_consecutive = 15
        now_pred = []
        consecutive_actions = {}
        missing_actions = {}

        while all([cap.isOpened() for cap in caps]):

            prediction_s = ""
            Y_pred_prob_percent = []

            frames = []
            cap_number = 0
            for cap in caps:
                # If the video has already started, read it normally
                if video_started[cap_number]:
                    success, frame = cap.read()
                    if not success:
                        break
                else:
                    # If it has not started, it pauses in black until the synchronization is complete
                    frame = np.zeros((height, width, 3), dtype=np.uint8)

                black_frame = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

                if cap_number == 0:  # First video (pose)
                    if frame_number >= self.pose_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True  # Marks the video as started

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        features = self.prepare_prediction(self.data_pose["iterations"][frame_number - self.pose_sync])
                        features_df = pd.DataFrame([[features[name] for name in features.keys()]], columns=features.keys())

                        feature_names = list(features_df.columns)

                        features_df = pd.DataFrame([[features[name] for name in feature_names]], columns=feature_names)

                        prediction = model.predict(features_df)
                        Y_pred_prob = model.predict_proba(features_df)
                        Y_pred_prob_percent = (Y_pred_prob * 100).round(2) 
                        np.set_printoptions(suppress=True, precision=2)

                        now_pred = []
                        prediction = prediction[0]


                        for idx, predi in enumerate(prediction):
                            if idx == 0 and predi == 1:
                                now_pred.append("hands_using_wheel/both")
                            elif idx == 1 and predi == 1:
                                now_pred.append("hands_using_wheel/only_left")
                            elif idx == 2 and predi == 1:
                                now_pred.append("hands_using_wheel/only_right")
                            elif idx == 3 and predi == 1:
                                now_pred.append("driver_actions/radio")
                            elif idx == 4 and predi == 1:
                                now_pred.append("driver_actions/drinking")
                            elif idx == 5 and predi == 1:
                                now_pred.append("driver_actions/reach_side")

                        new_actions = []  

                        print(now_pred)

                        for action in now_pred:
                            if action in consecutive_actions:
                                consecutive_actions[action] += 1
                            else:
                                consecutive_actions[action] = 1
                            if consecutive_actions[action] >= needed_consecutive:
                                print(consecutive_actions[action])
                                new_actions.append(action)

                        print(new_actions)

                        for action in list(consecutive_actions.keys()):
                            if action not in now_pred:
                                if action in missing_actions:
                                    missing_actions[action] += 1
                                else:
                                    if(consecutive_actions[action] < needed_consecutive):
                                        del consecutive_actions[action] 
                                        break
                                    else:
                                        missing_actions[action] = 1  

                                if missing_actions[action] >= needed_consecutive:
                                    del consecutive_actions[action] 
                                    del missing_actions[action] 
                                elif(missing_actions[action] < needed_consecutive and consecutive_actions[action] >= needed_consecutive):
                                    new_actions.append(action)
                            else:
                                if action in missing_actions:
                                    del missing_actions[action]  

                        prediction_s = new_actions  
                        print(prediction_s)

                        self.paint_frame(frame, frame_number - self.pose_sync, "pose")
                        frame = cv2.resize(frame, (reduced_width, reduced_height))
                    else:
                        frame = black_frame  # Black till synchronization of pose_sync

                elif cap_number == 1:  # Second video (hands)
                    if frame_number >= self.hands_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # print("llamando a paint_frame")
                        self.paint_frame(frame, frame_number - self.hands_sync , "hands")
                        frame = cv2.resize(frame, (reduced_width, reduced_height))
                    else:
                        frame = black_frame  # Black till synchronization of hands_sync

                elif cap_number == 2:  # Third video (face)
                    if frame_number >= self.face_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.paint_frame(frame, frame_number - self.face_sync, "face")
                        frame = cv2.resize(frame, (reduced_width, reduced_height))
                    else:
                        frame = black_frame  # Black till synchronization of face_sync

                frames.append(frame)
                cap_number += 1

            if len(frames) != len(caps):
                break

            combined_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

            # Each frame in its quadrant
            combined_frame[0:reduced_height, 0:reduced_width] = frames[0]  # First video
            if len(frames) > 1:
                combined_frame[0:reduced_height, reduced_width:reduced_width*2] = frames[1]  # Second video
            if len(frames) > 2:
                combined_frame[reduced_height:reduced_height*2, 0:reduced_width] = frames[2]  # Third video

            # Blank space for actions
            if frame_number in self.actions:
                actions = self.actions[frame_number]
                # actions = action['type'].split()

                for pred_act in prediction_s:
                    if pred_act in actions:
                        # counter_goods += 1
                        self.counter_goods += 1
                        # self.counter_total += 1
                    # else:
                    self.counter_total += 1

                # if prediction_s == "":
                #     self.counter_total += 1

                if len(prediction_s) == 0:
                    self.counter_total += 1

                valid_actions = [act for act in actions if act.startswith("driver_actions") or act.startswith("hands_using_wheel")]
                # valid_actions = [act for act in actions if act.startswith("hands_using_wheel")]


                y_offset = 30
                start_y = height // 2 + y_offset 
                line_height = 20  
                x_offset = width // 2 + 10 

                for i, valid_action in enumerate(valid_actions):
                    cv2.putText(
                        combined_frame,
                        valid_action,
                        (x_offset, start_y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

                prediction_start_y = start_y + len(valid_actions) * line_height 
                
                # if prediction_s == "hands_using_wheel/both ":
                #     prediction_s = "driver_actions/safe_drive"
                
                text = " ".join(prediction_s)  # Une las palabras con un espacio
                cv2.putText(
                    combined_frame,
                    text,
                    (x_offset, prediction_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )


                if len(Y_pred_prob_percent) > 0:
                    probabilities_start_y = prediction_start_y + line_height
                    cv2.putText(
                        combined_frame,
                        "Probabilidades:",
                        (x_offset, probabilities_start_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                    
                    probability_classes = ["Both", "Left", "Right", "Radio", "Drinking", "Reachside"]
                    for j, class_name in enumerate(probability_classes):
                        prob_text = f"{class_name}: {Y_pred_prob_percent[0][j]}%"
                        cv2.putText(
                            combined_frame,
                            prob_text,
                            (x_offset, probabilities_start_y + (j + 1) * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                percent_correct = (self.counter_goods / self.counter_total) * 100
                bottom_y = combined_frame.shape[0] - 10  # Descuento de margen inferior

                cv2.putText(
                    combined_frame,
                    f"Porcentaje de aciertos: {percent_correct:.2f}%",
                    (x_offset, bottom_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                frame_time = time.time() - init_time
                total_time += frame_time
                init_time = time.time()

                print(f"Frame {frame_number} - Tiempo de procesamiento: {frame_time:.2f} s")

            cv2.imshow('Combined Video', combined_frame)
            out.write(combined_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_number += 1
            

        for cap in caps:
            cap.release()
        out.release()
        cv2.destroyAllWindows()

    def paint_frame(self, frame, frame_number, json):
        # print("Frame: ", frame_number)
        if json == "hands":
            data = self.data_hands
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        for x, y, z in iterations["hands"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        for x, y in iterations["centers"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (255, 0, 0), 30)

                        hand_left = iterations["hands"][:21]
                        hand_right = iterations["hands"][21:]

                        self.draw_connections(frame, hand_left, self.HANDS_CONNECTION)
                        self.draw_connections(frame, hand_right, self.HANDS_CONNECTION)

        elif json == "pose":
            data = self.data_pose
            # print("Pose")
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        for x, y, z in iterations["pose"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        pose_data = iterations["pose"][:8]
                        left_hand = iterations["pose"][8:29]
                        right_hand = iterations["pose"][29:50]
                        left_center = iterations["pose"][50]
                        right_center = iterations["pose"][51]

                        cv2.circle(frame, (int(left_center[0] * frame.shape[1]), int(left_center[1] * frame.shape[0])), 5, (255, 0, 0), 30)
                        cv2.circle(frame, (int(right_center[0] * frame.shape[1]), int(right_center[1] * frame.shape[0])), 5, (255, 0, 0), 30)

                        self.draw_connections(frame, pose_data, self.POSE_CONNECTION)
                        self.draw_connections(frame, left_hand, self.HANDS_CONNECTION)
                        self.draw_connections(frame, right_hand, self.HANDS_CONNECTION)

        elif json == "face":
            data = self.data_face
            # print("Face")
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:

                        for x, y, indx in iterations["face"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                            # print("Face: ", x, y)
    
                        x, y = iterations["gaze"][0]
                        x2, y2 = iterations["gaze"][1]

                        x, y = int(x), int(y)
                        x2, y2 = int(x2), int(y2)
                        
                        cv2.line(frame, (x, y), (x2, y2), (0, 0, 255), 7)

                        self.draw_connections(frame, iterations["face"], self.FACE_CONNECTION)

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

# Convertir etiquetas en columnas multilabel
def convert_to_multilabel(df):
    df['label'] = df['label'].map(label_mapping)  # Mapear nombres de etiquetas
    multilabels = pd.get_dummies(df['label'])  # Convertir a columnas binarias
    df = pd.concat([df, multilabels], axis=1)  # Unir al dataset
    df.drop(columns=['label'], inplace=True)  # Eliminar columna original
    return df

json_files_1 = [  '/home/arantxa/tfg/tfgData/data_real_car/hands_using_wheel_both.json', 
                '/home/arantxa/tfg/tfgData/data_real_car/hands_using_wheel_only_left.json', 
                '/home/arantxa/tfg/tfgData/data_real_car/hands_using_wheel_only_right.json',
                '/home/arantxa/tfg/tfgData/data_real_car/driver_actions_radio.json', 
                '/home/arantxa/tfg/tfgData/data_real_car/driver_actions_drinking.json', 
                '/home/arantxa/tfg/tfgData/data_real_car/driver_actions_reach_side.json']

# json_files_2 = [  '/home/arantxa/tfg/tfgData/data_test/hands_using_wheel_both.json', 
#                 '/home/arantxa/tfg/tfgData/data_test/hands_using_wheel_only_left.json', 
#                 '/home/arantxa/tfg/tfgData/data_test/hands_using_wheel_only_right.json',
#                 '/home/arantxa/tfg/tfgData/data_test/driver_actions_radio.json', 
#                 '/home/arantxa/tfg/tfgData/data_test/driver_actions_drinking.json', 
#                 '/home/arantxa/tfg/tfgData/data_test/driver_actions_reach_side.json']

# Mapeo de etiquetas a nombres más adecuados para multilabel
label_mapping = {
    "hands_using_wheel/both": "both_hands",
    "hands_using_wheel/only_left": "left_hand",
    "hands_using_wheel/only_right": "right_hand",
    "driver_actions/radio": "radio",
    "driver_actions/drinking": "drinking",
    "driver_actions/reach_side": "reach_side"
}

# Cargar datos de entrenamiento
RandomForest_performer = RandomForest(*json_files_1)
RandomForest_performer.open_jsons()
for dataset in [RandomForest_performer.both, RandomForest_performer.onlyleft, RandomForest_performer.onlyright, 
                RandomForest_performer.radio, RandomForest_performer.drinking, RandomForest_performer.reachside]:
    RandomForest_performer.prepare_data(dataset)

dataset = pd.DataFrame(RandomForest_performer.rows)
dataset = convert_to_multilabel(dataset)

X = dataset.iloc[:, :-6] # todo menos etiquetas
Y = dataset.iloc[:, -6:] # etiquetas

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.2, random_state=1, stratify=Y) 

X_train = pd.DataFrame(X_train, columns=['center_left_x', 'center_left_y', 'center_right_x', 'center_right_y', 'pose_0_x', 'pose_0_y', 'pose_1_x', 'pose_1_y', 'pose_2_x', 'pose_2_y', 'pose_3_x', 'pose_3_y', 'pose_4_x', 'pose_4_y', 'pose_5_x', 'pose_5_y'])


train_class_counts = Y_train.sum(axis=0)
test_class_counts = Y_test.sum(axis=0)

print("Distribución de clases en el conjunto de entrenamiento:")
print(train_class_counts)

print("\nDistribución de clases en el conjunto de prueba:")
print(test_class_counts)


Y_train_bin = np.array(list(Y_train)) 
Y_test_bin = np.array(list(Y_test)) 

# Definir y entrenar modelo multilabel
model = OneVsRestClassifier(RandomForestClassifier(random_state=1))
model.fit(X_train, Y_train)

# Evaluación
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

class_labels = ['Both', 'Left', 'Right', 'Radio', 'Drinking', 'Reachside']

# print(Y_train)


n_labels = Y_train.shape[1]
global_conf_matrix = np.zeros((n_labels, n_labels))

no_predictions_train = 0
yes_predictions_train = 0

for i in range(Y_train.shape[0]): 
    for j in range(n_labels): 
        if Y_train.iloc[i, j] == 1:
            for k in range(n_labels):  
                global_conf_matrix[j, k] += train_preds[i, k]  

            if np.sum(train_preds[i]) == 0:
                no_predictions_train += 1
            else:
                yes_predictions_train += 1

print(f"Número de ejemplos sin predicciones: {no_predictions_train}")
print(f"Número de ejemplos con predicciones: {yes_predictions_train}")

row_sums = global_conf_matrix.sum(axis=1, keepdims=True) 
global_conf_matrix_normalized = global_conf_matrix / row_sums

plt.figure(figsize=(10, 7))
sns.heatmap(global_conf_matrix_normalized.T, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels,  
            yticklabels=class_labels)  
plt.xlabel("Etiqueta Real")
plt.ylabel("Predicción")
plt.title("Matriz de Confusión de train")
# plt.show()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"train_{timestamp}.png"
plt.savefig(filename)  

global_conf_matrix = np.zeros((n_labels, n_labels))

no_predictions_test = 0
yes_predictions_test = 0

for i in range(Y_test.shape[0]): 
    for j in range(n_labels): 
        if Y_test.iloc[i, j] == 1: 
            for k in range(n_labels): 
                global_conf_matrix[j, k] += test_preds[i, k]  
            
            if np.sum(test_preds[i]) == 0:
                no_predictions_test += 1
            else:
                yes_predictions_test += 1

print(f"Número de ejemplos sin predicciones: {no_predictions_test}")
print(f"Número de ejemplos con predicciones: {yes_predictions_test}")
            
row_sums = global_conf_matrix.sum(axis=1, keepdims=True) 
global_conf_matrix_normalized = global_conf_matrix / row_sums

plt.figure(figsize=(10, 7))
sns.heatmap(global_conf_matrix_normalized.T, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels,  # Usar las etiquetas personalizadas
            yticklabels=class_labels)  
plt.xlabel("Etiqueta Real")
plt.ylabel("Predicción")
plt.title("Matriz de Confusión de test")
# plt.show()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"test_{timestamp}.png"
plt.savefig(filename)  

sys.argv = [sys.argv[0], '/home/arantxa/tfg/tfgData/jsons_dir/1_s1/frames.json',
            '/home/arantxa/tfg/tfgData/jsons_dir/1_s1/hands.json',
            '/home/arantxa/tfg/tfgData/jsons_dir/1_s1/pose.json',
            '/home/arantxa/tfg/tfgData/jsons_dir/1_s1/face.json',
            '/home/arantxa/tfg/tfgData/jsons_dir/1_s1/pose.mp4',
            '/home/arantxa/tfg/tfgData/jsons_dir/1_s1/hands.mp4',
            '/home/arantxa/tfg/tfgData/jsons_dir/1_s1/face.mp4']

vr = videoReconstructor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
video_paths = [sys.argv[5], sys.argv[6], sys.argv[7]]
vr.open_jsons()
vr.reconstruct(video_paths)

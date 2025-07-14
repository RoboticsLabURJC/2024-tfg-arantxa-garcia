import os
import json
import time
import cv2
import numpy as np
import pandas as pd
from actions_model import train_actions_model
from gaze_model import train_gaze_model
from phone_model import PhoneDetector
import sys
import mediapipe as mp
import gaze
import copy
from collections import deque
import joblib

def tf(to_tf, original, axis):
    return to_tf[axis] - original[axis]

def calc_ang(p1, p2, center):
    u, v = np.array(p1) - np.array(center), np.array(p2) - np.array(center)
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clip to avoid domain errors

class videoReconstructor:

    def __init__(self, json_1, json_2, json_3, json_4, video_1, video_2, video_3, model_actions, model_gaze, phone_detector): #, steering_model, le_actions, le_steering):
        self.files = [json_1, json_2, json_3, json_4]
        self.video_paths = [video_1, video_2, video_3]
        self.actions = self.load_actions_from_json()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands 
        self.hands_pose = mp.solutions.hands.Hands() 
        self.hands_only = mp.solutions.hands.Hands() 
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.data_hands = []
        self.data_face = []
        self.data_pose = []

        self.counter_goods = 0
        self.counter_total = 0

        self.model_actions = model_actions
        self.model_gaze = model_gaze
        self.phone_detector = phone_detector

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

    def calculate_center_of_mass(self, landmarks):
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]

        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)

        return center_x, center_y

    def normalize_gaze(self, gaze, frame, flag):
        if flag == 0:
            return (gaze[0] - frame.shape[1] / 2) / (frame.shape[1] / 2)
        elif flag == 1:
            return (gaze[1] - frame.shape[0] / 2) / (frame.shape[0] / 2)

    def prepare_prediction_action(self, data):

        right_elbow = data['pose'][1]

        features_action = {
            "center_left_x": tf(data['pose'][50], right_elbow, 0),
            "center_left_y": tf(data['pose'][50], right_elbow, 1),
            "center_right_x": tf(data['pose'][51], right_elbow, 0),
            "center_right_y": tf(data['pose'][51], right_elbow, 1),
            "pose_0_x": tf(data['pose'][0], right_elbow, 0),
            "pose_0_y": tf(data['pose'][0], right_elbow, 1),
            "pose_1_x": tf(data['pose'][1], right_elbow, 0),
            "pose_1_y": tf(data['pose'][1], right_elbow, 1),
            "pose_2_x": tf(data['pose'][2], right_elbow, 0),
            "pose_2_y": tf(data['pose'][2], right_elbow, 1),
            "pose_3_x": tf(data['pose'][3], right_elbow, 0),
            "pose_3_y": tf(data['pose'][3], right_elbow, 1),
            "pose_4_x": tf(data['pose'][4], right_elbow, 0),
            "pose_4_y": tf(data['pose'][4], right_elbow, 1),
            "pose_5_x": tf(data['pose'][5], right_elbow, 0),
            "pose_5_y": tf(data['pose'][5], right_elbow, 1),
            "elbow_right": calc_ang(data['pose'][1], data['pose'][5], data['pose'][3]),
            "elbow_left": calc_ang(data['pose'][0], data['pose'][4], data['pose'][2]),
            "wrist_right": calc_ang(data['pose'][3], data['pose'][51], data['pose'][5]),
            "wrist_left": calc_ang(data['pose'][2], data['pose'][50], data['pose'][4]),
            }
        
        return features_action
    
    def prepare_prediction_gaze(self, data, frame):
        nose = data['face'][1]
        features_gaze  = {
            "face_0_x": tf(data['face'][0], nose, 0),
            "face_0_y": tf(data['face'][0], nose, 1),
            "face_1_x": tf(data['face'][1], nose, 0),
            "face_1_y": tf(data['face'][1], nose, 1),
            "face_2_x": tf(data['face'][2], nose, 0),
            "face_2_y": tf(data['face'][2], nose, 1),
            "face_3_x": tf(data['face'][3], nose, 0),
            "face_3_y": tf(data['face'][3], nose, 1),            
            "face_4_x": tf(data['face'][4], nose, 0),
            "face_4_y": tf(data['face'][4], nose, 1),
            "face_5_x": tf(data['face'][5], nose, 0),
            "face_5_y": tf(data['face'][5], nose, 1),
            "face_6_x": tf(data['face'][6], nose, 0),
            "face_6_y": tf(data['face'][6], nose, 1),
            "face_7_x": tf(data['face'][7], nose, 0),
            "face_7_y": tf(data['face'][7], nose, 1),
            "face_8_x": tf(data['face'][8], nose, 0),
            "face_8_y": tf(data['face'][8], nose, 1),
            "face_9_x": tf(data['face'][9], nose, 0),
            "face_9_y": tf(data['face'][9], nose, 1),
            "face_10_x": tf(data['face'][10], nose, 0),
            "face_10_y": tf(data['face'][10], nose, 1),
            "face_11_x": tf(data['face'][11], nose, 0),
            "face_11_y": tf(data['face'][11], nose, 1),
            "face_12_x": tf(data['face'][12], nose, 0),
            "face_12_y": tf(data['face'][12], nose, 1),
            "face_13_x": tf(data['face'][13], nose, 0),
            "face_13_y": tf(data['face'][13], nose, 1),
            "face_14_x": tf(data['face'][14], nose, 0),
            "face_14_y": tf(data['face'][14], nose, 1),
            "face_15_x": tf(data['face'][15], nose, 0),
            "face_15_y": tf(data['face'][15], nose, 1),
            "face_16_x": tf(data['face'][16], nose, 0),
            "face_16_y": tf(data['face'][16], nose, 1),
            "face_17_x": tf(data['face'][17], nose, 0),
            "face_17_y": tf(data['face'][17], nose, 1),
            "face_18_x": tf(data['face'][18], nose, 0),
            "face_18_y": tf(data['face'][18], nose, 1),
            "face_19_x": tf(data['face'][19], nose, 0),
            "face_19_y": tf(data['face'][19], nose, 1),
            "face_20_x": tf(data['face'][20], nose, 0),
            "face_20_y": tf(data['face'][20], nose, 1),
            "face_21_x": tf(data['face'][21], nose, 0),
            "face_21_y": tf(data['face'][21], nose, 1),
            "face_22_x": tf(data['face'][22], nose, 0),
            "face_22_y": tf(data['face'][22], nose, 1),
            "face_23_x": tf(data['face'][23], nose, 0),
            "face_23_y": tf(data['face'][23], nose, 1),
            "face_24_x": tf(data['face'][24], nose, 0),
            "face_24_y": tf(data['face'][24], nose, 1),
            "face_25_x": tf(data['face'][25], nose, 0),
            "face_25_y": tf(data['face'][25], nose, 1),
            "face_26_x": tf(data['face'][26], nose, 0),
            "face_26_y": tf(data['face'][26], nose, 1),
            "gaze_x1": self.normalize_gaze(data['gaze'][0], frame, 0), 
            "gaze_y1": self.normalize_gaze(data['gaze'][0], frame, 1),
            "gaze_x2": self.normalize_gaze(data['gaze'][1], frame, 0),
            "gaze_y2": self.normalize_gaze(data['gaze'][1], frame, 1)  
        }
        return features_gaze

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

    def update_pose_json(self, results_pose, results_hands):
        # Indices in self.data_pose['pose']
        # -----------------------------------
        #  0 -  7   -> Body keypoints (shoulders, elbows, wrists, hips)
        #  8 - 28  -> Left hand keypoints
        # 29 - 49  -> Right hand keypoints
        #    50    -> Center of mass of the left hand
        #    51    -> Center of mass of the right hand

        self.data_pose['pose'] = [[0, 0, idx] for idx in range(53)]

        if results_pose.pose_landmarks:
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if idx == 11:
                    self.data_pose['pose'][0] = [landmark.x, landmark.y, idx]
                elif idx == 12:
                    self.data_pose['pose'][1] = [landmark.x, landmark.y, idx]
                elif idx == 13:
                    self.data_pose['pose'][2] = [landmark.x, landmark.y, idx]
                elif idx == 14:
                    self.data_pose['pose'][3] = [landmark.x, landmark.y, idx]
                elif idx == 15:
                    self.data_pose['pose'][4] = [landmark.x, landmark.y, idx]
                elif idx == 16:
                    self.data_pose['pose'][5] = [landmark.x, landmark.y, idx]
                elif idx == 23:
                    self.data_pose['pose'][6] = [landmark.x, landmark.y, idx]
                elif idx == 24:
                    self.data_pose['pose'][7] = [landmark.x, landmark.y, idx]

        left_hand_landmarks = None
        right_hand_landmarks = None

        if results_hands.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                handedness = results_hands.multi_handedness[hand_idx].classification[0].label
                if handedness == 'Left':
                    left_hand_landmarks = hand_landmarks
                else:
                    right_hand_landmarks = hand_landmarks

        if left_hand_landmarks:
            for idx, landmark in enumerate(left_hand_landmarks.landmark):
                self.data_pose['pose'][idx + 21 + 8] = [landmark.x, landmark.y, idx]

            left_center_x, left_center_y = self.calculate_center_of_mass(left_hand_landmarks.landmark)
            self.data_pose['pose'][51] = [left_center_x, left_center_y, 50]

        if right_hand_landmarks:
            for idx, landmark in enumerate(right_hand_landmarks.landmark):
                self.data_pose['pose'][idx + 8] = [landmark.x, landmark.y, idx]

            right_center_x, right_center_y = self.calculate_center_of_mass(right_hand_landmarks.landmark)
            self.data_pose['pose'][50] = [right_center_x, right_center_y, 51]

        new_iteration = {
            'pose': self.data_pose['pose'].copy()
        }

        return new_iteration

    def update_face_json(self, results_face, frame):
        self.data_face['face'] = [[0.0, 0.0, i] for i in range(28)]  
        self.data_face['gaze'] = [[0.0, 0.0] for _ in range(2)]

        if results_face.multi_face_landmarks:
            self.data_face['face'] = [[0.0, 0.0, i] for i in range(28)]
            self.data_face['gaze'] = [[0.0, 0.0] for _ in range(2)]

            p1, p2 = gaze.gaze(frame, results_face.multi_face_landmarks[0])
            self.data_face['gaze'][0] = [p1[0], p1[1]] 
            self.data_face['gaze'][1] = [p2[0], p2[1]]  

            for face_idx, face_landmarks in enumerate(results_face.multi_face_landmarks):
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx == 0:
                        self.data_face['face'][0] = [landmark.x, landmark.y, idx]
                    elif idx == 4:
                        self.data_face['face'][1] = [landmark.x, landmark.y, idx]
                    elif idx == 17:
                        self.data_face['face'][2] = [landmark.x, landmark.y, idx]
                    elif idx == 46:
                        self.data_face['face'][3] = [landmark.x, landmark.y, idx]
                    elif idx == 48:
                        self.data_face['face'][4] = [landmark.x, landmark.y, idx]
                    elif idx == 50:
                        self.data_face['face'][5] = [landmark.x, landmark.y, idx]
                    elif idx == 61:
                        self.data_face['face'][6] = [landmark.x, landmark.y, idx]
                    elif idx == 105:
                        self.data_face['face'][7] = [landmark.x, landmark.y, idx]
                    elif idx == 107:
                        self.data_face['face'][8] = [landmark.x, landmark.y, idx]
                    elif idx == 122:
                        self.data_face['face'][9] = [landmark.x, landmark.y, idx]
                    elif idx == 130:
                        self.data_face['face'][10] = [landmark.x, landmark.y, idx]
                    elif idx == 133:
                        self.data_face['face'][11] = [landmark.x, landmark.y, idx]
                    elif idx == 145:
                        self.data_face['face'][12] = [landmark.x, landmark.y, idx]
                    elif idx == 159:
                        self.data_face['face'][13] = [landmark.x, landmark.y, idx]
                    elif idx == 206:
                        self.data_face['face'][14] = [landmark.x, landmark.y, idx]
                    elif idx == 276:
                        self.data_face['face'][15] = [landmark.x, landmark.y, idx]
                    elif idx == 278:
                        self.data_face['face'][16] = [landmark.x, landmark.y, idx]
                    elif idx == 280:
                        self.data_face['face'][17] = [landmark.x, landmark.y, idx]
                    elif idx == 291:
                        self.data_face['face'][18] = [landmark.x, landmark.y, idx]
                    elif idx == 334:
                        self.data_face['face'][19] = [landmark.x, landmark.y, idx]
                    elif idx == 336:
                        self.data_face['face'][20] = [landmark.x, landmark.y, idx]
                    elif idx == 351:
                        self.data_face['face'][21] = [landmark.x, landmark.y, idx]
                    elif idx == 359:
                        self.data_face['face'][22] = [landmark.x, landmark.y, idx]
                    elif idx == 362:
                        self.data_face['face'][23] = [landmark.x, landmark.y, idx]
                    elif idx == 374:
                        self.data_face['face'][24] = [landmark.x, landmark.y, idx]
                    elif idx == 386:
                        self.data_face['face'][25] = [landmark.x, landmark.y, idx]
                    elif idx == 426:
                        self.data_face['face'][26] = [landmark.x, landmark.y, idx]


        new_iteration = {
            'frame': len(self.data_face['iterations']),
            'face': copy.deepcopy(self.data_face['face']), 
            'gaze': copy.deepcopy(self.data_face['gaze'])
        }

        return new_iteration

    def reconstruct(self, video_paths):
        caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

        if not all([cap.isOpened() for cap in caps]):
            print("No se pudo abrir uno o más videos.")
            return

        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video dimensions: {width}x{height}")
        fps = caps[0].get(cv2.CAP_PROP_FPS)

        reduced_width = width // 2
        reduced_height = height // 2
        output_size = (reduced_width * 2, reduced_height * 2)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('combined_video.mp4', fourcc, fps, output_size)

        frame_number = 0
        video_started = [False] * len(caps)
        timeline = []

        phone_detected = False
        gaze_result = []
        last_20_frames_actions = []
        last_10_frames_phone = []

        while all([cap.isOpened() for cap in caps]):
            prediction_s = []
            prediction_gaze = []
            frames = []
            cap_number = 0

            for cap in caps:
                if video_started[cap_number]:
                    success, frame = cap.read()
                    if not success:
                        break
                else:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)

                black_frame = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

                if cap_number == 0 and frame_number >= self.pose_sync:
                    if not video_started[cap_number]:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        video_started[cap_number] = True

                    painted_frame = frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results_pose = self.pose.process(frame_rgb)
                    results_hands = self.hands_pose.process(frame_rgb)
                    data_pose = self.update_pose_json(results_pose, results_hands)
                    is_phone_now = self.phone_detector.detect(frame)

                    last_10_frames_phone.append(is_phone_now)

                    if len(last_10_frames_phone) > 5:
                        last_10_frames_phone.pop(0)

                    phone_count = sum(last_10_frames_phone)
                    if phone_count > 10:
                        phone_detected = True
                    else:
                        phone_detected = False

                    features = self.prepare_prediction_action(data_pose)

                    features_df = pd.DataFrame([[features[name] for name in features.keys()]],
                                                columns=features.keys())

                    probas = self.model_actions.predict_proba(features_df)[0]

                    print(probas)
                   
                    now_pred = []
                    if phone_detected == 1:
                        if probas[1] > 0.3:
                            now_pred = ["hands_using_wheel/only_left"]
                        elif probas[2] > 0.3:
                            now_pred = ["hands_using_wheel/only_right"]
                    else:
                        for idx, prob in enumerate(probas):
                            if prob > 0.3:
                                if idx == 0:
                                    now_pred.append("hands_using_wheel/both")
                                    break
                                elif idx == 1:
                                    now_pred.append("hands_using_wheel/only_left")
                                elif idx == 2:
                                    now_pred.append("hands_using_wheel/only_right")
                                elif idx == 3:
                                    now_pred.append("driver_actions/radio") 
                                elif idx == 4:
                                    now_pred.append("driver_actions/drinking")
                                elif idx == 5:
                                    now_pred.append("driver_actions/reach_side")

                    print(f"Frame {frame_number}: Actions prediction: {now_pred}")

                    last_20_frames_actions.append(now_pred)

                    if len(last_20_frames_actions) > 20:
                        last_20_frames_actions.pop(0)

                    action_counts = {}
                    for frame in last_20_frames_actions:
                        for action in frame:
                            if action in action_counts:
                                action_counts[action] += 1
                            else:
                                action_counts[action] = 1

                    prediction_s = [action for action, count in action_counts.items() if count > 10]

                    if not prediction_s:
                        prediction_s = [] 
                        
                    self.paint_frame(painted_frame, frame_number - self.pose_sync, "pose")
                    frame = cv2.resize(painted_frame, (reduced_width, reduced_height))

                elif cap_number == 0:
                    frame = black_frame

                elif cap_number == 1 and frame_number >= self.hands_sync:
                    if not video_started[cap_number]:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        video_started[cap_number] = True

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.paint_frame(frame, frame_number - self.hands_sync, "hands")
                    frame = cv2.resize(frame, (reduced_width, reduced_height))
                
                elif cap_number == 1:
                    frame = black_frame

                elif cap_number == 2 and frame_number >= self.face_sync:
                    if not video_started[cap_number]:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        video_started[cap_number] = True

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results_face = self.face_mesh.process(frame_rgb)
                    data_face = self.update_face_json(results_face, frame_rgb)
                    self.paint_frame(frame, frame_number - self.face_sync, "face")

                    features = self.prepare_prediction_gaze(data_face, frame_rgb)
                    features_df = pd.DataFrame([[features[name] for name in features.keys()]],
                                                columns=features.keys())

                    prediction_label = self.model_gaze.predict(features_df)[0]

                    gaze = f"gaze_zone/{prediction_label}"

                    prediction_gaze = [gaze] 
                    gaze_result = prediction_gaze
                    frame = cv2.resize(frame, (reduced_width, reduced_height))

                elif cap_number == 2:
                    frame = black_frame

                frames.append(frame)
                cap_number += 1

            if len(frames) != len(caps):
                break

            combined_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            combined_frame[0:reduced_height, 0:reduced_width] = frames[0]
            if len(frames) > 1:
                combined_frame[0:reduced_height, reduced_width:reduced_width*2] = frames[1]
            if len(frames) > 2:
                combined_frame[reduced_height:reduced_height*2, 0:reduced_width] = frames[2]

            if frame_number in self.actions:
                actions = self.actions[frame_number]
                for pred_act in prediction_s:
                    if pred_act in actions:
                        self.counter_goods += 1
                    self.counter_total += 1
                if len(prediction_s) == 0:
                    self.counter_total += 1

            # we will save the frame only if there is a prediction or phone detected or gaze result
            if prediction_s or phone_detected or gaze_result:
                timeline.append({
                    "frame": frame_number,
                    "timestamp": frame_number / fps,
                    "actions": prediction_s,
                    "phone": phone_detected,
                    "gaze": gaze_result
                })

            out.write(combined_frame)
            frame_number += 1

        with open("prediction_timeline.json", "w") as f:
            json.dump(timeline, f, indent=2)

        for cap in caps:
            cap.release()
        out.release()

    def paint_frame(self, frame, frame_number, json):
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
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:

                        for x, y, indx in iterations["face"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
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

phone_detector = PhoneDetector()
action_model = joblib.load('actions_model.pkl')
gaze_model = joblib.load('gaze_model.pkl')


sys.argv = [sys.argv[0], '1_s1/frames.json',
            '1_s1/hands.json',
            '1_s1/pose.json',
            '1_s1/face.json',
            '1_s1/pose.mp4',
            '1_s1/hands.mp4',
            '1_s1/face.mp4']

vr = videoReconstructor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], action_model, gaze_model, phone_detector) #, steering_model, le_steering, le_action)
video_paths = [sys.argv[5], sys.argv[6], sys.argv[7]]
vr.open_jsons()
vr.reconstruct(video_paths)
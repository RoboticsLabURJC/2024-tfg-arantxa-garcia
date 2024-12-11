import sys
import json
import numpy as np
import pandas as pd
import time
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

class SVM:
    def __init__(self, json_both, json_onlyleft, json_onlyright, json_radio, json_drinking, json_reachside, json_phonecallright):
        self.both_j = json_both
        self.onlyleft_j = json_onlyleft
        self.onlyright_j = json_onlyright
        self.radio_j = json_radio
        self.drinking_j = json_drinking
        self.reachside_j = json_reachside
        self.phonecallright_j = json_phonecallright

        self.both = []
        self.onlyleft = []
        self.onlyright = []
        self.radio = []
        self.drinking = []
        self.reachside = []
        self.phonecallright = []

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

        try:
            with open(self.phonecallright_j, 'r', encoding='utf-8-sig') as f:
                self.phonecallright = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.phonecallright_j}. Trying alternative encoding...")
            with open(self.phonecallright_j, 'r', encoding='latin-1') as f:
                self.phonecallright = json.load(f)

    def prepare_data(self, data):

        for item in data:

            features = {
                "center_left_x": item['pose']['pose'][50][0],
                "center_left_y": item['pose']['pose'][50][1],
                "center_right_x": item['pose']['pose'][51][0],
                "center_right_y": item['pose']['pose'][51][1],
                "pose_0_x": item['pose']['pose'][0][0],
                "pose_0_y": item['pose']['pose'][0][1],
                "pose_1_x": item['pose']['pose'][1][0],
                "pose_1_y": item['pose']['pose'][1][1],
                "pose_2_x": item['pose']['pose'][2][0],
                "pose_2_y": item['pose']['pose'][2][1],
                "pose_3_x": item['pose']['pose'][3][0],
                "pose_3_y": item['pose']['pose'][3][1],
                "pose_4_x": item['pose']['pose'][4][0],
                "pose_4_y": item['pose']['pose'][4][1],
                "pose_5_x": item['pose']['pose'][5][0],
                "pose_5_y": item['pose']['pose'][5][1],
                # "json": item['json'],
                # "frame": item['frame'],
                "label": item['type'] # cambiar el label
            }
            
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

    def prepare_prediction(self, data):

        features = {
            "center_left_x": data['pose'][50][0],
            "center_left_y": data['pose'][50][1],
            "center_right_x": data['pose'][51][0],
            "center_right_y": data['pose'][51][1],
            "pose_0_x": data['pose'][0][0],
            "pose_0_y": data['pose'][0][1],
            "pose_1_x": data['pose'][1][0],
            "pose_1_y": data['pose'][1][1],
            "pose_2_x": data['pose'][2][0],
            "pose_2_y": data['pose'][2][1],
            "pose_3_x": data['pose'][3][0],
            "pose_3_y": data['pose'][3][1],
            "pose_4_x": data['pose'][4][0],
            "pose_4_y": data['pose'][4][1],
            "pose_5_x": data['pose'][5][0],
            "pose_5_y": data['pose'][5][1]
        }

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
                        # print(frame)
                        features = self.prepare_prediction(self.data_pose["iterations"][frame_number - self.pose_sync])
                        # print(features)
                        prediction = multilabel_model.predict([list(features.values())])
                        Y_pred_prob = multilabel_model.predict_proba([list(features.values())])
                        Y_pred_prob_percent = (Y_pred_prob * 100).round(2) 
                        np.set_printoptions(suppress=True, precision=2)
                        print(prediction)

                        for n in range(len(prediction[0])):
                            if prediction[0][n] == 1:
                                if n == 0:
                                    prediction_s = prediction_s + "hands_using_wheel/both "
                                elif n == 1:
                                    prediction_s = prediction_s + "hands_using_wheel/only_left "
                                elif n == 2:
                                    prediction_s = prediction_s + "hands_using_wheel/only_right "
                                elif n == 3:
                                    prediction_s = prediction_s + "driver_actions/radio "
                                elif n == 4:
                                    prediction_s = prediction_s + "driving_actions/drinking "
                                elif n == 5:
                                    prediction_s = prediction_s + "driver_actions/reach Side "
                                elif n == 6:
                                    prediction_s = prediction_s + "driver_actions/phone Call Right "

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

                for pred_act in prediction_s.split():
                    if pred_act in actions:
                        # counter_goods += 1
                        self.counter_goods += 1
                        self.counter_total += 1
                    else:
                        self.counter_total += 1

                if prediction_s == "":
                    self.counter_total += 1

                valid_actions = [act for act in actions if act.startswith("driver_actions") or act.startswith("hands_using_wheel")]

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
                cv2.putText(
                    combined_frame,
                    prediction_s,
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
                    
                    probability_classes = ["Both", "Left", "Right", "Radio", "Drinking", "Reachside", "Phonecallr"]
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

json_files = [  '/home/arantxa/universidad/TFG/src/balanced_data/hands_using_wheel_both.json', 
                '/home/arantxa/universidad/TFG/src/balanced_data/hands_using_wheel_only_left.json', 
                '/home/arantxa/universidad/TFG/src/balanced_data/hands_using_wheel_only_right.json',
                '/home/arantxa/universidad/TFG/src/balanced_data/driver_actions_radio.json', 
                '/home/arantxa/universidad/TFG/src/balanced_data/driver_actions_drinking.json', 
                '/home/arantxa/universidad/TFG/src/balanced_data/driver_actions_reach_side.json', 
                '/home/arantxa/universidad/TFG/src/balanced_data/driver_actions_phonecall_right.json']

        
SVM_performer = SVM(*json_files)
SVM_performer.open_jsons()
SVM_performer.prepare_data(SVM_performer.both)
SVM_performer.prepare_data(SVM_performer.onlyleft)
SVM_performer.prepare_data(SVM_performer.onlyright)
SVM_performer.prepare_data(SVM_performer.radio)
SVM_performer.prepare_data(SVM_performer.drinking)
SVM_performer.prepare_data(SVM_performer.reachside)
SVM_performer.prepare_data(SVM_performer.phonecallright)

dataset = pd.DataFrame(SVM_performer.rows)
print(dataset)

dataset['label'] = dataset['label'].replace("hands_using_wheel/both", 0)
dataset['label'] = dataset['label'].replace("hands_using_wheel/only_left", 1)
dataset['label'] = dataset['label'].replace("hands_using_wheel/only_right", 2)
dataset['label'] = dataset['label'].replace("driver_actions/radio", 3)
dataset['label'] = dataset['label'].replace("driver_actions/drinking", 4)
dataset['label'] = dataset['label'].replace("driver_actions/reach_side", 5)
dataset['label'] = dataset['label'].replace("driver_actions/phonecall_right", 6)

dataset_multilabel = pd.get_dummies(dataset['label'], prefix='class').astype(int)

dataset = pd.concat([dataset, dataset_multilabel], axis=1)
print(dataset)

num_classes = dataset['label'].max() + 1

dataset['multilabel'] = dataset['label'].apply(lambda x: [1 if i == x else 0 for i in range(num_classes)])

dataset = dataset.drop(columns=['label'])
dataset = dataset.drop(columns=['class_0'])
dataset = dataset.drop(columns=['class_1'])
dataset = dataset.drop(columns=['class_2'])
dataset = dataset.drop(columns=['class_3'])
dataset = dataset.drop(columns=['class_4'])
dataset = dataset.drop(columns=['class_5'])
dataset = dataset.drop(columns=['class_6'])

print(dataset)

cols_to_group = dataset.columns.difference(["multilabel"])
grouped = dataset.groupby(list(cols_to_group))["multilabel"].apply(
    lambda x: [int(any(values)) for values in zip(*x)]
).reset_index()

dataset = grouped

# -----------------------------------------------------
# salia que había 44 iguales (o sea 22 menos)
# -----------------------------------------------------


X = dataset.iloc[:, :-1] # todo menos etiquetas
Y = dataset.iloc[:, -1] # etiquetas

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.25, random_state=1, stratify=Y) 

train_class_counts = pd.Series(Y_train).value_counts()

test_class_counts = pd.Series(Y_test).value_counts()

print("Distribución de clases en el conjunto de entrenamiento:")
print(train_class_counts)

print("\nDistribución de clases en el conjunto de prueba:")
print(test_class_counts)

Y_train_bin = np.array(list(Y_train)) 
Y_test_bin = np.array(list(Y_test)) 

print("Formato de Y_train_bin:", Y_train_bin.shape)

multilabel_model = OneVsRestClassifier(SVC(kernel="linear", probability=True,random_state=1))
multilabel_model.fit(X_train, Y_train_bin)

Y_pred_multilabel = multilabel_model.predict(X_test)
Y_train_pred = multilabel_model .predict(X_train)

print("Predicciones multilabel (primeras 5 filas):\n", Y_pred_multilabel[:5])


accuracy = accuracy_score(Y_train_bin, Y_train_pred)
print("Precisión (entrenamiento):", accuracy)

accuracy = accuracy_score(Y_test_bin, Y_pred_multilabel)
print("Precisión:", accuracy)

# etiquetas de las clases
class_labels = ['Both', 'Left', 'Right', 'Radio', 'Drinking', 'Reachside', 'Phonecallr']

n_labels = Y_train_bin.shape[1] # total de etiquetas

# Inicializar la matriz de confusión global
global_conf_matrix = np.zeros((n_labels, n_labels))

for i in range(Y_train_bin.shape[0]): 
    for j in range(n_labels): 
        if Y_train_bin[i, j] == 1: # Si la etiqueta real es 1
            for k in range(n_labels):  # Ver en qué clase se predijo
                global_conf_matrix[j, k] += Y_train_pred[i, k]  

row_sums = global_conf_matrix.sum(axis=1, keepdims=True) 
global_conf_matrix_normalized = global_conf_matrix / row_sums

# Visualizar la matriz de confusión global normalizada
plt.figure(figsize=(10, 7))
sns.heatmap(global_conf_matrix_normalized.T, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels,  # Usar las etiquetas personalizadas
            yticklabels=class_labels)  
plt.xlabel("Etiqueta Real")
plt.ylabel("Predicción")
plt.title("Matriz de Confusión de train")
plt.show()

# Inicializar la matriz de confusión global
global_conf_matrix = np.zeros((n_labels, n_labels))

for i in range(Y_test_bin.shape[0]): 
    for j in range(n_labels): 
        if Y_test_bin[i, j] == 1: # Si la etiqueta real es 1
            for k in range(n_labels):  # Ver en qué clase se predijo
                global_conf_matrix[j, k] += Y_pred_multilabel[i, k]  

row_sums = global_conf_matrix.sum(axis=1, keepdims=True) 
global_conf_matrix_normalized = global_conf_matrix / row_sums

# Visualizar la matriz de confusión global normalizada
plt.figure(figsize=(10, 7))
sns.heatmap(global_conf_matrix_normalized.T, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels,  # Usar las etiquetas personalizadas
            yticklabels=class_labels)  
plt.xlabel("Etiqueta Real")
plt.ylabel("Predicción")
plt.title("Matriz de Confusión de test")
plt.show()

sys.argv = [sys.argv[0], '/home/arantxa/universidad/TFG/src/balanced_data/driver_actions_drinking.json', '/home/arantxa/universidad/TFG/src/balanced_jsons/']

sys.argv = [sys.argv[0], '/home/arantxa/universidad/TFG/src/j_pruebas/1_s1/frames.json',
            '/home/arantxa/universidad/TFG/src/j_pruebas/1_s1/hands.json',
            '/home/arantxa/universidad/TFG/src/j_pruebas/1_s1/pose.json',
            '/home/arantxa/universidad/TFG/src/j_pruebas/1_s1/face.json',
            '/home/arantxa/universidad/TFG/src/j_pruebas/1_s1/pose.mp4',
            '/home/arantxa/universidad/TFG/src/j_pruebas/1_s1/hands.mp4',
            '/home/arantxa/universidad/TFG/src/j_pruebas/1_s1/face.mp4']

vr = videoReconstructor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
video_paths = [sys.argv[5], sys.argv[6], sys.argv[7]]
vr.open_jsons()
vr.reconstruct(video_paths)

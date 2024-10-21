import cv2
import mediapipe as mp
import sys
import numpy as np
import json
import time
import csv
import os
from helpers import relative, relativeT
import cv2 
from time import sleep

class videoReconstructor:
    def __init__(self, json_1, json_2, json_3, json_4, video_1, video_2, video_3):
        self.files = [json_1, json_2, json_3, json_4]
        self.video_paths = [video_1, video_2, video_3]
        self.actions = self.load_actions_from_json()

        self.data_hands = []
        self.data_face = []
        self.data_pose = []

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
        print(video_paths)

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
                    frame = np.zeros((height, width, 3), dtype=np.uint8)  # Frame negro

                black_frame = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

                if cap_number == 0:  # First video (pose)
                    if frame_number >= self.pose_sync:
                        if not video_started[cap_number]:  # Since frame 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True  # Marks the video as started

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # print(frame)
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
                for i, action in enumerate(actions):
                    cv2.putText(combined_frame, action, (reduced_width + 10, reduced_height + 30 + i * 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

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
            # print("Hands")
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        print("Frame: ", frame_number)
                        # Supongamos que los keypoints están en 'iterations["left"]'
                        keypoints = iterations["left"]  # Esto debe contener los kp1, kp2, ..., kp15, center
                        for key, value in keypoints.items():
                            x = int(value["x"] * frame.shape[1])
                            y = int(value["y"] * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        # print("Left: ", keypoints)
                        # sleep(100)

                        self.draw_connections(frame, keypoints, self.HANDS_CONNECTION)

                        keypoints_center = iterations["left"]["center"]
                        x = int(keypoints_center["x"] * frame.shape[1])
                        y = int(keypoints_center["y"] * frame.shape[0])
                        cv2.circle(frame, (x, y), 20, (255, 0, 0), -1)

                            # print(f"x: {x}, y: {y}")

                        keypoints = iterations["right"]  # Esto debe contener los kp1, kp2, ..., kp15, center
                        for key, value in keypoints.items():
                            x = int(value["x"] * frame.shape[1])
                            y = int(value["y"] * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        self.draw_connections(frame, keypoints, self.HANDS_CONNECTION)

                        keypoints_center = iterations["right"]["center"]
                        x = int(keypoints_center["x"] * frame.shape[1])
                        y = int(keypoints_center["y"] * frame.shape[0])
                        cv2.circle(frame, (x, y), 20, (255, 0, 0), -1)

                            # print(f"x: {x}, y: {y}")

        elif json == "pose":
            data = self.data_pose
            # print("Pose")
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        # Supongamos que los keypoints están en 'iterations["keypoints"]'
                        # print(iterations)
                        keypoints_trunk = iterations["trunk"]
                        for key, value in keypoints_trunk.items():
                            x = int(value["x"] * frame.shape[1])
                            y = int(value["y"] * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        self.draw_connections(frame, keypoints_trunk, self.POSE_CONNECTION)
                        
                        keypoints_arm_left = iterations["arms"]["left"]
                        for key, value in keypoints_arm_left.items():
                            x = int(value["x"] * frame.shape[1])
                            y = int(value["y"] * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        self.draw_connections(frame, keypoints_arm_left, self.POSE_CONNECTION)
                        
                        keypoints_arm_right = iterations["arms"]["right"]
                        for key, value in keypoints_arm_right.items():
                            x = int(value["x"] * frame.shape[1])
                            y = int(value["y"] * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        self.draw_connections(frame, keypoints_arm_right, self.POSE_CONNECTION)
                            
                        keypoints_hand_left = iterations["hands"]["left"]
                        for key, value in keypoints_hand_left.items():
                            x = int(value["x"] * frame.shape[1])
                            y = int(value["y"] * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                        self.draw_connections(frame, keypoints_hand_left, self.HANDS_CONNECTION)

                        keypoints_hand_left_center = iterations["hands"]["left"]["center"]
                        x = int(keypoints_hand_left_center["x"] * frame.shape[1])
                        y = int(keypoints_hand_left_center["y"] * frame.shape[0])
                        cv2.circle(frame, (x, y), 20, (255, 0, 0), -1)

                        keypoints_hand_right = iterations["hands"]["right"]
                        for key, value in keypoints_hand_right.items():
                            x = int(value["x"] * frame.shape[1])
                            y = int(value["y"] * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                        self.draw_connections(frame, keypoints_hand_right, self.HANDS_CONNECTION)

                        keypoints_hand_right_center = iterations["hands"]["right"]["center"]
                        x = int(keypoints_hand_right_center["x"] * frame.shape[1])
                        y = int(keypoints_hand_right_center["y"] * frame.shape[0])
                        cv2.circle(frame, (x, y), 20, (255, 0, 0), -1)

        elif json == "face":
            data = self.data_face
            # print("Face")
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        keypoints = iterations["face"]
                        for key, value in keypoints.items():
                            if(key.startswith("kp")):
                                x = int(value["x"] * frame.shape[1])
                                y = int(value["y"] * frame.shape[0])
                                cv2.circle(frame, (x, y), 5, (128, 255, 32), -1)

                            else:
                                # print("ALGUIEN?")
                                p1 = keypoints["gaze"]["p1"]
                                p2 = keypoints["gaze"]["p2"]

                                x1 = int(p1["x"]) # * frame.shape[1])
                                y1 = int(p1["y"]) # * frame.shape[0])
                                x2 = int(p2["x"]) # * frame.shape[1])
                                y2 = int(p2["y"]) # * frame.shape[0])

                                # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

                                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

                        self.draw_connections(frame, keypoints, self.FACE_CONNECTION)




                            # print(f"x: {x}, y: {y}")

    def draw_connections(self, frame, keypoints, connections):
        for connection in connections:
            idx1, idx2 = connection  # Índices de los puntos a conectar

            kp1 = next((v for k, v in keypoints.items() if v.get("index") == idx1 and k.startswith("kp")), None)
            kp2 = next((v for k, v in keypoints.items() if v.get("index") == idx2 and k.startswith("kp")), None)

            if kp1 and kp2:
                x1 = int(kp1["x"] * frame.shape[1])
                y1 = int(kp1["y"] * frame.shape[0])
                x2 = int(kp2["x"] * frame.shape[1])
                y2 = int(kp2["y"] * frame.shape[0])

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)




                                

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Falta el directorio.")
    else:
        vr = videoReconstructor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
        video_paths = [sys.argv[5], sys.argv[6], sys.argv[7]]
        vr.open_jsons()
        vr.reconstruct(video_paths)

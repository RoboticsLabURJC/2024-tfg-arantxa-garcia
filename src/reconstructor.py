"""

Uso: python reconstructor.py <json_original> <json_hands> <json_pose> <json_face> <video_body> <video_hands> <video_face>

Json_original: Archivo JSON original con las acciones y sincronización de cámaras.
Json_hands: Archivo JSON con los datos de las manos.
Json_pose: Archivo JSON con los datos de la pose.
Json_face: Archivo JSON con los datos de la cara.
Video_body: Video con la cámara del pose.
Video_hands: Video con la cámara de las manos.
Video_face: Video con la cámara de la cara.

Imprimirá un video con los frames de las manos, la cara y la pose de cada acción pasados por MediaPipe.

"""

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

left_only = 0
left_and_drinking = 0
left_and_radio = 0
left_and_reaching = 0

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
                    frame = np.zeros((height, width, 3), dtype=np.uint8)

                black_frame = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

                if cap_number == 0:  # First video (pose)
                    if frame_number >= self.pose_sync:
                        if not video_started[cap_number]:
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
                print(actions)
                print("-------------------------------------------")
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
        print("Frame: ", frame_number)
        if json == "hands":
            data = self.data_hands
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        # for x, y, z in iterations["hands"]:
                        #     x = int(x * frame.shape[1])
                        #     y = int(y * frame.shape[0])
                        #     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        # for x, y in iterations["centers"]:
                        #     x = int(x * frame.shape[1])
                        #     y = int(y * frame.shape[0])
                        #     cv2.circle(frame, (x, y), 5, (255, 0, 0), 30)

                        # hand_left = iterations["hands"][:21]
                        hand_right = iterations["hands"][21:]

                        # self.draw_connections(frame, hand_left, self.HANDS_CONNECTION)
                        self.draw_connections(frame, hand_right, self.HANDS_CONNECTION)

        if json == "pose":
            data = self.data_pose
            # print("Pose")
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        for x, y, z in iterations["pose"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        pose_data = iterations["pose"][:8]
                        left_hand = iterations["pose"][8:29]
                        right_hand = iterations["pose"][29:50]
                        left_center = iterations["pose"][50]
                        right_center = iterations["pose"][51]

                        # cv2.circle(frame, (int(left_center[0] * frame.shape[1]), int(left_center[1] * frame.shape[0])), 5, (255, 0, 0), 30)
                        # cv2.circle(frame, (int(right_center[0] * frame.shape[1]), int(right_center[1] * frame.shape[0])), 5, (255, 0, 0), 30)

                        # self.draw_connections(frame, pose_data, self.POSE_CONNECTION)
                        # self.draw_connections(frame, left_hand, self.HANDS_CONNECTION)
                        self.draw_connections(frame, right_hand, self.HANDS_CONNECTION)

        # elif json == "face":
        #     print("Frame: ", frame_number)
        #     data = self.data_face
        #     # print("Face")
        #     if "iterations" in data:
        #         for iterations in data["iterations"]:
        #             if iterations["frame"] == frame_number:

        #                 for x, y, indx in iterations["face"]:
        #                     x = int(x * frame.shape[1])
        #                     y = int(y * frame.shape[0])
        #                     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        #                     # print("Face: ", x, y)

        #                 x, y = iterations["gaze"][0]
        #                 x2, y2 = iterations["gaze"][1]

        #                 x, y = int(x), int(y)
        #                 x2, y2 = int(x2), int(y2)
                        
        #                 cv2.line(frame, (x, y), (x2, y2), (0, 0, 255), 7)

        #                 self.draw_connections(frame, iterations["face"], self.FACE_CONNECTION)

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
        print("Falta el directorio.")
    else:
        vr = videoReconstructor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
        video_paths = [sys.argv[5], sys.argv[6], sys.argv[7]]
        vr.open_jsons()
        vr.reconstruct(video_paths)

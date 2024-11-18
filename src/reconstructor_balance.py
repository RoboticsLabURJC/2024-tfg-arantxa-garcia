import cv2
import sys
import numpy as np
import json
import os
import time

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
        try:
            with open(self.balanced_json, 'r', encoding='utf-8-sig') as f:
                self.data = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.balanced_json}. Trying alternative encoding...")
            with open(self.balanced_json, 'r', encoding='latin-1') as f:
                self.data = json.load(f)

    def reconstruct(self):
        self.open_json()

        for action in self.data['actions']:
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

            self.show_collage(face_im, hands_im, pose_im, action)




    def show_collage(self, face_im, hands_im, pose_im, action):
        frame1 = hands_im
        frame2 = pose_im
        frame3 = face_im

        if frame1 is None or frame2 is None or frame3 is None:
            print("Error al leer uno de los frames")
            return
        
        # self.paint_frame(action['hands'], "hands", frame1)
        # self.paint_frame(action['pose'], "pose", frame2)
        # self.paint_frame(action['face'], "face", frame3)

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
        print("Frame: ", frame_number)
        if json == "hands":
            # # print("Hands")
            # if "iterations" in data:
            #     for iterations in data["iterations"]:
            #         if iterations["frame"] == frame_number:
            #             # print(iterations["hands"])
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

            # self.draw_connections(frame, iterations["hands"], self.HANDS_CONNECTION)
            self.draw_connections(frame, hand_right, self.HANDS_CONNECTION)
            self.draw_connections(frame, hand_left, self.HANDS_CONNECTION)

        elif json == "pose":
            for x, y, z in data['pose']['pose']:
                x = int(x * frame.shape[1])
                y = int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        # print("Pose: ", iterations["pose"])
                        # print("len: ", len(iterations["pose"]))

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
                # print("Face: ", x, y)

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
        print("Falta el archivo JSON.")
    else:
        vr = balanceReconstructor(sys.argv[1], sys.argv[2])
        vr.reconstruct()

# generar un nuevo dataset con los frames independientes, tener un directorio por clase y que cada directorio tenga su propio json
# hay que referenciar en el json de que video viene cada frame
# asegurarme de que en el nuevi dataset este todo sincronizado
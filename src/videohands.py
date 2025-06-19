"""

Uso: python videohands.py <json_general> <video_path_body> <video_path_hands> <video_path_face> [--load]

--load: Crea 3 jsons con los datos de las camaras y los guarda en el directorio actual.
--combine: Combina los 3 videos en uno solo, mostrando cada camara en un cuadrante de la pantalla.

"""

import cv2
import mediapipe as mp
import sys
import numpy as np
import json
import os
from helpers import relative, relativeT
import copy
import gaze
from phone_model import PhoneDetector


class VideoProcessor:
    def __init__(self, video_path, json_path, draw_pose=False, draw_hands=False, draw_face=False, pose_colors=None):
        self.video_path = video_path
        self.json_path = json_path
        self.draw_pose = draw_pose
        self.draw_hands = draw_hands
        self.draw_face = draw_face

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands 
        self.hands_pose = mp.solutions.hands.Hands() 
        self.hands_only = mp.solutions.hands.Hands() 
        self.mp_face_mesh = mp.solutions.face_mesh
        # self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.face_sync = 0
        self.hands_sync = 0
        self.pose_sync = 0

        self.data_pose = {}
        self.data_face = {}
        self.data_hands = {}

        default_pose_colors = {
            'LEFT_ARM': (255, 0, 0),
            'RIGHT_ARM': (0, 255, 0),
            'LEFT_LEG': (0, 0, 255),
            'RIGHT_LEG': (255, 255, 0),
            'TRUNK': (255, 0, 255),
            'HEAD': (0, 255, 255)
        }

        self.pose_colors = pose_colors if pose_colors is not None else default_pose_colors
        self.actions = self.load_actions_from_json()

    def calculate_center_of_mass(self, landmarks):
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]

        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)

        return center_x, center_y

    def process_pose(self, frame, results_pose):
        if results_pose.pose_landmarks:
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                if start_idx in self.mp_pose.PoseLandmark.__members__.values() and end_idx in self.mp_pose.PoseLandmark.__members__.values():
                    start_landmark = results_pose.pose_landmarks.landmark[start_idx]
                    end_landmark = results_pose.pose_landmarks.landmark[end_idx]

                    if start_idx in [11, 13, 15, 17, 19, 21] and end_idx in [11, 13, 15, 17, 19, 21]:
                        color = self.pose_colors['LEFT_ARM']
                    elif start_idx in [12, 14, 16, 18, 20, 22] and end_idx in [12, 14, 16, 18, 20, 22]:
                        color = self.pose_colors['RIGHT_ARM']
                    elif start_idx in [11, 12, 24, 23] and end_idx in [11, 12, 24, 23]:
                        color = self.pose_colors['TRUNK']
                    else:
                        continue

                    start_point = (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0]))
                    end_point = (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, color, 2)

    def update_pose_json(self, results_pose, results_hands):
        # Índices en self.data_pose['pose']
        # -----------------------------------
        #  0 -  7   -> Puntos clave del cuerpo (hombros, codos, muñecas, caderas)
        #  8 - 28   -> Puntos de la mano izquierda
        #  29 - 49  -> Puntos de la mano derecha
        #    50     -> Centro de masa de la mano izquierda
        #    51     -> Centro de masa de la mano derecha


        self.data_pose['pose'] = [[0, 0, idx] for idx in range(53)]

        counter = 0

        if results_pose.pose_landmarks:
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                # if idx in [11, 12, 23, 24, 13, 15, 14, 16]:
                #     self.data_pose['pose'][counter] = [landmark.x, landmark.y, idx]
                #     counter += 1
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

        if left_hand_landmarks: # ESTO NO FUNCIONA BIEN CHECKEAR EL INDICE
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
            'frame': len(self.data_pose['iterations']),
            'pose': self.data_pose['pose'].copy()
        }

        self.data_pose['iterations'].append(new_iteration)

    def process_hands(self, frame, results_hands):
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )

                center_x, center_y = self.calculate_center_of_mass(hand_landmarks.landmark)
                center_point = (int(center_x * frame.shape[1]), int(center_y * frame.shape[0]))

                cv2.circle(frame, center_point, 20, (255, 128, 64), -1)

    def update_hands_json(self, results_hands):
        self.data_hands['hands'] = [[0.0, 0.0, 0.0] for _ in range(42)]  # 21 puntos para cada mano
        self.data_hands['centers'] = [[0.0, 0.0], [0.0, 0.0]]  # [left_center, right_center]

        if results_hands.multi_hand_landmarks:
            self.data_hands['hands'] = [[0.0, 0.0, 0.0] for _ in range(42)]
            self.data_hands['centers'] = [[0.0, 0.0], [0.0, 0.0]]

            for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                handedness = results_hands.multi_handedness[hand_idx].classification[0].label
                hand_offset = 21 if handedness == 'Left' else 0 # para que los puntos de la mano derecha empiecen en el 21  

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    self.data_hands['hands'][hand_offset + idx] = [landmark.x, landmark.y, idx]

                center_x, center_y = self.calculate_center_of_mass(hand_landmarks.landmark)
                if handedness == 'Left':
                    self.data_hands['centers'][1] = [center_x, center_y]
                else:
                    self.data_hands['centers'][0] = [center_x, center_y]

        new_iteration = {
            'frame': len(self.data_hands['iterations']),
            'hands': [coord.copy() for coord in self.data_hands['hands']],
            'centers': [center.copy() for center in self.data_hands['centers']]
        }
        self.data_hands['iterations'].append(new_iteration)

    def process_face(self, frame, results_face):
        if results_face.multi_face_landmarks:
            # p1, p2 = gaze.gaze(frame, results_face.multi_face_landmarks[0]) 

            for face_landmarks in results_face.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

    def update_face_json(self, results_face, frame):
        self.data_face['face'] = [[0.0, 0.0, i] for i in range(28)]  
        self.data_face['gaze'] = [[0.0, 0.0] for _ in range(2)]

        if results_face.multi_face_landmarks:
            self.data_face['face'] = [[0.0, 0.0, i] for i in range(28)]
            self.data_face['gaze'] = [[0.0, 0.0] for _ in range(2)]

            p1, p2 = gaze.gaze(frame, results_face.multi_face_landmarks[0])
            self.data_face['gaze'][0] = [p1[0], p1[1]] 
            self.data_face['gaze'][1] = [p2[0], p2[1]]  

            counter = 0

            for face_idx, face_landmarks in enumerate(results_face.multi_face_landmarks):
                for idx, landmark in enumerate(face_landmarks.landmark):
                    # if idx in {17, 61, 291, 0, 206, 426, 50, 48, 4, 278, 280, 145, 122, 351, 374, 130, 133, 362, 359, 159, 386, 46, 276, 105, 107, 336, 334}:
                    #     # combined_landmarks.append([landmark.x, landmark.y, idx])
                    #     self.data_face['face'][counter] = [landmark.x, landmark.y, idx]
                    #     counter += 1
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

        self.data_face['iterations'].append(new_iteration)

    def process_video(self): 
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"No se pudo abrir el video: {self.video_path}")
            return

        frame_data = []
        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_pose = self.pose.process(frame_rgb)
            results_hands = self.hands.process(frame_rgb)
            results_face = self.face_mesh.process(frame_rgb)

            if self.draw_pose:
                self.process_pose(frame, results_pose)
            if self.draw_hands:
                self.process_hands(frame, results_hands)
            if self.draw_face:
                self.process_face(frame, results_face)

            # Agregar información de este frame
            actions = self.actions.get(frame_number, [])
            frame_data.append({
                'frame_number': frame_number,
                'actions': ', '.join(actions),
                'pose_sync': self.pose_sync,
                'hands_sync': self.hands_sync,
                'face_sync': self.face_sync
            })

            self.print_action_for_frame(frame, frame_number)

            cv2.imshow('MediaPipe Hands + Pose + Face', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()

    def combine_videos(self, video_paths):
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
            print("Frame number: ", frame_number)

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
                        results_pose = self.pose.process(frame_rgb)
                        results_hands = self.hands_pose.process(frame_rgb)
                        self.process_pose(frame, results_pose)
                        self.process_hands(frame, results_hands) 
                        # print(frame)
                        frame = cv2.resize(frame, (reduced_width, reduced_height))
                    else:
                        frame = black_frame  # Black till synchronization of pose_sync

                elif cap_number == 1:  # Second video (hands)
                    if frame_number >= self.hands_sync:
                        if not video_started[cap_number]: 
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True 

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_hands = self.hands_only.process(frame_rgb)
                        self.process_hands(frame, results_hands)
                        frame = cv2.resize(frame, (reduced_width, reduced_height))
                    else:
                        frame = black_frame  # Black till synchronization of hands_sync

                elif cap_number == 2:  # Third video (face)
                    if frame_number >= self.face_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True  

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_face = self.face_mesh.process(frame_rgb)
                        self.process_face(frame, results_face) 
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

    def load_json(self, video_paths, phone_detector):
        self.create_json()

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
        frame_count = [0, 0, 0]
        video_started = [False, False, False]  # Flags to track when each video starts
        end_video = False
        prev_frame = [None, None, None]

        while all([cap.isOpened() for cap in caps]):

            frames = []
            cap_number = 0
            for cap in caps:
                # If the video has already started, read it normally
                if video_started[cap_number]:
                    success, frame = cap.read()
                    if not success:
                        end_video = True
                        break

                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Frame actual según OpenCV

                    if current_frame != frame_count[cap_number] + 1:
                        print(f"Frame perdido: Esperado {frame_count[cap_number] + 1}, leído {current_frame}")

                    if prev_frame[cap_number] is not None:
                        diff = np.abs(frame.astype("int") - prev_frame[cap_number].astype("int"))
                        avg_diff = np.mean(diff)

                        if avg_diff < 5: 
                            print("Frame duplicado detectado")
                        elif avg_diff > 200:  
                            print("Frame corrupto detectado")

                    frame_count[cap_number] = current_frame
                else:
                    # If it has not started, it pauses in black until the synchronization is complete
                    frame = np.zeros((height, width, 3), dtype=np.uint8)  # Frame negro

                black_frame = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

                if cap_number == 0:  # First video (pose)
                    if frame_number >= self.pose_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_pose = self.pose.process(frame_rgb)
                        results_hands = self.hands_pose.process(frame_rgb)
                        self.process_pose(frame, results_pose)
                        self.process_hands(frame, results_hands)
                        is_phone = phone_detector.detect(frame_rgb)

                        self.update_pose_json(results_pose, results_hands, is_phone)
                    else:
                        frame = black_frame  # Black till synchronization of pose_sync

                elif cap_number == 1:  # Second video (hands)
                    # print("entra al segundo video")
                    if frame_number >= self.hands_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_hands = self.hands_only.process(frame_rgb)
                        self.process_hands(frame, results_hands)
                        # print("aqui si")
                        self.update_hands_json(results_hands)
                    else:
                        frame = black_frame  # Black till synchronization of hands_sync

                elif cap_number == 2:  # Third video (face)
                    if frame_number >= self.face_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_face = self.face_mesh.process(frame_rgb)
                        self.process_face(frame, results_face)

                        self.update_face_json(results_face, frame_rgb)
                    else:
                        frame = black_frame  # Black till synchronization of face_sync

                cap_number += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

            # if(frame_number % 500 == 0):
            #     # Guarda todos los JSON cada 400 frames
            #     # print("Guardando JSON...")
            #     print(frame_number)
            #     self.save_json_files() # GUARDO LA PRIMERA POSICION 400 VECES

            frame_number += 1
            print(frame_number)

            if end_video:
                break

        print("se acabó")
        self.save_json_files()

        for cap in caps:
            cap.release()
        out.release()

    def save_json_files(self):
        # print(self.data_hands)
        # print("\n")
        # print("--------------------------------------------------------------------------------------")
        # print("\n")
        with open("pose.json", 'w') as json_file:
            json.dump(self.data_pose, json_file, indent=4)

        with open("hands.json", 'w') as json_file:
            json.dump(self.data_hands, json_file, indent=4)

        with open("face.json", 'w') as json_file:
            json.dump(self.data_face, json_file, indent=4)

        # time.sleep(2)
    
    def load_actions_from_json(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.json_path}. Trying alternative encoding...")
            with open(self.json_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        
        actions = {}

        key = "openlabel" if "openlabel" in data else "vcd" if "vcd" in data else None

        if key and "actions" in data[key]:
            for frame_id, frame_data in data[key]["actions"].items():
                if "type" in frame_data:
                    for frame_interval in frame_data["frame_intervals"]:
                        frame_start = frame_interval["frame_start"]
                        frame_end = frame_interval["frame_end"]
                        for frame in range(frame_start, frame_end + 1):
                            if frame not in actions:
                                actions[frame] = []
                            if frame_data["type"] not in actions[frame]:
                                actions[frame].append(frame_data["type"])

        if key and "streams" in data[key]:
            for frame_id, frame_data in data[key]["streams"].items():
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

    def print_action_for_frame(self, frame, frame_number):
        if frame_number in self.actions:
            actions = self.actions[frame_number]
            for i, action in enumerate(actions):
                cv2.putText(frame, action, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
    def create_json(self):
        print("Creando JSON...")

        # Pose JSON
        if os.path.exists("pose.json"):
            with open("pose.json", 'r') as json_file:
                self.data_pose = json.load(json_file)
        else:
            self.data_pose = {
                'iterations': []
            }

        with open("pose.json", 'w') as json_file:
            json.dump(self.data_pose, json_file, indent=4)

        # Face JSON
        if os.path.exists("face.json"):
            with open("face.json", 'r') as json_file:
                self.data_face = json.load(json_file)
        else:
            self.data_face = {
                'iterations': []
            }

        with open("face.json", 'w') as json_file:
            json.dump(self.data_face, json_file, indent=4)

        # Hands JSON
        if os.path.exists("hands.json"):
            with open("hands.json", 'r') as json_file:
                self.data_hands = json.load(json_file)
        else:
            self.data_hands = {
                'iterations': []
            }

        with open("hands.json", 'w') as json_file:
            json.dump(self.data_hands, json_file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Falta el json y el path del video.")
    else:
        json_path = sys.argv[1]
        video_path = sys.argv[2]
        draw_pose = "--pose" in sys.argv
        draw_hands = "--hands" in sys.argv
        draw_face = "--face" in sys.argv
        combine_videos = "--combine" in sys.argv
        load_json = "--load" in sys.argv

        processor = VideoProcessor(video_path, json_path, draw_pose=draw_pose, draw_hands=draw_hands, draw_face=draw_face)
        phone_detector = PhoneDetector()

        if combine_videos:
            if len(sys.argv) < 6:
                print("Faltan rutas de videos para combinar.")
            else:
                video_paths = sys.argv[2:5]
                processor.combine_videos(video_paths)
        elif load_json:
            video_paths = sys.argv[2:5]
            processor.load_json(video_paths, phone_detector)
        else:
            processor.process_video()

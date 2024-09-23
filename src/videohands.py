import cv2
import mediapipe as mp
import sys
import numpy as np
import json
import time
import csv
import os

class VideoProcessor:
    def __init__(self, video_path, json_path, draw_pose=False, draw_hands=False, draw_face=False, pose_colors=None):
        self.video_path = video_path
        self.json_path = json_path
        self.draw_pose = draw_pose
        self.draw_hands = draw_hands
        self.draw_face = draw_face

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands # con uno vale para ambos videos ?????????????????????????????????????????????????????????
        # self.hands = self.mp_hands.Hands()
        self.hands_pose = mp.solutions.hands.Hands()  # Para el video de pose
        self.hands_only = mp.solutions.hands.Hands()  # Para el video de hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
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
            'TORSO': (255, 0, 255),
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
                        color = self.pose_colors['TORSO']
                    else:
                        continue

                    start_point = (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0]))
                    end_point = (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, color, 2)

    def update_pose_json(self, results_pose, results_hands):
        if results_pose.pose_landmarks:
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                # Define los puntos clave que quieres guardar en el JSON.
                if idx == 11:  # pnt1 en torso
                    self.data_pose['torso']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    self.data_pose['brazos']['izquierdo']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 12:  # pnt2 en torso
                    self.data_pose['torso']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    self.data_pose['brazos']['derecho']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 23:
                    self.data_pose['torso']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 24:
                    self.data_pose['torso']['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 13:  # pnt1 en brazo izquierdo
                    self.data_pose['brazos']['izquierdo']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 15:
                    self.data_pose['brazos']['izquierdo']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 14:  # pnt1 en brazo derecho
                    self.data_pose['brazos']['derecho']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 16:
                    self.data_pose['brazos']['derecho']['pnt3'] = {'x': landmark.x, 'y': landmark.y}

        if results_hands.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                handedness = results_hands.multi_handedness[hand_idx].classification[0].label

                # Decide si es mano izquierda o derecha
                hand_key = 'izquierda' if handedness == 'Left' else 'derecha'

                # Itera sobre los landmarks de la mano
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx == 0:
                        self.data_pose['manos'][hand_key]['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 1:
                        self.data_pose['manos'][hand_key]['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 4:
                        self.data_pose['manos'][hand_key]['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 5:
                        self.data_pose['manos'][hand_key]['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 7:
                        self.data_pose['manos'][hand_key]['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 8:
                        self.data_pose['manos'][hand_key]['pnt6'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 9:
                        self.data_pose['manos'][hand_key]['pnt7'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 11:
                        self.data_pose['manos'][hand_key]['pnt8'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 12:
                        self.data_pose['manos'][hand_key]['pnt9'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 13:
                        self.data_pose['manos'][hand_key]['pnt10'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 15:
                        self.data_pose['manos'][hand_key]['pnt11'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 16: 
                        self.data_pose['manos'][hand_key]['pnt12'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 17:
                        self.data_pose['manos'][hand_key]['pnt13'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 19:
                        self.data_pose['manos'][hand_key]['pnt14'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 20:
                        self.data_pose['manos'][hand_key]['pnt15'] = {'x': landmark.x, 'y': landmark.y}

                center_x, center_y = self.calculate_center_of_mass(hand_landmarks.landmark)
                self.data_pose['manos'][hand_key]['center'] = {'x': center_x, 'y': center_y}
                
        new_iteration = {
            'frame': len(self.data_pose['iterations']) + 1,  # número de iteración (puedes modificarlo)
            'torso': self.data_pose['torso'],
            'brazos': self.data_pose['brazos'],
            'manos': self.data_pose['manos']
        }

        self.data_pose['iterations'].append(new_iteration)

        # Guarda los cambios en el archivo JSON
        # with open("pose_json.json", 'w') as json_file:
        #     json.dump(self.data_pose, json_file, indent=4)

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
        if results_hands.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                handedness = results_hands.multi_handedness[hand_idx].classification[0].label

                # Decide si es mano izquierda o derecha
                hand_key = 'izquierda' if handedness == 'Left' else 'derecha'

                # Itera sobre los landmarks de la mano
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx == 0:  # pnt1 en la mano
                        self.data_hands[hand_key]['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 1:  # pnt2 en la mano
                        self.data_hands[hand_key]['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 4:
                        self.data_hands[hand_key]['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 5:
                        self.data_hands[hand_key]['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 7:
                        self.data_hands[hand_key]['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 8:
                        self.data_hands[hand_key]['pnt6'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 9:
                        self.data_hands[hand_key]['pnt7'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 11:
                        self.data_hands[hand_key]['pnt8'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 12:
                        self.data_hands[hand_key]['pnt9'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 13:
                        self.data_hands[hand_key]['pnt10'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 15:
                        self.data_hands[hand_key]['pnt11'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 16:
                        self.data_hands[hand_key]['pnt12'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 17:
                        self.data_hands[hand_key]['pnt13'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 19:
                        self.data_hands[hand_key]['pnt14'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 20:
                        self.data_hands[hand_key]['pnt15'] = {'x': landmark.x, 'y': landmark.y}

                # Calcula el centro de la mano (opcional)
                center_x, center_y = self.calculate_center_of_mass(hand_landmarks.landmark)
                self.data_hands[hand_key]['center'] = {'x': center_x, 'y': center_y}

        new_iteration = {
            'frame': len(self.data_hands['iterations']) + 1,  # número de iteración (puedes modificar
            'izquierda': self.data_hands['izquierda'],
            'derecha': self.data_hands['derecha']
        }

        self.data_hands['iterations'].append(new_iteration)

        # Guarda los cambios en el archivo JSON
        # with open("hands_json.json", 'w') as json_file:
        #     json.dump(self.data_hands, json_file, indent=4)

    def process_face(self, frame, results_face):
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

    def update_face_json(self, results_face):
        if results_face.multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(results_face.multi_face_landmarks):
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx == 475:
                        self.data_face['ojos']['izquierdo']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 477:
                        self.data_face['ojos']['izquierdo']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 381:
                        self.data_face['ojos']['izquierdo']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 377:
                        self.data_face['ojos']['izquierdo']['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 382:
                        self.data_face['ojos']['izquierdo']['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 395:
                        self.data_face['ojos']['izquierdo']['pnt6'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 470:
                        self.data_face['ojos']['derecho']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 472:
                        self.data_face['ojos']['derecho']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 153:
                        self.data_face['ojos']['derecho']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 163:
                        self.data_face['ojos']['derecho']['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 154:
                        self.data_face['ojos']['derecho']['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 7:
                        self.data_face['ojos']['derecho']['pnt6'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 0:
                        self.data_face['boca']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 267:
                        self.data_face['boca']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 37:
                        self.data_face['boca']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 270:
                        self.data_face['boca']['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 40:
                        self.data_face['boca']['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 13:
                        self.data_face['boca']['pnt6'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 14:
                        self.data_face['boca']['pnt7'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 17:
                        self.data_face['boca']['pnt8'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 178:
                        self.data_face['boca']['pnt9'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 402:
                        self.data_face['boca']['pnt10'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 336:
                        self.data_face['cejas']['izquierda']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 334:
                        self.data_face['cejas']['izquierda']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 300:
                        self.data_face['cejas']['izquierda']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 283:
                        self.data_face['cejas']['izquierda']['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 285:
                        self.data_face['cejas']['izquierda']['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 107:
                        self.data_face['cejas']['derecha']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 105:
                        self.data_face['cejas']['derecha']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 70:
                        self.data_face['cejas']['derecha']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 53:
                        self.data_face['cejas']['derecha']['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 55:
                        self.data_face['cejas']['derecha']['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 1:
                        self.data_face['nariz']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 4:
                        self.data_face['nariz']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 5:
                        self.data_face['nariz']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 64:
                        self.data_face['nariz']['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 294:
                        self.data_face['nariz']['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 280:
                        self.data_face['mejillas']['izquierda']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 347:
                        self.data_face['mejillas']['izquierda']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 352:
                        self.data_face['mejillas']['izquierda']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 50:
                        self.data_face['mejillas']['derecha']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 118:
                        self.data_face['mejillas']['derecha']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 132:
                        self.data_face['mejillas']['derecha']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 151:
                        self.data_face['frente']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 10:
                        self.data_face['frente']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 9:
                        self.data_face['frente']['pnt3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 337:
                        self.data_face['frente']['pnt4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 108:
                        self.data_face['frente']['pnt5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 119:
                        self.data_face['menton']['pnt1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 32:
                        self.data_face['menton']['pnt2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 262:
                        self.data_face['menton']['pnt3'] = {'x': landmark.x, 'y': landmark.y}

        new_iteration = {
            'frame': len(self.data_face['iterations']) + 1,  # número de iteración (puedes modificarlo)
            'ojos': self.data_face['ojos'],
            'boca': self.data_face['boca'],
            'cejas': self.data_face['cejas'],
            'nariz': self.data_face['nariz'],
            'mejillas': self.data_face['mejillas'],
            'frente': self.data_face['frente'],
            'menton': self.data_face['menton']
        }

        self.data_face['iterations'].append(new_iteration)        

        # Guarda los cambios en el archivo JSON
        # with open("face_json.json", 'w') as json_file:
        #     json.dump(self.data_face, json_file, indent=4)

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

    def load_json(self, video_paths):

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
        video_started = [False, False, False]  # Flags to track when each video starts
        end_video = False

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
                else:
                    # If it has not started, it pauses in black until the synchronization is complete
                    frame = np.zeros((height, width, 3), dtype=np.uint8)  # Frame negro

                black_frame = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

                if cap_number == 0:  # First video (pose)
                    # print("cap 0")
                    if frame_number >= self.pose_sync:
                        if not video_started[cap_number]:  # Since frame 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True  # Marks the video as started

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_pose = self.pose.process(frame_rgb)
                        results_hands = self.hands_pose.process(frame_rgb)
                        self.process_pose(frame, results_pose)
                        self.process_hands(frame, results_hands) 

                        self.update_pose_json(results_pose, results_hands)
                    else:
                        frame = black_frame  # Black till synchronization of pose_sync

                elif cap_number == 1:  # Second video (hands)
                    # print("cap 1")
                    if frame_number >= self.hands_sync:
                        if not video_started[cap_number]: 
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True 

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_hands = self.hands_only.process(frame_rgb)
                        self.process_hands(frame, results_hands)

                        self.update_hands_json(results_hands)
                    else:
                        frame = black_frame  # Black till synchronization of hands_sync

                elif cap_number == 2:  # Third video (face)
                    # print("cap 2")
                    if frame_number >= self.face_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True  

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_face = self.face_mesh.process(frame_rgb)
                        self.process_face(frame, results_face)

                        self.update_face_json(results_face)
                    else:
                        frame = black_frame  # Black till synchronization of face_sync

                cap_number += 1

            if cv2.waitKey(1) & 0xFF == 27:  
                break

            if(frame_number % 400 == 0):
                with open("pose_json.json", 'w') as json_file:
                    json.dump(self.data_pose, json_file, indent=4)
                
                with open("hands_json.json", 'w') as json_file:
                    json.dump(self.data_hands, json_file, indent=4)

                with open("face_json.json", 'w') as json_file:
                    json.dump(self.data_face, json_file, indent=4)

            frame_number += 1

            print(frame_number)

            if end_video:
               break

        print("se acabó")
        with open("pose_json.json", 'w') as json_file:
            json.dump(self.data_pose, json_file, indent=4)
        
        with open("hands_json.json", 'w') as json_file:
            json.dump(self.data_hands, json_file, indent=4)

        with open("face_json.json", 'w') as json_file:
            json.dump(self.data_face, json_file, indent=4)

        for cap in caps:
            cap.release()
        out.release()

    def load_actions_from_json(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.json_path}. Trying alternative encoding...")
            with open(self.json_path, 'r', encoding='latin-1') as f:
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

    def print_action_for_frame(self, frame, frame_number):
        if frame_number in self.actions:
            actions = self.actions[frame_number]
            for i, action in enumerate(actions):
                cv2.putText(frame, action, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
    def create_json(self):
        print("Creando JSON...")

        # Pose JSON
        if os.path.exists("pose_json.json"):
            with open("pose_json.json", 'r') as json_file:
                self.data_pose = json.load(json_file)
        else:
            self.data_pose = {
                'iterations': [],
                'torso': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 5)},
                'brazos': {
                    'izquierdo': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 4)},
                    'derecho': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 4)}
                },
                'manos': {
                    'izquierda': {
                        **{f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 16)},  # Añadimos los puntos
                        'center': {'x': 0, 'y': 0}  # Luego añadimos el 'center'
                    },
                    'derecha': {
                        **{f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 16)},  # Añadimos los puntos
                        'center': {'x': 0, 'y': 0}  # Luego añadimos el 'center'
                    }
                }
            }

        with open("pose_json.json", 'w') as json_file:
            json.dump(self.data_pose, json_file, indent=4)

        # Face JSON
        if os.path.exists("face_json.json"):
            with open("face_json.json", 'r') as json_file:
                self.data_face = json.load(json_file)
        else:
            self.data_face = {
                'iterations': [],
                'ojos': {
                    'izquierdo': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 7)},
                    'derecho': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 7)}
                },
                'boca': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 11)},
                'cejas': {
                    'izquierda': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 6)},
                    'derecha': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 6)}
                },
                'nariz': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 6)},
                'mejillas': {
                    'izquierda': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 4)},
                    'derecha': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 4)}
                },
                'frente': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 4)},
                'menton': {f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 4)}
            }

        with open("face_json.json", 'w') as json_file:
            json.dump(self.data_face, json_file, indent=4)

        # Hands JSON
        if os.path.exists("hands_json.json"):
            with open("hands_json.json", 'r') as json_file:
                self.data_hands = json.load(json_file)
        else:
            self.data_hands = {
                'iterations': [],
                'izquierda': {
                    **{f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 16)},  # Añadir puntos
                    'center': {'x': 0, 'y': 0}  # Añadir 'center' dentro de 'izquierda'
                },
                'derecha': {
                    **{f'pnt{i}': {'x': 0, 'y': 0} for i in range(1, 16)},  # Añadir puntos
                    'center': {'x': 0, 'y': 0}  # Añadir 'center' dentro de 'derecha'
                }
            }

        with open("hands_json.json", 'w') as json_file:
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
        
        if combine_videos:
            if len(sys.argv) < 6:
                print("Faltan rutas de videos para combinar.")
            else:
                video_paths = sys.argv[2:5]
                processor.combine_videos(video_paths)
        elif load_json:
            video_paths = sys.argv[2:5]
            processor.load_json(video_paths)
        else:
            processor.process_video()

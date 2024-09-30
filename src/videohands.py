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
import copy


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
        if results_pose.pose_landmarks:
            for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                # Define los puntos clave que quieres guardar en el JSON.
                if idx == 11:  # kp1 en trunk
                    self.data_pose['trunk']['kp1'] = {'x': landmark.x, 'y': landmark.y}
                    self.data_pose['arms']['left']['kp1'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 12:  # kp2 en trunk
                    self.data_pose['trunk']['kp2'] = {'x': landmark.x, 'y': landmark.y}
                    self.data_pose['arms']['right']['kp1'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 23:
                    self.data_pose['trunk']['kp3'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 24:
                    self.data_pose['trunk']['kp4'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 13:  # kp1 en brazo left
                    self.data_pose['arms']['left']['kp2'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 15:
                    self.data_pose['arms']['left']['kp3'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 14:  # kp1 en brazo right
                    self.data_pose['arms']['right']['kp2'] = {'x': landmark.x, 'y': landmark.y}
                elif idx == 16:
                    self.data_pose['arms']['right']['kp3'] = {'x': landmark.x, 'y': landmark.y}

        if results_hands.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                handedness = results_hands.multi_handedness[hand_idx].classification[0].label

                # Decide si es mano left o right
                hand_key = 'left' if handedness == 'Left' else 'right'

                # Itera sobre los landmarks de la mano
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx == 0:
                        self.data_pose['hands'][hand_key]['kp1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 1:
                        self.data_pose['hands'][hand_key]['kp2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 2:
                        self.data_pose['hands'][hand_key]['kp3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 3:
                        self.data_pose['hands'][hand_key]['kp4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 4:
                        self.data_pose['hands'][hand_key]['kp5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 5:
                        self.data_pose['hands'][hand_key]['kp6'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 6:
                        self.data_pose['hands'][hand_key]['kp7'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 7:
                        self.data_pose['hands'][hand_key]['kp8'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 8:
                        self.data_pose['hands'][hand_key]['kp9'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 9:
                        self.data_pose['hands'][hand_key]['kp10'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 10:
                        self.data_pose['hands'][hand_key]['kp11'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 11: 
                        self.data_pose['hands'][hand_key]['kp12'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 12:
                        self.data_pose['hands'][hand_key]['kp13'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 13:
                        self.data_pose['hands'][hand_key]['kp14'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 14:
                        self.data_pose['hands'][hand_key]['kp15'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 15:
                        self.data_pose['hands'][hand_key]['kp16'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 16:
                        self.data_pose['hands'][hand_key]['kp17'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 17:
                        self.data_pose['hands'][hand_key]['kp18'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 18:
                        self.data_pose['hands'][hand_key]['kp19'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 19:
                        self.data_pose['hands'][hand_key]['kp20'] = {'x': landmark.x, 'y': landmark.y}

                center_x, center_y = self.calculate_center_of_mass(hand_landmarks.landmark)
                self.data_pose['hands'][hand_key]['center'] = {'x': center_x, 'y': center_y}
                
        new_iteration = {
            'frame': len(self.data_pose['iterations']) + 1,  # número de iteración (puedes modificarlo)
            'trunk': self.data_pose['trunk'].copy(),
            'arms': copy.deepcopy(self.data_pose['arms']),
            'hands': copy.deepcopy(self.data_pose['hands'])
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

                # Decide si es mano left o right
                hand_key = 'left' if handedness == 'Left' else 'right'

                # Itera sobre los landmarks de la mano
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx == 0:  # kp1 en la mano
                        self.data_hands[hand_key]['kp1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 1:  # kp2 en la mano
                        self.data_hands[hand_key]['kp2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 2:
                        self.data_hands[hand_key]['kp3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 3:
                        self.data_hands[hand_key]['kp4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 4:
                        self.data_hands[hand_key]['kp5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 5:
                        self.data_hands[hand_key]['kp6'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 6:
                        self.data_hands[hand_key]['kp7'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 7:
                        self.data_hands[hand_key]['kp8'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 8:
                        self.data_hands[hand_key]['kp9'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 9:
                        self.data_hands[hand_key]['kp10'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 10:
                        self.data_hands[hand_key]['kp11'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 11:
                        self.data_hands[hand_key]['kp12'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 12:
                        self.data_hands[hand_key]['kp13'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 13:
                        self.data_hands[hand_key]['kp14'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 14:
                        self.data_hands[hand_key]['kp15'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 15:
                        self.data_hands[hand_key]['kp16'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 16:
                        self.data_hands[hand_key]['kp17'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 17:
                        self.data_hands[hand_key]['kp18'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 18:
                        self.data_hands[hand_key]['kp19'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 19:
                        self.data_hands[hand_key]['kp20'] = {'x': landmark.x, 'y': landmark.y}

                # Calcula el centro de la mano (opcional)
                center_x, center_y = self.calculate_center_of_mass(hand_landmarks.landmark)
                self.data_hands[hand_key]['center'] = {'x': center_x, 'y': center_y}

        # Haz una copia profunda de los diccionarios 'left' y 'right' antes de agregarlos a la lista 'iterations'
        new_iteration = {
            'frame': len(self.data_hands['iterations']) + 1,  # número de iteración
            'left': self.data_hands['left'].copy(),  # Copia del diccionario 'left'
            'right': self.data_hands['right'].copy()  # Copia del diccionario 'right'
        }

        self.data_hands['iterations'].append(new_iteration)

        # Guarda los cambios en el archivo JSON
        # with open("hands_json.json", 'w') as json_file:
        #     json.dump(self.data_hands, json_file, indent=4)

    def gaze(self, frame, points): # LO HE SACADO DE https://github.com/amitt1236/Gaze_estimation/tree/master
        """
        The gaze function gets an image and face landmarks from mediapipe framework.
        The function draws the gaze direction into the frame.
        """

        '''
        2D image points.
        relative takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y) format
        '''
        image_points = np.array([
            relative(points.landmark[4], frame.shape),  # Nose tip
            relative(points.landmark[152], frame.shape),  # Chin
            relative(points.landmark[263], frame.shape),  # Left eye left corner
            relative(points.landmark[33], frame.shape),  # Right eye right corner
            relative(points.landmark[287], frame.shape),  # Left Mouth corner
            relative(points.landmark[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        '''
        2D image points.
        relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y,0) format
        '''
        image_points1 = np.array([
            relativeT(points.landmark[4], frame.shape),  # Nose tip
            relativeT(points.landmark[152], frame.shape),  # Chin
            relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
            relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
            relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
            relativeT(points.landmark[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
            (28.9, -28.9, -24.1)  # Right mouth corner
        ])

        '''
        3D model eye points
        The center of the eye ball
        '''
        Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

        '''
        camera matrix estimation
        '''
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                    dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # 2d pupil location
        # print("landmark length: ", len(points.landmark))

        left_pupil = relative(points.landmark[467], frame.shape)
        # right_pupil = relative(points.landmark[473], frame.shape)

        # Transformation between image point to world point
        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

        if transformation is not None:  # if estimateAffine3D secsseded
            # project pupil image point into 3d world point 
            pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

            # Project a 3D gaze direction onto the image plane.
            (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            # project 3D head pose into the image plane
            (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                            rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)
            # correct gaze for head rotation
            gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

            # Draw gaze line into screen
            p1 = (int(left_pupil[0]), int(left_pupil[1]))
            p2 = (int(gaze[0]), int(gaze[1]))
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            
    def process_face(self, frame, results_face):
        if results_face.multi_face_landmarks:
            self.gaze(frame, results_face.multi_face_landmarks[0])
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
                    if idx == 17:
                        self.data_face['face']['kp1'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 61:
                        self.data_face['face']['kp2'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 291:
                        self.data_face['face']['kp3'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 0:
                        self.data_face['face']['kp4'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 206:
                        self.data_face['face']['kp5'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 426:
                        self.data_face['face']['kp6'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 50:
                        self.data_face['face']['kp7'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 48:
                        self.data_face['face']['kp8'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 4:
                        self.data_face['face']['kp9'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 278:
                        self.data_face['face']['kp10'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 280:
                        self.data_face['face']['kp11'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 145:
                        self.data_face['face']['kp12'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 122:
                        self.data_face['face']['kp13'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 351:
                        self.data_face['face']['kp14'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 374:
                        self.data_face['face']['kp15'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 130:
                        self.data_face['face']['kp16'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 133:
                        self.data_face['face']['kp17'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 362:
                        self.data_face['face']['kp18'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 359:
                        self.data_face['face']['kp19'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 159:
                        self.data_face['face']['kp20'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 386:
                        self.data_face['face']['kp21'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 46:
                        self.data_face['face']['kp22'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 276:
                        self.data_face['face']['kp23'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 105:
                        self.data_face['face']['kp24'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 107:
                        self.data_face['face']['kp25'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 336:
                        self.data_face['face']['kp26'] = {'x': landmark.x, 'y': landmark.y}
                    elif idx == 334:
                        self.data_face['face']['kp27'] = {'x': landmark.x, 'y': landmark.y}
                    

        new_iteration = {
            'frame': len(self.data_face['iterations']) + 1,  # número de iteración (puedes modificarlo)
            'face': copy.deepcopy(self.data_face['face'])
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
                    if frame_number >= self.pose_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results_pose = self.pose.process(frame_rgb)
                        results_hands = self.hands_pose.process(frame_rgb)
                        self.process_pose(frame, results_pose)
                        self.process_hands(frame, results_hands)

                        self.update_pose_json(results_pose, results_hands)
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

                        self.update_face_json(results_face)
                    else:
                        frame = black_frame  # Black till synchronization of face_sync

                cap_number += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

            if(frame_number % 500 == 0):
                # Guarda todos los JSON cada 400 frames
                # print("Guardando JSON...")
                self.save_json_files() # GUARDO LA PRIMERA POSICION 400 VECES

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
        with open("pose_json.json", 'w') as json_file:
            json.dump(self.data_pose, json_file, indent=4)

        with open("hands_json.json", 'w') as json_file:
            json.dump(self.data_hands, json_file, indent=4)

        with open("face_json.json", 'w') as json_file:
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
                'trunk': {f'kp{i}': {'x': 0, 'y': 0} for i in range(1, 5)}, # cambialro luego todos
                'arms': {
                    'left': {f'kp{i}': {'x': 0, 'y': 0} for i in range(1, 4)},
                    'right': {f'kp{i}': {'x': 0, 'y': 0} for i in range(1, 4)}
                },
                'hands': {
                    'left': {
                        **{f'kp{i}': {'x': 0, 'y': 0} for i in range(1, 20)},  # Añadimos los puntos
                        'center': {'x': 0, 'y': 0}  # Luego añadimos el 'center'
                    },
                    'right': {
                        **{f'kp{i}': {'x': 0, 'y': 0} for i in range(1, 20)},  # Añadimos los puntos
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
                'face': {f'kp{i}': {'x': 0, 'y': 0} for i in range(1, 28)}
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
                'left': {
                    **{f'kp{i}': {'x': 0, 'y': 0} for i in range(1, 20)},  # Añadir puntos
                    'center': {'x': 0, 'y': 0}  # Añadir 'center' dentro de 'left'
                },
                'right': {
                    **{f'kp{i}': {'x': 0, 'y': 0} for i in range(1, 20)},  # Añadir puntos
                    'center': {'x': 0, 'y': 0}  # Añadir 'center' dentro de 'right'
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

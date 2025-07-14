from ultralytics import YOLO
import cv2
import json
import numpy as np
import mediapipe as mp
import sys
import os
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

class PhoneDetector:
    def __init__(self):
        self.model = YOLO("yolov8x.pt")

    def detect(self, image):
        results = self.model(image, verbose=False)

        for result in results:
            boxes = result.boxes
            for cls, box in zip(boxes.cls, boxes.xyxy):
                if int(cls) == 67:  # 67 = cell phone in COCO
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    normalized_area = area / (image.shape[0] * image.shape[1])
                    normalized_area_S = f"{normalized_area:.2f}"  
                    
                    if normalized_area < 0.05:
                        return 1
                    else:
                        break
        # cv2.imshow("Phone", image)
        # cv2.waitKey(1)
        return 0

    
global frames_with_phone_dectected_well 
global frames_with_phone_dectected_bad 
global frames_without_phone_dectected_well 
global frames_without_phone_dectected_bad 

frames_with_phone_dectected_well = 0
frames_with_phone_dectected_bad = 0
frames_without_phone_dectected_well = 0
frames_without_phone_dectected_bad = 0

global is_phone
is_phone = None
    
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

    def load_json(self, video_paths, phone_detector):

        global frames_with_phone_dectected_well 
        global frames_with_phone_dectected_bad 
        global frames_without_phone_dectected_well 
        global frames_without_phone_dectected_bad 
        global is_phone

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

                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) 

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
                    frame = np.zeros((height, width, 3), dtype=np.uint8)

                black_frame = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

                if cap_number == 0:  # First video (pose)
                    if frame_number >= self.pose_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        is_phone = phone_detector.detect(frame.copy())  

                    else:
                        frame = black_frame  # Black till synchronization of pose_sync

                elif cap_number == 1:  # Second video (hands)
                    if frame_number >= self.hands_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    else:
                        frame = black_frame  # Black till synchronization of hands_sync

                elif cap_number == 2:  # Third video (face)
                    if frame_number >= self.face_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame = black_frame  # Black till synchronization of face_sync

                cap_number += 1

            actions = self.actions.get(frame_number, [])

            phone_actions = {
                "driver_actions/phonecall_right",
                "driver_actions/phonecall_left",
                "driver_actions/texting_left",
                "driver_actions/texting_right"
            }

            found = False
            for action in actions:
                if action in phone_actions:
                    if is_phone == 1:
                        frames_with_phone_dectected_well += 1
                    else:
                        frames_with_phone_dectected_bad += 1
                    found = True
                    break  
            if not found:
                if is_phone == 1:
                    frames_without_phone_dectected_bad += 1
                else:
                    frames_without_phone_dectected_well += 1


            is_phone = None  # Reset


            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_number += 1

            if end_video:
                break

        for cap in caps:
            cap.release()
        out.release()

        TP = frames_with_phone_dectected_well

        FN = frames_with_phone_dectected_bad

        TN = frames_without_phone_dectected_well

        FP = frames_without_phone_dectected_bad

        conf_matrix = np.array([
            [TN, FP],  
            [FN, TP]  
        ])

        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Teléfono", "Teléfono"],
                    yticklabels=["No Teléfono", "Teléfono"])
        plt.xlabel("Predicción del modelo")
        plt.ylabel("Realidad (anotado)")
        plt.title("Matriz de confusión")
        plt.tight_layout()
        plt.show()

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

        video_paths = sys.argv[2:5]
        processor.load_json(video_paths, phone_detector)

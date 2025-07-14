import torch
import requests
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

from PIL import Image

from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands 
hands_pose = mp.solutions.hands.Hands() 
hands_only = mp.solutions.hands.Hands() 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

default_pose_colors = {
    'LEFT_ARM': (255, 0, 0),
    'RIGHT_ARM': (0, 255, 0),
    'LEFT_LEG': (0, 0, 255),
    'RIGHT_LEG': (255, 255, 0),
    'TRUNK': (255, 0, 255),
    'HEAD': (0, 255, 255)
}

pose_colors = None

pose_colors = pose_colors if pose_colors is not None else default_pose_colors

left_elbow_vitpose = []
right_elbow_vitpose = []
left_shoulder_vitpose = []
right_shoulder_vitpose = []
left_wrist_vitpose = []
right_wrist_vitpose = []
left_hip_vitpose = []
right_hip_vitpose = []

left_elbow_mediapipe = []
right_elbow_mediapipe = []
left_shoulder_mediapipe = []
right_shoulder_mediapipe = []
left_wrist_mediapipe = []
right_wrist_mediapipe = []
left_hip_mediapipe = []
right_hip_mediapipe = []

distance_left_shoulder = []
distance_right_shoulder = []
distance_left_elbow = []
distance_right_elbow = []
distance_left_wrist = []
distance_right_wrist = []
distance_left_hip = []
distance_right_hip = []

frame_width = 1280
frame_height = 720

not_found_mediapipe = 0
not_found_vitpose = 0
total_frames = 0

def draw_points(result, image, keypoints, scores, pose_keypoint_color, keypoint_score_threshold, radius, show_keypoint_weight):
    print(f"Number of keypoints detected: {len(keypoints)}")
    if pose_keypoint_color is not None:
        assert len(pose_keypoint_color) == len(keypoints)
    for kid, (kpt, kpt_score) in enumerate(zip(keypoints, scores)):
        x_coord, y_coord = int(kpt[0]), int(kpt[1])
        if kid == 5:
            left_shoulder_vitpose.append((x_coord, y_coord))
        if kid == 6:
            right_shoulder_vitpose.append((x_coord, y_coord))
        if kid == 7:
            left_elbow_vitpose.append((x_coord, y_coord))
        if kid == 8:
            right_elbow_vitpose.append((x_coord, y_coord))
        if kid == 9:
            left_wrist_vitpose.append((x_coord, y_coord))
        if kid == 10:
            right_wrist_vitpose.append((x_coord, y_coord))
        if kid == 11:
            left_hip_vitpose.append((x_coord, y_coord))
        if kid == 12:
            right_hip_vitpose.append((x_coord, y_coord))

        if kpt_score > keypoint_score_threshold:
            color = tuple(int(c) for c in pose_keypoint_color[kid])
            if show_keypoint_weight:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)
                cv2.circle(result, (int(x_coord), int(y_coord)), radius, color, -1)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
                cv2.addWeighted(result, transparency, result, 1 - transparency, 0, dst=result)
            else:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)
                cv2.circle(result, (int(x_coord), int(y_coord)), radius, color, -1)

    return result

def draw_links(result, image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold, thickness, show_keypoint_weight, stick_width = 2):
    height, width, _ = image.shape
    if keypoint_edges is not None and link_colors is not None:
        assert len(link_colors) == len(keypoint_edges)
        for sk_id, sk in enumerate(keypoint_edges):

            if sk_id < 4:
                continue

            x1, y1, score1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]), scores[sk[0]])
            x2, y2, score2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]), scores[sk[1]])
            if (
                x1 > 0
                and x1 < width
                and y1 > 0
                and y1 < height
                and x2 > 0
                and x2 < width
                and y2 > 0
                and y2 < height
                and score1 > keypoint_score_threshold
                and score2 > keypoint_score_threshold
            ):
                color = tuple(int(c) for c in link_colors[sk_id])
                if show_keypoint_weight:
                    X = (x1, x2)
                    Y = (y1, y2)
                    mean_x = np.mean(X)
                    mean_y = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    polygon = cv2.ellipse2Poly(
                        (int(mean_x), int(mean_y)), (int(length / 2), int(stick_width)), int(angle), 0, 360, 1
                    )
                    cv2.fillConvexPoly(image, polygon, color)
                    cv2.fillConvexPoly(result, polygon, [255, 0, 0])
                    transparency = max(0, min(1, 0.5 * (keypoints[sk[0], 2] + keypoints[sk[1], 2])))
                    cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
                    cv2.addWeighted(result, transparency, result, 1 - transparency, 0, dst=result)
                else:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)
                    cv2.line(result, (x1, y1), (x2, y2), [255, 0, 0], thickness=thickness)

    return result

def calculate_center_of_mass( landmarks):
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]

    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)

    return center_x, center_y

def process_pose(result, frame, results_pose):
    print("frame", frame.shape)
    if results_pose.pose_landmarks:
        counter_mini = 0
        for landmark in results_pose.pose_landmarks.landmark:
            if counter_mini == 11:
                left_shoulder_mediapipe.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
            if counter_mini == 12:
                right_shoulder_mediapipe.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
            if counter_mini == 13:
                left_elbow_mediapipe.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
            if counter_mini == 14:
                right_elbow_mediapipe.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
            if counter_mini == 15:
                left_wrist_mediapipe.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
            if counter_mini == 16:
                right_wrist_mediapipe.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
            if counter_mini == 23:
                left_hip_mediapipe.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
            if counter_mini == 24:
                right_hip_mediapipe.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))
            counter_mini += 1
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            if start_idx in mp_pose.PoseLandmark.__members__.values() and end_idx in mp_pose.PoseLandmark.__members__.values():
                start_landmark = results_pose.pose_landmarks.landmark[start_idx]
                end_landmark = results_pose.pose_landmarks.landmark[end_idx]

                if start_idx in [11, 13, 15, 17, 19, 21] and end_idx in [11, 13, 15, 17, 19, 21]:
                    color = pose_colors['LEFT_ARM']
                elif start_idx in [12, 14, 16, 18, 20, 22] and end_idx in [12, 14, 16, 18, 20, 22]:
                    color = pose_colors['RIGHT_ARM']
                elif start_idx in [11, 12, 24, 23] and end_idx in [11, 12, 24, 23]:
                    color = pose_colors['TRUNK']
                elif start_idx in [23, 25, 27, 29, 31, 33] and end_idx in [23, 25, 27, 29, 31, 33]:
                    continue
                elif start_idx in [24, 26, 28, 30, 32, 34] and end_idx in [24, 26, 28, 30, 32, 34]:
                    continue
                elif start_idx in [0, 1, 15, 16] and end_idx in [0, 1, 15, 16]:
                    color = pose_colors['HEAD']
                else:
                    continue
                
                start_point = (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0]))
                end_point = (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, color, 2)
                cv2.line(result, start_point, end_point, [0, 0, 255], 2)

    return result

def process_hands(result, frame, results_hands):
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            center_x, center_y = calculate_center_of_mass(hand_landmarks.landmark)
            center_point = (int(center_x * frame.shape[1]), int(center_y * frame.shape[0]))

            cv2.circle(frame, center_point, 20, (255, 128, 64), -1)
            cv2.circle(result, center_point, 20, (0, 0, 255), -1)

    return result

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def compute_normalized_distances(vitpose_list, mediapipe_list, distance_list):
    global total_frames, not_found_vitpose, not_found_mediapipe
    for vit, mediapipe in zip(vitpose_list, mediapipe_list):
        total_frames += 1
        if all(vit) == 0 or all(mediapipe) == 0:
            if all(vit) == 0:
                not_found_vitpose += 1
            if all(mediapipe) == 0:
                not_found_mediapipe += 1
            continue
        x_vit, y_vit = vit[0] / frame_width, vit[1] / frame_height
        x_mediapipe, y_mediapipe = mediapipe[0] / frame_width, mediapipe[1] / frame_height
        distance = euclidean_distance((x_vit, y_vit), (x_mediapipe, y_mediapipe))
        distance_list.append(distance)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ]
)

link_colors = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
keypoint_colors = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

cap = cv2.VideoCapture("prueba_vitpose.mp4")

while cap.isOpened():
    ret,frame = cap.read()

    if not ret:
        break

    image = Image.fromarray(frame)

    inputs = person_image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    )
    result = results[0]

    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
    image_pose_result = pose_results[0]

    keypoint_edges = model.config.edges

    numpy_image = np.array(image)
    frame_image = np.array(image)
    result_image = np.array(image)

    if len(image_pose_result) == 0:
        print("No people detected.")
        left_shoulder_vitpose.append((0, 0))
        right_shoulder_vitpose.append((0, 0))
        left_elbow_vitpose.append((0, 0))
        right_elbow_vitpose.append((0, 0))
        left_wrist_vitpose.append((0, 0))
        right_wrist_vitpose.append((0, 0))
        left_hip_vitpose.append((0, 0))
        right_hip_vitpose.append((0, 0))

    else:
        print(f"Number of people detected: {len(image_pose_result)}")

        best_pose_result = max(image_pose_result, key=lambda p: sum(p["scores"]))

        scores = np.array(best_pose_result["scores"])
        keypoints = np.array(best_pose_result["keypoints"])

        result_image = draw_points(
            result_image, numpy_image, keypoints, scores,
            keypoint_colors, keypoint_score_threshold=0.3,
            radius=2, show_keypoint_weight=False
        )

        result_image = draw_links(
            result_image, numpy_image, keypoints, scores,
            keypoint_edges, link_colors, keypoint_score_threshold=0.3,
            thickness=3, show_keypoint_weight=False
        )

    frame = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame)
    results_hands = hands_pose.process(frame)

    if results_pose.pose_landmarks:
        result_image = process_pose(result_image, frame, results_pose)
    else:
        print("No pose landmarks detected.")
        left_shoulder_mediapipe.append((0, 0))
        right_shoulder_mediapipe.append((0, 0))
        left_elbow_mediapipe.append((0, 0))
        right_elbow_mediapipe.append((0, 0))
        left_wrist_mediapipe.append((0, 0))
        right_wrist_mediapipe.append((0, 0))
        left_hip_mediapipe.append((0, 0))
        right_hip_mediapipe.append((0, 0))

    cv2.imshow("Pose estimation", result_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

compute_normalized_distances(left_shoulder_vitpose, left_shoulder_mediapipe, distance_left_shoulder)
compute_normalized_distances(right_shoulder_vitpose, right_shoulder_mediapipe, distance_right_shoulder)
compute_normalized_distances(left_elbow_vitpose, left_elbow_mediapipe, distance_left_elbow)
compute_normalized_distances(right_elbow_vitpose, right_elbow_mediapipe, distance_right_elbow)
compute_normalized_distances(left_wrist_vitpose, left_wrist_mediapipe, distance_left_wrist)
compute_normalized_distances(right_wrist_vitpose, right_wrist_mediapipe, distance_right_wrist)
compute_normalized_distances(left_hip_vitpose, left_hip_mediapipe, distance_left_hip)
compute_normalized_distances(right_hip_vitpose, right_hip_mediapipe, distance_right_hip)

mean_distance_left_shoulder = sum(distance_left_shoulder) / len(distance_left_shoulder)
mean_distance_right_shoulder = sum(distance_right_shoulder) / len(distance_right_shoulder)
mean_distance_left_elbow = sum(distance_left_elbow) / len(distance_left_elbow)
mean_distance_right_elbow = sum(distance_right_elbow) / len(distance_right_elbow)
mean_distance_left_wrist = sum(distance_left_wrist) / len(distance_left_wrist)
mean_distance_right_wrist = sum(distance_right_wrist) / len(distance_right_wrist)
mean_distance_left_hip = sum(distance_left_hip) / len(distance_left_hip)
mean_distance_right_hip = sum(distance_right_hip) / len(distance_right_hip)

print("Missing vitpose frames percentage: ", (not_found_vitpose / total_frames) * 100)
print("Missing mediapipe frames percentage: ", (not_found_mediapipe / total_frames) * 100)

labels = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip'
]

mean_distances = [
    mean_distance_left_shoulder,
    mean_distance_right_shoulder,
    mean_distance_left_elbow,
    mean_distance_right_elbow,
    mean_distance_left_wrist,
    mean_distance_right_wrist,
    mean_distance_left_hip,
    mean_distance_right_hip
]

import matplotlib.pyplot as plt

all_distances = [
    distance_left_shoulder,
    distance_right_shoulder,
    distance_left_elbow,
    distance_right_elbow,
    distance_left_wrist,
    distance_right_wrist,
    distance_left_hip,
    distance_right_hip
]

labels = [
    'Hombro Izq.', 'Hombro Der.',
    'Codo Izq.', 'Codo Der.',
    'Muñeca Izq.', 'Muñeca Der.',
    'Cadera Izq.', 'Cadera Der.'
]

plt.figure(figsize=(12, 6))
plt.boxplot(all_distances, labels=labels, patch_artist=True,
            boxprops=dict(facecolor='skyblue'), showfliers=False)

plt.title('Distribución de Distancias por Articulación')
plt.xlabel('Articulación')
plt.ylabel('Distancia Normalizada')

plt.tight_layout()
plt.show()
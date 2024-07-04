import cv2
import mediapipe as mp
import sys

def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    pose_colors = { # colores para las partes del cuerpo
        'LEFT_ARM': (255, 0, 0),
        'RIGHT_ARM': (0, 255, 0),
        'LEFT_LEG': (0, 0, 255),
        'RIGHT_LEG': (255, 255, 0),
        'TORSO': (255, 0, 255),
        'HEAD': (0, 255, 255)

    }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # open cv lee en BGR, y el video es RGB

        results_pose = pose.process(frame_rgb) # detectamos la pose y las manos
        results_hands = hands.process(frame_rgb) 

        if results_pose.pose_landmarks: # pintamos la pose en el frame

            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                if start_idx in mp_pose.PoseLandmark.__members__.values() and end_idx in mp_pose.PoseLandmark.__members__.values():
                    start_landmark = results_pose.pose_landmarks.landmark[start_idx]
                    end_landmark = results_pose.pose_landmarks.landmark[end_idx]

                    # Determinar el color basado en la parte del cuerpo
                    if start_idx in [11, 13, 15, 17, 19, 21] and end_idx in [11, 13, 15, 17, 19, 21]:
                        color = pose_colors['LEFT_ARM']
                    elif start_idx in [12, 14, 16, 18, 20, 22] and end_idx in [12, 14, 16, 18, 20, 22]:
                        color = pose_colors['RIGHT_ARM']
                    elif start_idx in [23, 25, 27, 29, 31] and end_idx in [23, 25, 27, 29, 31]:
                        color = pose_colors['LEFT_LEG']
                    elif start_idx in [24, 26, 28, 30, 32] and end_idx in [24, 26, 28, 30, 32]:
                        color = pose_colors['RIGHT_LEG']
                    elif start_idx in [11, 12, 24, 23] and end_idx in [11, 12, 24, 23]:
                        color = pose_colors['TORSO']
                    else:
                        color = pose_colors['HEAD']

                    start_point = (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0]))
                    end_point = (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, color, 2)

            # mp_drawing.draw_landmarks(
            #     frame,
            #     results_pose.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2)
            # )

        if results_hands.multi_hand_landmarks: # pintamos las manos en el frame
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow('MediaPipe Hands + Pose', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("falta el video cazurro")
    else:
        video_path = sys.argv[1]
        process_video(video_path)

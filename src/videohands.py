import cv2
import mediapipe as mp
import sys

def process_video(video_path):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # open cv lee en BGR, y el video es RGB

        results = pose.process(frame_rgb) # detecta la pose en el frame

        if results.pose_landmarks: # pintamos la pose en el frame
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        # cv2.imshow('MediaPipe Pose', frame) # mostramos el frame con la pose

        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

        # results = hands.process(frame_rgb) # detecta las manos en el frame

        # if results.multi_hand_landmarks: # pintamos las manos en el frame
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(
        #             frame, 
        #             hand_landmarks, 
        #             mp_hands.HAND_CONNECTIONS
        #         )

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


# la mano izquierda la pilla mucho mejor lo de hands, lo de pose falla bastante más. Face detection va bastante mal
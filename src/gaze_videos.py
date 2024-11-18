import cv2
import mediapipe as mp
import gaze

class VideoProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_full = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh_cropped = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                                                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def process_face(self, frame, results_face):
        """Dibuja los puntos clave de la cara en el frame."""
        if results_face.multi_face_landmarks:
            p1, p2 = gaze.gaze(frame, results_face.multi_face_landmarks[0])

            for face_landmarks in results_face.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

    def split_video_with_face_detection(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error al abrir el video")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        tercio_start = frame_width // 3
        tercio_end = 2 * frame_width // 3

        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_frame = frame[:, tercio_start:tercio_end].copy()
            rgb_cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            results_face_full = self.face_mesh_full.process(rgb_frame)
            results_face_cropped = self.face_mesh_cropped.process(rgb_cropped_frame)

            self.process_face(frame, results_face_full)
            self.process_face(cropped_frame, results_face_cropped)

            cv2.imshow('Original Video with Face Mesh', frame)
            cv2.imshow('Second Third Video with Face Mesh', cropped_frame)

            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Ruta al video
video_path = 'face.mp4'
processor = VideoProcessor()
processor.split_video_with_face_detection(video_path)

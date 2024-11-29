import json
import pandas as pd
import os
import sys
import numpy as np
import cv2  
import time
import matplotlib.pyplot as plt

class Balancer_plot:
    def __init__(self):
        self.total_frames_per_action = {}
        self.global_frame_stats = {
            'hands_using_wheel/both': {},
            'hands_using_wheel/only_left': {},
            'hands_using_wheel/only_right': {},
            'driver_actions/radio': {},
            'driver_actions/drinking': {},
            'driver_actions/reach_side': {},
            'driver_actions/phonecall_right': {},
            'driver_actions/texting_left': {}
        }


    def create_directories(self, base_path):
        actions = ['hands_using_wheel/both', 'hands_using_wheel/only_left', 'hands_using_wheel/only_right', 'driver_actions/radio', 'driver_actions/drinking', 'driver_actions/reach_side', 'driver_actions/phonecall_right', 'driver_actions/texting_left']
        for action in actions:
            action_path = os.path.join(base_path, action)
            os.makedirs(action_path, exist_ok=True)

    def check_hand(self, data_hands_pose, data_pose):
        new_pose = np.array(data_pose)
        x_pose = new_pose[:, 0]
        y_pose = new_pose[:, 1]

        new_hand_pose = np.array(data_hands_pose)
        x_hand_pose = new_hand_pose[:, 0]
        y_hand_pose = new_hand_pose[:, 1]

        if np.any(x_pose == 0.0) or np.any(y_pose == 0.0) or np.any(x_hand_pose == 0.0) or np.any(y_hand_pose == 0.0):
            return False

        return True

    def check_rest(self, data):
        new_data = np.array(data)
        data_x = new_data[:, 0]
        data_y = new_data[:, 1]

        if np.any(data_x == 0.0) or np.any(data_y == 0.0):
            return False
        
        return True

    def balance_frames(self, json_file_groups):
        combined_balanced_df = pd.DataFrame()
        global_frame_count = {action_type: 0 for action_type in ['hands_using_wheel/both', 'hands_using_wheel/only_left', 'hands_using_wheel/only_right', 'driver_actions/radio', 'driver_actions/drinking', 'driver_actions/reach_side', 'driver_actions/phonecall_right', 'driver_actions/texting_left']}

        for json_file_set in json_file_groups:

            photos_to_save = []
            frame_seen = []

            subdir_path = json_file_set['subdirectory']

            if subdir_path not in self.global_frame_stats:
                for action in global_frame_count.keys():
                    self.global_frame_stats[action][subdir_path] = {'total': 0, 'valid': 0}

            frames_file_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'frames.json'), None)
            hands_file_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'hands.json'), None)
            face_file_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'face.json'), None)
            pose_file_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'pose.json'), None)
            pose_video_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'pose.mp4'), None)
            hands_video_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'hands.mp4'), None)
            face_video_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'face.mp4'), None)

            with open(hands_file_path, 'r') as file:
                data_hands = json.load(file)
            with open(face_file_path, 'r') as file:
                data_face = json.load(file)
            with open(pose_file_path, 'r') as file:
                data_pose = json.load(file)

            if not frames_file_path:
                print(f"No se encontró 'frames.json' en el subdirectorio: {subdir_path}")
                continue

            with open(frames_file_path, 'r') as file:
                data = json.load(file)

            actions = data['openlabel']['actions']
            action_data = []

            for frame_id, frame_data in data["openlabel"]["streams"].items():
                if "face_camera" in frame_id:
                    face_sync = frame_data["stream_properties"]["sync"]["frame_shift"]
                elif "hands_camera" in frame_id:
                    hands_sync = frame_data["stream_properties"]["sync"]["frame_shift"]
                elif "body_camera" in frame_id:
                    pose_sync = frame_data["stream_properties"]["sync"]["frame_shift"]

            print(f"Sync ORIGINAL: {hands_sync}, {face_sync}, {pose_sync}")

            for key, action in actions.items():
                action_type = action['type']
                if action_type in global_frame_count:
                    for interval in action['frame_intervals']:
                        frame_start = interval['frame_start']
                        frame_end = interval['frame_end']

                        for i in range(frame_start, frame_end + 1):
                            if (i >= len(data_hands['iterations']) or i >= len(data_face['iterations']) or i >= len(data_pose['iterations'])):
                                continue

                            self.global_frame_stats[action_type][subdir_path]['total'] += 1

                            face_frame = i - face_sync
                            hands_frame = i - hands_sync
                            pose_frame = i - pose_sync

                            if face_frame < 0 or hands_frame < 0 or pose_frame < 0:
                                print(f"Frame fuera de rango: {i}, sync: {sync}")
                                continue

                            d_hands = data_hands['iterations'][hands_frame]['hands']
                            d_face = data_face['iterations'][face_frame]['face']
                            d_pose = data_pose['iterations'][pose_frame]['pose']

                            left_hand = d_hands[21:]
                            right_hand = d_hands[0:21]
                            pose_body = d_pose[0:8]
                            left_hand_pose = d_pose[29:50]
                            right_hand_pose = d_pose[8:29]

                            if (
                                (action_type == 'hands_using_wheel/both' and 
                                self.check_hand(left_hand_pose, pose_body) and 
                                self.check_hand(right_hand_pose, pose_body)) or
                                (action_type == 'hands_using_wheel/only_left' and 
                                self.check_hand(left_hand_pose, pose_body)) or
                                (action_type == 'hands_using_wheel/only_right' and 
                                self.check_hand(right_hand_pose, pose_body)) or 
                                (action_type == 'driver_actions/radio' and
                                self.check_hand(right_hand_pose, pose_body)) or
                                (action_type == 'driver_actions/drinking' and
                                self.check_hand(right_hand_pose, pose_body)) or
                                (action_type == 'driver_actions/reach_side' and
                                self.check_hand(right_hand_pose, pose_body)) or
                                (action_type == 'driver_actions/phonecall_right' and
                                self.check_hand(right_hand_pose, pose_body)) or
                                (action_type == 'driver_actions/texting_left' and
                                self.check_hand(left_hand_pose, pose_body))
                            ):
                                if i in frame_seen:
                                    continue

                                self.global_frame_stats[action_type][subdir_path]['valid'] += 1
                                frame_seen.append(i)
                                global_frame_count[action_type] += 1

            print("Total frames per action: ", global_frame_count)

            df = pd.DataFrame(action_data)
            combined_balanced_df = pd.concat([combined_balanced_df, df], ignore_index=True)

            sync = [hands_sync, face_sync, pose_sync]

        balanced_json = combined_balanced_df.to_dict(orient='records')

        print("Global frame stats: ", self.global_frame_stats)

        action_totals = {}
        for action, sessions in self.global_frame_stats.items():
            total_valid = sum(session['valid'] for session in sessions.values())
            total_total = sum(session['total'] for session in sessions.values())
            action_totals[action] = {'valid': total_valid, 'total': total_total}

        actions = list(action_totals.keys())
        valid_counts = [action_totals[action]['valid'] for action in actions]
        total_counts = [action_totals[action]['total'] for action in actions]

        x = range(len(actions))
        width = 0.35

        plt.bar(x, total_counts, width, label='Total', color='skyblue', alpha=0.7)
        plt.bar(x, valid_counts, width, label='Valid', color='green', alpha=0.7, bottom=[0]*len(x))

        # Configuración del gráfico
        plt.xticks(x, actions, rotation=45, ha='right')
        plt.xlabel('Acción')
        plt.ylabel('Cantidad')
        plt.title('Totales y válidos por acción')
        plt.legend()

        # Mostrar el gráfico
        plt.tight_layout()
        plt.show()



    def get_json_files(self, directory_path):
        json_file_groups = []
        for subdir, _, files in os.walk(directory_path):
            json_files = [os.path.join(subdir, f) for f in files if f.endswith('.json') or f.endswith('.mp4')]
            if json_files:
                json_file_groups.append({
                    'subdirectory': subdir,
                    'files': json_files
                })
        return json_file_groups

    def save_images(self, photos_to_save, subdir_path, base_output_path, pose_video_path, hands_video_path, face_video_path, sync):
        for photo in photos_to_save:
            frame = photo[0]
            action_type = photo[1]
            action_subdir = os.path.join(base_output_path, action_type)

            new_d = subdir_path.split('/')[1]
            cap_pose = cv2.VideoCapture(pose_video_path)
            cap_hands = cv2.VideoCapture(hands_video_path)
            cap_face = cv2.VideoCapture(face_video_path)

            if not cap_pose.isOpened():
                print(f"Error al abrir el video de pose: {pose_video_path}")
                continue
            if not cap_hands.isOpened():
                print(f"Error al abrir el video de manos: {hands_video_path}")
                continue
            if not cap_face.isOpened():
                print(f"Error al abrir el video de cara: {face_video_path}")
                continue

            pose_frame = max(0, frame - sync[2])
            hands_frame = max(0, frame - sync[0])
            face_frame = max(0, frame - sync[1])

            cap_pose.set(cv2.CAP_PROP_POS_FRAMES, pose_frame)
            cap_hands.set(cv2.CAP_PROP_POS_FRAMES, hands_frame)
            cap_face.set(cv2.CAP_PROP_POS_FRAMES, face_frame)

            ret_pose, frame_pose = cap_pose.read()
            ret_hands, frame_hands = cap_hands.read()
            ret_face, frame_face = cap_face.read()

            if not ret_pose:
                print(f"Error al leer el frame de pose en {pose_frame}")
            if not ret_hands:
                print(f"Error al leer el frame de manos en {hands_frame}")
            if not ret_face:
                print(f"Error al leer el frame de cara en {face_frame}")
            
            if not (ret_pose and ret_hands and ret_face):
                print(f"Error al leer los frames en {frame} para {action_type}")
                continue

            pose_image_path = os.path.join(action_subdir, f"{new_d}_{frame}_pose.png")
            hands_image_path = os.path.join(action_subdir, f"{new_d}_{frame}_hands.png")
            face_image_path = os.path.join(action_subdir, f"{new_d}_{frame}_face.png")

            if not cv2.imwrite(pose_image_path, frame_pose):
                print(f"Error al guardar la imagen de pose: {pose_image_path}")

            if not cv2.imwrite(hands_image_path, frame_hands):
                print(f"Error al guardar la imagen de manos: {hands_image_path}")

            if not cv2.imwrite(face_image_path, frame_face):
                print(f"Error al guardar la imagen de cara: {face_image_path}")

            # Liberar los capturadores de video
            cap_pose.release()
            cap_hands.release()
            cap_face.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python balancer.py dir")
    else:
        directory_path = sys.argv[1]
        # output_file = sys.argv[2]
        base_output_path = "balance_prueba"
        
        balancer = Balancer_plot()
        balancer.create_directories(base_output_path)
        json_file_groups = balancer.get_json_files(directory_path)
        balancer.balance_frames(json_file_groups)



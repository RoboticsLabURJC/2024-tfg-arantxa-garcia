"""


A ELIMINAR


"""

import json
import pandas as pd
import os
import sys

class Balancer_v2:
    def __init__(self):
        # Diccionario para llevar el total de frames por tipo de acción
        self.total_frames_per_action = {}

    def check_left_hand(self, data_hands, iteration): # que cámara habría que checkear para las manos?? ambas??
        print(f"Checking left hand in iteration {iteration}")
        # print(data_hands)
        left_hand = data_hands['iterations'][iteration]['left']['kp1']['x']

        if left_hand == 0:
            return False
        return True
    
    def check_right_hand(self, data_hands, iteration):
        right_hand = data_hands['iterations'][iteration]['right']['kp1']['x']

        if right_hand == 0:
            return False
        return True

    def check_face(self, data_face, iteration):
        face = data_face['iterations'][iteration]['face']['kp1']['x']

        if face == 0:
            return False
        return True

    def check_left_arm(self, data_pose, iteration):
        pose = data_pose['iterations'][iteration]['arms']['left']['kp1']['x']

        if pose == 0:
            return False
        return True        

    def check_right_arm(self, data_pose, iteration):
        pose = data_pose['iterations'][iteration]['arms']['right']['kp1']['x']

        if pose == 0:
            return False
        
        return True

    def check_trunk(self, data_pose, iteration):
        pose = data_pose['iterations'][iteration]['trunk']['kp1']['x']

        if pose == 0:
            return False
        
        return True

    def balance_frames(self, json_file_groups):
        print(f"Balanceando frames de {len(json_file_groups)} subdirectorios...")
        combined_balanced_df = pd.DataFrame()
        frame_limit = 100

        # Limitar a 100 frames globalmente para cada tipo de acción
        global_frame_count = {action_type: 0 for action_type in ['hands_using_wheel/both', 'hands_using_wheel/only_left', 'hands_using_wheel/only_right']}

        # Procesar cada subdirectorio de archivos JSON
        for json_file_set in json_file_groups:
            print(f"Procesando subdirectorio: {json_file_set['subdirectory']}")
            subdir_path = json_file_set['subdirectory']
            frames_file_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'frames.json'), None)
            hands_file_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'hands.json'), None)
            face_file_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'face.json'), None)
            pose_file_path = next((f for f in json_file_set['files'] if os.path.basename(f) == 'pose.json'), None)

            with open(hands_file_path, 'r') as file:
                data_hands = json.load(file)
            with open(face_file_path, 'r') as file:
                data_face = json.load(file)
            with open(pose_file_path, 'r') as file:
                data_pose = json.load(file)

            if not frames_file_path:
                print(f"No se encontró 'frames.json' en el subdirectorio: {subdir_path}")
                continue

            # Leer los datos del archivo 'frames.json'
            with open(frames_file_path, 'r') as file:
                data = json.load(file)

            actions = data['openlabel']['actions']
            action_data = []

            # Procesar la sincronización de las cámaras
            for frame_id, frame_data in data["openlabel"]["streams"].items():
                if "face_camera" in frame_id:
                    face_sync = frame_data["stream_properties"]["sync"]["frame_shift"]
                elif "hands_camera" in frame_id:
                    hands_sync = frame_data["stream_properties"]["sync"]["frame_shift"]
                elif "body_camera" in frame_id:
                    pose_sync = frame_data["stream_properties"]["sync"]["frame_shift"]

            print(f"Sincronización de cámaras: face={face_sync}, hands={hands_sync}, pose={pose_sync}")

            # Filtrar las acciones de interés y sus intervalos de frames
            for key, action in actions.items():
                action_type = action['type']
                if action_type in global_frame_count and global_frame_count[action_type] < frame_limit:
                    for interval in action['frame_intervals']:
                        frame_start = interval['frame_start']
                        frame_end = interval['frame_end']

                        for i in range(frame_start, frame_end + 1):
                            if global_frame_count[action_type] >= frame_limit:
                                break

                            face_frame = i - face_sync
                            hands_frame = i - hands_sync
                            pose_frame = i - pose_sync

                            if (self.check_left_hand(data_hands, hands_frame) or 
                                self.check_right_hand(data_hands, hands_frame)):

                                action_data.append({
                                    'type': action_type,
                                    'frame': i
                                })
                                global_frame_count[action_type] += 1

            # Agregar los frames válidos del subdirectorio actual al DataFrame combinado
            df = pd.DataFrame(action_data)
            combined_balanced_df = pd.concat([combined_balanced_df, df], ignore_index=True)

        # Guardar el DataFrame balanceado a JSON
        balanced_json = combined_balanced_df.to_dict(orient='records')
        output_data = {'actions': balanced_json}

        with open('combined_balanced_frames_dataset.json', 'w') as output_file:
            json.dump(output_data, output_file, indent=4)


    def get_json_files(self, directory_path):
        json_file_groups = []
        for subdir, _, files in os.walk(directory_path):
            json_files = [os.path.join(subdir, f) for f in files if f.endswith('.json')]
            if json_files:
                json_file_groups.append({
                    'subdirectory': subdir,
                    'files': json_files
                })
        return json_file_groups

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python balancer_v2.py <directorio_principal>")
    else:
        directory_path = sys.argv[1]
        balancer = Balancer_v2()
        json_file_groups = balancer.get_json_files(directory_path)

        # Ejecutar el balanceo sobre los archivos 'frames.json' en cada grupo
        balancer.balance_frames(json_file_groups)

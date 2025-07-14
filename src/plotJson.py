"""

Arggument: The directory where the JSON files are located.

"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

class PlotJson:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.files = []
        self.json_path = ""
        self.actions = []
        self.actions_especified = []
        self.actions_by_time = []
        self.actions_by_frame = {}  
        self.actions_with_all_frames = []
    
    def get_files(self):
        try:
            if os.path.isdir(self.directory_path):
                self.files = [file for file in os.listdir(self.directory_path)
                                 if os.path.isfile(os.path.join(self.directory_path, file))]
            else:
                print(f"{self.directory_path} no es un directorio válido.")
        except Exception as e:
            print(f"Error al acceder al directorio: {e}")

    def load_actions(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.json_path}. Trying alternative encoding...")
            with open(self.json_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        
        if "openlabel" in data:

            for frame_id, frame_data in data["openlabel"]["actions"].items():
                if "type" in frame_data:
                    for frame_interval in frame_data["frame_intervals"]:
                        frame_start = frame_interval["frame_start"]
                        frame_end = frame_interval["frame_end"]

                        for frame in range(frame_start, frame_end + 1):
                            if frame not in self.actions_by_frame:
                                self.actions_by_frame[frame] = []
                            if frame_data["type"] not in self.actions_by_frame[frame]:
                                self.actions_by_frame[frame].append(frame_data["type"])

                        frame_inter = frame_end - frame_start

                        if not any(frame_data["type"] == action[0] for action in self.actions):
                            self.actions.append([frame_data["type"], 1, frame_inter, frame_inter, frame_inter])
                            self.actions_with_all_frames.append([frame_data["type"], frame_inter])
                        else:
                            for action in self.actions:
                                if action[0] == frame_data["type"]:
                                    action[1] += 1
                                    if frame_inter < action[2]:
                                        action[2] = frame_inter
                                    if frame_inter > action[3]:
                                        action[3] = frame_inter
                                    action[4] += frame_end - frame_start
                                    break
                            for action in self.actions_with_all_frames:
                                if action[0] == frame_data["type"]:
                                    action.append(frame_inter)
                        action_type = frame_data["type"].split('/')[0]

                        if not any(action_type == action[0] for action in self.actions_especified):
                            self.actions_especified.append([action_type, 1])
                        else:
                            for action in self.actions_especified:
                                if action[0] == action_type:
                                    action[1] += 1
                                    break

        else:
            for frame_id, frame_data in data["vcd"]["actions"].items():
                if "type" in frame_data:
                    for frame_interval in frame_data["frame_intervals"]:
                        frame_start = frame_interval["frame_start"]
                        frame_end = frame_interval["frame_end"]

                        for frame in range(frame_start, frame_end + 1):
                            if frame not in self.actions_by_frame:
                                self.actions_by_frame[frame] = []
                            if frame_data["type"] not in self.actions_by_frame[frame]:
                                self.actions_by_frame[frame].append(frame_data["type"])

                        frame_inter = frame_end - frame_start

                        if not any(frame_data["type"] == action[0] for action in self.actions):
                            self.actions.append([frame_data["type"], 1, frame_inter, frame_inter, frame_inter])
                            self.actions_with_all_frames.append([frame_data["type"], frame_inter])
                        else:
                            for action in self.actions:
                                if action[0] == frame_data["type"]:
                                    action[1] += 1
                                    if frame_inter < action[2]:
                                        action[2] = frame_inter
                                    if frame_inter > action[3]:
                                        action[3] = frame_inter
                                    action[4] += frame_end - frame_start
                                    break
                            for action in self.actions_with_all_frames:
                                if action[0] == frame_data["type"]:
                                    action.append(frame_inter)

                        action_type = frame_data["type"].split('/')[0]

                        if not any(action_type == action[0] for action in self.actions_especified):
                            self.actions_especified.append([action_type, 1])
                        else:
                            for action in self.actions_especified:
                                if action[0] == action_type:
                                    action[1] += 1
                                    break

    def load_actions_from_json(self):

        self.actions = []
        self.actions_especified = []

        for file in self.files:
            if file.endswith(".json"):
                self.json_path = os.path.join(self.directory_path, file)
                self.load_actions()

    def plot_result(self):
        categories = [item[0] for item in self.actions]
        values_total = [item[1] for item in self.actions]
        frame_counts = [item[4] for item in self.actions]  
        min_frames = [item[2] for item in self.actions]    
        max_frames = [item[3] for item in self.actions]    
        avg_frames = [item[4] / item[1] for item in self.actions]  

        fig, ax1 = plt.subplots(figsize=(10, 6))

        x = np.arange(len(categories))
        bar_width = 0.4 

        ax1.bar(x - bar_width/2, values_total, bar_width, color='skyblue', label='Total Actions')
        ax1.set_xlabel('Actions')
        ax1.set_ylabel('Count', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=90) 

        ax2 = ax1.twinx()
        ax2.bar(x + bar_width/2, frame_counts, bar_width, color='green', alpha=0.6, label='Frames per Action')
        ax2.set_ylabel('Frames', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        plt.title('Total number of actions and frames')
        fig.tight_layout() 

        action_corrected = []

        for i in range(len(self.actions_with_all_frames)):
            action_corrected.append([])

            for j in range(1, len(self.actions_with_all_frames[i])):  
                action_corrected[i].append(self.actions_with_all_frames[i][j])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(action_corrected, labels=categories, patch_artist=True, showmeans=True, meanline=True, showfliers=False)
        
        ax.set_xticklabels(categories, rotation=90)

        ax.set_xlabel('Actions')
        ax.set_ylabel('Nº de frames')

        plt.title('Boxplot')
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Falta el directorio.")
    else:
        route = sys.argv[1]
        manager = PlotJson(route)
        manager.get_files()
        manager.load_actions_from_json()
        manager.plot_result()
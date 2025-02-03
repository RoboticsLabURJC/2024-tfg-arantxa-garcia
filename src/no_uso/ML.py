"""

En desuso

Usé el script para ver que convertía bien los datos de los json a un dataframe de pandas.

"""


import sys
import json
import numpy as np
import pandas as pd
import time
import cv2

class ML:
    def __init__(self, json_both, json_onlyleft, json_onlyright, json_radio, json_drinking, json_reachside, json_phonecallright):
        self.both_j = json_both
        self.onlyleft_j = json_onlyleft
        self.onlyright_j = json_onlyright
        self.radio_j = json_radio
        self.drinking_j = json_drinking
        self.reachside_j = json_reachside
        self.phonecallright_j = json_phonecallright

        self.both = []
        self.onlyleft = []
        self.onlyright = []
        self.radio = []
        self.drinking = []
        self.reachside = []
        self.phonecallright = []

        self.rows = []

    def open_jsons(self):

        try:
            with open(self.both_j, 'r', encoding='utf-8-sig') as f:
                self.both = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.both_j}. Trying alternative encoding...")
            with open(self.both_j, 'r', encoding='latin-1') as f:
                self.both = json.load(f)

        try:
            with open(self.onlyleft_j, 'r', encoding='utf-8-sig') as f:
                self.onlyleft = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.onlyleft_j}. Trying alternative encoding...")
            with open(self.onlyleft_j, 'r', encoding='latin-1') as f:
                self.onlyleft = json.load(f)

        try:
            with open(self.onlyright_j, 'r', encoding='utf-8-sig') as f:
                self.onlyright = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.onlyright_j}. Trying alternative encoding...")
            with open(self.onlyright_j, 'r', encoding='latin-1') as f:
                self.onlyright = json.load(f)

        try:
            with open(self.radio_j, 'r', encoding='utf-8-sig') as f:
                self.radio = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.radio_j}. Trying alternative encoding...")
            with open(self.radio_j, 'r', encoding='latin-1') as f:
                self.radio = json.load(f)

        try:
            with open(self.drinking_j, 'r', encoding='utf-8-sig') as f:
                self.drinking = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.drinking_j}. Trying alternative encoding...")
            with open(self.drinking_j, 'r', encoding='latin-1') as f:
                self.drinking = json.load(f)

        try:
            with open(self.reachside_j, 'r', encoding='utf-8-sig') as f:
                self.reachside = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.reachside_j}. Trying alternative encoding...")
            with open(self.reachside_j, 'r', encoding='latin-1') as f:
                self.reachside = json.load(f)

        try:
            with open(self.phonecallright_j, 'r', encoding='utf-8-sig') as f:
                self.phonecallright = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.phonecallright_j}. Trying alternative encoding...")
            with open(self.phonecallright_j, 'r', encoding='latin-1') as f:
                self.phonecallright = json.load(f)

    def prepare_data(self, data):

        for item in data:

            # print(item['pose']['pose'][50][0])
            # time.sleep(100)

            features = {
                "center_left_x": item['pose']['pose'][50][0],
                "center_left_y": item['pose']['pose'][50][1],
                "center_right_x": item['pose']['pose'][51][0],
                "center_right_y": item['pose']['pose'][51][1],
                "pose_0_x": item['pose']['pose'][0][0],
                "pose_0_y": item['pose']['pose'][0][1],
                "pose_1_x": item['pose']['pose'][1][0],
                "pose_1_y": item['pose']['pose'][1][1],
                "pose_2_x": item['pose']['pose'][2][0],
                "pose_2_y": item['pose']['pose'][2][1],
                "pose_3_x": item['pose']['pose'][3][0],
                "pose_3_y": item['pose']['pose'][3][1],
                "pose_4_x": item['pose']['pose'][4][0],
                "pose_4_y": item['pose']['pose'][4][1],
                "pose_5_x": item['pose']['pose'][5][0],
                "pose_5_y": item['pose']['pose'][5][1],
                "type": item['type']
            }

            # # Aplanar 'hands', 'face', y 'pose' si contienen más datos relevantes
            # hands_data = item.get("hands", {})
            # features.update(hands_data) 
            
            self.rows.append(features)

        # df = pd.DataFrame(rows)
        # print(df.head())


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python3 ML.py json_both json_onlyleft json_onlyright json_radio json_drinking json_reachside json_phonecallright")
        sys.exit(1)
        
    ML_performer = ML(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    ML_performer.open_jsons()
    ML_performer.prepare_data(ML_performer.both)
    ML_performer.prepare_data(ML_performer.onlyleft)
    ML_performer.prepare_data(ML_performer.onlyright)
    ML_performer.prepare_data(ML_performer.radio)
    ML_performer.prepare_data(ML_performer.drinking)
    ML_performer.prepare_data(ML_performer.reachside)
    ML_performer.prepare_data(ML_performer.phonecallright)

    dataset = pd.DataFrame(ML_performer.rows)
    print(dataset)



    
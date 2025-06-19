import cv2
import sys
import numpy as np
import json
import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter

hist_features_1 = []
hist_features_2 = []

def dist_btw_hands(hand1):
    return hand1[0]

def prepare_data(item, hist_features):
    dist_features = dist_btw_hands(item['pose']['pose'][51])
    hist_features.append(dist_features)

def paint_hist(hist1, hist2, title, label1, label2):
    bins = 10
    min_val = min(min(hist1, default=0), min(hist2, default=0))
    max_val = max(max(hist1, default=1), max(hist2, default=1))
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    plt.hist(hist1, bins=bin_edges, color='blue', edgecolor='black', alpha=0.6, label=label1)
    plt.hist(hist2, bins=bin_edges, color='red', edgecolor='black', alpha=0.6, label=label2)
    plt.xlabel("Distancia")
    plt.ylabel("Repeticiones")
    plt.title(title)
    plt.legend()
    plt.xlim(0.6, 1)
    plt.show()

class BalanceReconstructor:
    def __init__(self, json_file, hist_features):
        self.data = []
        self.json_file = json_file
        self.hist_features = hist_features
    
    def open_json(self):
        print(f"Opening JSON file {self.json_file}")
        try:
            with open(self.json_file, 'r', encoding='utf-8-sig') as f:
                self.data = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.json_file}. Trying alternative encoding...")
            with open(self.json_file, 'r', encoding='latin-1') as f:
                self.data = json.load(f)
    
    def reconstruct(self):
        self.open_json()
        for action in self.data:
            prepare_data(action, self.hist_features)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python reconstructor_balance.py <json_file_1> <json_file_2>")
    else:
        json_file_1 = sys.argv[1]
        json_file_2 = sys.argv[2]
        
        br1 = BalanceReconstructor(json_file_1, hist_features_1)
        br2 = BalanceReconstructor(json_file_2, hist_features_2)
        
        br1.reconstruct()
        br2.reconstruct()
        
        paint_hist(hist_features_1, hist_features_2, "Comparación posicion mano derecha en X", json_file_1, json_file_2)

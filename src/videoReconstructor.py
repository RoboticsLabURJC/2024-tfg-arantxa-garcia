class videoReconstructor:

    def __init__(self, json_1, json_2, json_3, json_4, video_1, video_2, video_3):
        self.files = [json_1, json_2, json_3, json_4]
        self.video_paths = [video_1, video_2, video_3]
        self.actions = self.load_actions_from_json()

        self.data_hands = []
        self.data_face = []
        self.data_pose = []

        self.counter_goods = 0
        self.counter_total = 0

        self.HANDS_CONNECTION = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (13, 17), (17, 18), (18, 19), (19, 20)
        ]

        self.FACE_CONNECTION = [
            (17, 291), (17, 61), (0, 61), (0, 291),
            (61, 4), (4, 291), (4, 48), (4, 278),
            (291, 426), (61, 206), (61, 50), (291, 280),
            (206, 48), (426, 278), (48, 50), (278, 280),
            (4, 107), (4, 336), (50, 145), (280, 374),
            (122, 107), (122, 145), (351, 336), (351, 374),
            (145, 130), (145, 133), (374, 359), (374, 362),
            (130, 159), (130, 46), (359, 386), (359, 276),
            (133, 159), (362, 386), (46, 105), (276, 334),
            (105, 107), (334, 336)
        ]

        self.POSE_CONNECTION = [
            (12, 24), (12, 11), (11, 23), (24, 23),
            (12, 14), (14, 16), (11, 13), (13, 15),
        ]

    def prepare_prediction(self, data):
        # print("entra prepare")

        right_elbow = data['pose'][1]
        # print(data)

        features_action = {
            "center_left_x": tf(data['pose'][50], right_elbow, 0),
            "center_left_y": tf(data['pose'][50], right_elbow, 1),
            "center_right_x": tf(data['pose'][51], right_elbow, 0),
            "center_right_y": tf(data['pose'][51], right_elbow, 1),
            "pose_0_x": tf(data['pose'][0], right_elbow, 0),
            "pose_0_y": tf(data['pose'][0], right_elbow, 1),
            "pose_1_x": tf(data['pose'][1], right_elbow, 0),
            "pose_1_y": tf(data['pose'][1], right_elbow, 1),
            "pose_2_x": tf(data['pose'][2], right_elbow, 0),
            "pose_2_y": tf(data['pose'][2], right_elbow, 1),
            "pose_3_x": tf(data['pose'][3], right_elbow, 0),
            "pose_3_y": tf(data['pose'][3], right_elbow, 1),
            "pose_4_x": tf(data['pose'][4], right_elbow, 0),
            "pose_4_y": tf(data['pose'][4], right_elbow, 1),
            "pose_5_x": tf(data['pose'][5], right_elbow, 0),
            "pose_5_y": tf(data['pose'][5], right_elbow, 1),
            "elbow_right": calc_ang(data['pose'][1], data['pose'][5], data['pose'][3]),
            "elbow_left": calc_ang(data['pose'][0], data['pose'][4], data['pose'][2]),
            "wrist_right": calc_ang(data['pose'][3], data['pose'][51], data['pose'][5]),
            "wrist_left": calc_ang(data['pose'][2], data['pose'][50], data['pose'][4]),
            "hand_right_x_1": tf(data['pose'][29], right_elbow, 0),
            "hand_right_y_1": tf(data['pose'][29], right_elbow, 1),
            "hand_right_x_2": tf(data['pose'][30], right_elbow, 0),
            "hand_right_y_2": tf(data['pose'][30], right_elbow, 1),
            "hand_right_x_3": tf(data['pose'][31], right_elbow, 0),
            "hand_right_y_3": tf(data['pose'][31], right_elbow, 1),
            "hand_right_x_4": tf(data['pose'][32], right_elbow, 0),
            "hand_right_y_4": tf(data['pose'][32], right_elbow, 1),
            "hand_right_x_5": tf(data['pose'][33], right_elbow, 0),
            "hand_right_y_5": tf(data['pose'][33], right_elbow, 1),
            "hand_right_x_6": tf(data['pose'][34], right_elbow, 0),
            "hand_right_y_6": tf(data['pose'][34], right_elbow, 1),
            "hand_right_x_7": tf(data['pose'][35], right_elbow, 0),
            "hand_right_y_7": tf(data['pose'][35], right_elbow, 1),
            "hand_right_x_8": tf(data['pose'][36], right_elbow, 0),
            "hand_right_y_8": tf(data['pose'][36], right_elbow, 1),
            "hand_right_x_9": tf(data['pose'][37], right_elbow, 0),
            "hand_right_y_9": tf(data['pose'][37], right_elbow, 1),
            "hand_right_x_10": tf(data['pose'][38], right_elbow, 0),
            "hand_right_y_10": tf(data['pose'][38], right_elbow, 1),
            "hand_right_x_11": tf(data['pose'][39], right_elbow, 0),
            "hand_right_y_11": tf(data['pose'][39], right_elbow, 1),
            "hand_right_x_12": tf(data['pose'][40], right_elbow, 0),
            "hand_right_y_12": tf(data['pose'][40], right_elbow, 1),
            "hand_right_x_13": tf(data['pose'][41], right_elbow, 0),
            "hand_right_y_13": tf(data['pose'][41], right_elbow, 1),
            "hand_right_x_14": tf(data['pose'][42], right_elbow, 0),
            "hand_right_y_14": tf(data['pose'][42], right_elbow, 1),
            "hand_right_x_15": tf(data['pose'][43], right_elbow, 0),
            "hand_right_y_15": tf(data['pose'][43], right_elbow, 1),
            "hand_right_x_16": tf(data['pose'][44], right_elbow, 0),
            "hand_right_y_16": tf(data['pose'][44], right_elbow, 1),
            "hand_right_x_17": tf(data['pose'][45], right_elbow, 0),
            "hand_right_y_17": tf(data['pose'][45], right_elbow, 1),
            "hand_right_x_18": tf(data['pose'][46], right_elbow, 0),
            "hand_right_y_18": tf(data['pose'][46], right_elbow, 1),
            "hand_right_x_19": tf(data['pose'][47], right_elbow, 0),
            "hand_right_y_19": tf(data['pose'][47], right_elbow, 1),
            "hand_right_x_20": tf(data['pose'][48], right_elbow, 0),
            "hand_right_y_20": tf(data['pose'][48], right_elbow, 1),
            "hand_right_x_21": tf(data['pose'][49], right_elbow, 0),
            "hand_right_y_21": tf(data['pose'][49], right_elbow, 1)
            }

        features_gaze  = {
                "face_0_x": data['face'][0][0],
                "face_0_y": data['face'][0][1],
                "face_1_x": data['face'][1][0],
                "face_1_y": data['face'][1][1],
                "face_2_x": data['face'][2][0],
                "face_2_y": data['face'][2][1],
                "face_3_x": data['face'][3][0],
                "face_3_y": data['face'][3][1],
                "face_4_x": data['face'][4][0],
                "face_4_y": data['face'][4][1],
                "face_5_x": data['face'][5][0],
                "face_5_y": data['face'][5][1],
                "face_6_x": data['face'][6][0],
                "face_6_y": data['face'][6][1],
                "face_7_x": data['face'][7][0],
                "face_7_y": data['face'][7][1],
                "face_8_x": data['face'][8][0],
                "face_8_y": data['face'][8][1],
                "face_9_x": data['face'][9][0],
                "face_9_y": data['face'][9][1],
                "face_10_x": data['face'][10][0],
                "face_10_y": data['face'][10][1],
                "face_11_x": data['face'][11][0],
                "face_11_y": data['face'][11][1],
                "face_12_x": data['face'][12][0],
                "face_12_y": data['face'][12][1],
                "face_13_x": data['face'][13][0],
                "face_13_y": data['face'][13][1],
                "face_14_x": data['face'][14][0],
                "face_14_y": data['face'][14][1],
                "face_15_x": data['face'][15][0],
                "face_15_y": data['face'][15][1],
                "face_16_x": data['face'][16][0],
                "face_16_y": data['face'][16][1],
                "face_17_x": data['face'][17][0],
                "face_17_y": data['face'][17][1],
                "face_18_x": data['face'][18][0],
                "face_18_y": data['face'][18][1],
                "face_19_x": data['face'][19][0],
                "face_19_y": data['face'][19][1],
                "face_20_x": data['face'][20][0],
                "face_20_y": data['face'][20][1],
                "face_21_x": data['face'][21][0],
                "face_21_y": data['face'][21][1],
                "face_22_x": data['face'][22][0],
                "face_22_y": data['face'][22][1],
                "face_23_x": data['face'][23][0],
                "face_23_y": data['face'][23][1],
                "face_24_x": data['face'][24][0],
                "face_24_y": data['face'][24][1],
                "face_25_x": data['face'][25][0],
                "face_25_y": data['face'][25][1],
                "face_26_x": data['face'][26][0],
                "face_26_y": data['face'][26][1],
                "gaze_0_x": data['gaze'][0][0],
                "gaze_0_y": data['gaze'][0][1],
                "gaze_1_x": data['gaze'][1][0],
                "gaze_1_y": data['gaze'][1][1]
            }

        return features_action, features_gaze

    def get_files(self):
        try:
            if os.path.isdir(self.directory_path):
                self.files = [file for file in os.listdir(self.directory_path)
                                 if os.path.isfile(os.path.join(self.directory_path, file))]
            else:
                print(f"{self.directory_path} no es un directorio válido.")
        except Exception as e:
            print(f"Error al acceder al directorio: {e}")

    def load_actions_from_json(self):
        try:
            with open(self.files[0], 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.files[0]}. Trying alternative encoding...")
            with open(self.files[0], 'r', encoding='latin-1') as f:
                data = json.load(f)

        actions = {}

        for frame_id, frame_data in data["openlabel"]["actions"].items():
            if "type" in frame_data:
                for frame_interval in frame_data["frame_intervals"]:
                    frame_start = frame_interval["frame_start"]
                    frame_end = frame_interval["frame_end"]
                    for frame in range(frame_start, frame_end + 1):
                        if frame not in actions:
                            actions[frame] = []
                        if frame_data["type"] not in actions[frame]:
                            actions[frame].append(frame_data["type"])

        for frame_id, frame_data in data["openlabel"]["streams"].items():
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

    def open_jsons(self):
        try:
            with open(self.files[1], 'r', encoding='utf-8-sig') as f:
                self.data_hands = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.files[1]}. Trying alternative encoding...")
            with open(self.files[1], 'r', encoding='latin-1') as f:
                self.data_hands = json.load(f)

        try:
            with open(self.files[2], 'r', encoding='utf-8-sig') as f:
                self.data_pose = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.files[2]}. Trying alternative encoding...")
            with open(self.files[2], 'r', encoding='latin-1') as f:
                self.data_pose = json.load(f)

        try:
            with open(self.files[3], 'r', encoding='utf-8-sig') as f:
                self.data_face = json.load(f)
        except UnicodeDecodeError:
            print(f"Failed to decode JSON file {self.files[3]}. Trying alternative encoding...")
            with open(self.files[3], 'r', encoding='latin-1') as f:
                self.data_face = json.load(f)

    def reconstruct(self, video_paths):
        # print("entra a aqui")
        time.sleep(0.5)
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
        video_started = [False, False, False]  # Flags to track when each video starts

        init_time = time.time()
        total_time = 0
        last_prediction = None
        consecutive_count = 0
        previous_prediction_s = ""

        needed_consecutive = 15
        now_pred = []
        consecutive_actions = {}
        missing_actions = {}

        # print("Iniciando reconstrucción...")
        # time.sleep(0.5)

        while all([cap.isOpened() for cap in caps]):

            # print("entra al while principio")

            prediction_s = ""
            Y_pred_prob_percent = []

            frames = []
            cap_number = 0

            # print("entra al while")
            # time.sleep(0.5)
            for cap in caps:
                # print("entra al for")
                # time.sleep(0.5)
                # If the video has already started, read it normally
                if video_started[cap_number]:
                    success, frame = cap.read()
                    if not success:
                        break
                else:
                    # If it has not started, it pauses in black until the synchronization is complete
                    frame = np.zeros((height, width, 3), dtype=np.uint8)

                black_frame = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

                # print("cap_number: ", cap_number)

                if cap_number == 0:  # First video (pose)
                    # print("entra a pose")
                    # time.sleep(0.5)
                    if frame_number >= self.pose_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True  # Marks the video as started

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        features = self.prepare_prediction(self.data_pose["iterations"][frame_number - self.pose_sync])
                        features_df = pd.DataFrame([[features[name] for name in features.keys()]], columns=features.keys())

                        feature_names = list(features_df.columns)

                        features_df = pd.DataFrame([[features[name] for name in feature_names]], columns=feature_names)

                        prediction = model.predict(features_df)
                        Y_pred_prob = model.predict_proba(features_df)
                        Y_pred_prob_percent = (Y_pred_prob * 100).round(2) 
                        np.set_printoptions(suppress=True, precision=2)

                        now_pred = []
                        prediction = prediction[0]


                        for idx, predi in enumerate(prediction):
                            if idx == 0 and predi == 1:
                                now_pred.append("hands_using_wheel/both")
                            elif idx == 1 and predi == 1:
                                now_pred.append("hands_using_wheel/only_left")
                            elif idx == 2 and predi == 1:
                                now_pred.append("hands_using_wheel/only_right")
                            elif idx == 3 and predi == 1:
                                now_pred.append("driver_actions/radio")
                            elif idx == 4 and predi == 1:
                                now_pred.append("driver_actions/drinking")
                            elif idx == 5 and predi == 1:
                                now_pred.append("driver_actions/reach_side")

                        new_actions = []  

                        print(now_pred)

                        for action in now_pred:
                            if action in consecutive_actions:
                                consecutive_actions[action] += 1
                            else:
                                consecutive_actions[action] = 1
                            if consecutive_actions[action] >= needed_consecutive:
                                print(consecutive_actions[action])
                                new_actions.append(action)

                        print(new_actions)

                        for action in list(consecutive_actions.keys()):
                            if action not in now_pred:
                                if action in missing_actions:
                                    missing_actions[action] += 1
                                else:
                                    if(consecutive_actions[action] < needed_consecutive):
                                        del consecutive_actions[action] 
                                        break
                                    else:
                                        missing_actions[action] = 1  

                                if missing_actions[action] >= needed_consecutive:
                                    del consecutive_actions[action] 
                                    del missing_actions[action] 
                                elif(missing_actions[action] < needed_consecutive and consecutive_actions[action] >= needed_consecutive):
                                    new_actions.append(action)
                            else:
                                if action in missing_actions:
                                    del missing_actions[action]  

                        prediction_s = new_actions  
                        print(prediction_s)

                        self.paint_frame(frame, frame_number - self.pose_sync, "pose")
                        frame = cv2.resize(frame, (reduced_width, reduced_height))
                    else:
                        # print("entra al else")
                        frame = black_frame  # Black till synchronization of pose_sync

                elif cap_number == 1:  # Second video (hands)
                    if frame_number >= self.hands_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # print("llamando a paint_frame")
                        self.paint_frame(frame, frame_number - self.hands_sync , "hands")
                        frame = cv2.resize(frame, (reduced_width, reduced_height))
                    else:
                        frame = black_frame  # Black till synchronization of hands_sync

                elif cap_number == 2:  # Third video (face)
                    if frame_number >= self.face_sync:
                        if not video_started[cap_number]:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            video_started[cap_number] = True

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.paint_frame(frame, frame_number - self.face_sync, "face")
                        frame = cv2.resize(frame, (reduced_width, reduced_height))
                    else:
                        frame = black_frame  # Black till synchronization of face_sync

                frames.append(frame)
                cap_number += 1

            if len(frames) != len(caps):
                break

            combined_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

            # Each frame in its quadrant
            combined_frame[0:reduced_height, 0:reduced_width] = frames[0]  # First video
            if len(frames) > 1:
                combined_frame[0:reduced_height, reduced_width:reduced_width*2] = frames[1]  # Second video
            if len(frames) > 2:
                combined_frame[reduced_height:reduced_height*2, 0:reduced_width] = frames[2]  # Third video

            # Blank space for actions
            if frame_number in self.actions:
                actions = self.actions[frame_number]
                # actions = action['type'].split()

                for pred_act in prediction_s:
                    if pred_act in actions:
                        # counter_goods += 1
                        self.counter_goods += 1
                        # self.counter_total += 1
                    # else:
                    self.counter_total += 1

                # if prediction_s == "":
                #     self.counter_total += 1

                if len(prediction_s) == 0:
                    self.counter_total += 1

                valid_actions = [act for act in actions if act.startswith("driver_actions") or act.startswith("hands_using_wheel")]
                # valid_actions = [act for act in actions if act.startswith("hands_using_wheel")]


                y_offset = 30
                start_y = height // 2 + y_offset 
                line_height = 20  
                x_offset = width // 2 + 10 

                for i, valid_action in enumerate(valid_actions):
                    cv2.putText(
                        combined_frame,
                        valid_action,
                        (x_offset, start_y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

                prediction_start_y = start_y + len(valid_actions) * line_height 
                
                # if prediction_s == "hands_using_wheel/both ":
                #     prediction_s = "driver_actions/safe_drive"
                
                text = " ".join(prediction_s)  # Une las palabras con un espacio
                cv2.putText(
                    combined_frame,
                    text,
                    (x_offset, prediction_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )


                if len(Y_pred_prob_percent) > 0:
                    probabilities_start_y = prediction_start_y + line_height
                    cv2.putText(
                        combined_frame,
                        "Probabilidades:",
                        (x_offset, probabilities_start_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                    
                    probability_classes = ["Both", "Left", "Right", "Radio", "Drinking", "Reachside"]
                    for j, class_name in enumerate(probability_classes):
                        prob_text = f"{class_name}: {Y_pred_prob_percent[0][j]}%"
                        cv2.putText(
                            combined_frame,
                            prob_text,
                            (x_offset, probabilities_start_y + (j + 1) * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                percent_correct = (self.counter_goods / self.counter_total) * 100
                bottom_y = combined_frame.shape[0] - 10  # Descuento de margen inferior

                cv2.putText(
                    combined_frame,
                    f"Porcentaje de aciertos: {percent_correct:.2f}%",
                    (x_offset, bottom_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                frame_time = time.time() - init_time
                total_time += frame_time
                init_time = time.time()

                print(f"Frame {frame_number} - Tiempo de procesamiento: {frame_time:.2f} s")
                print("--------------------------------------------------")


            cv2.imshow('Combined Video', combined_frame)
            out.write(combined_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_number += 1
            
        for cap in caps:
            cap.release()
        out.release()
        cv2.destroyAllWindows()

    def paint_frame(self, frame, frame_number, json):
        # print("Frame: ", frame_number)
        if json == "hands":
            data = self.data_hands
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        for x, y, z in iterations["hands"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        for x, y in iterations["centers"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (255, 0, 0), 30)

                        hand_left = iterations["hands"][:21]
                        hand_right = iterations["hands"][21:]

                        self.draw_connections(frame, hand_left, self.HANDS_CONNECTION)
                        self.draw_connections(frame, hand_right, self.HANDS_CONNECTION)

        elif json == "pose":
            data = self.data_pose
            # print("Pose")
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:
                        for x, y, z in iterations["pose"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        pose_data = iterations["pose"][:8]
                        left_hand = iterations["pose"][8:29]
                        right_hand = iterations["pose"][29:50]
                        left_center = iterations["pose"][50]
                        right_center = iterations["pose"][51]

                        cv2.circle(frame, (int(left_center[0] * frame.shape[1]), int(left_center[1] * frame.shape[0])), 5, (255, 0, 0), 30)
                        cv2.circle(frame, (int(right_center[0] * frame.shape[1]), int(right_center[1] * frame.shape[0])), 5, (255, 0, 0), 30)

                        self.draw_connections(frame, pose_data, self.POSE_CONNECTION)
                        self.draw_connections(frame, left_hand, self.HANDS_CONNECTION)
                        self.draw_connections(frame, right_hand, self.HANDS_CONNECTION)

        elif json == "face":
            data = self.data_face
            # print("Face")
            if "iterations" in data:
                for iterations in data["iterations"]:
                    if iterations["frame"] == frame_number:

                        for x, y, indx in iterations["face"]:
                            x = int(x * frame.shape[1])
                            y = int(y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                            # print("Face: ", x, y)
    
                        x, y = iterations["gaze"][0]
                        x2, y2 = iterations["gaze"][1]

                        x, y = int(x), int(y)
                        x2, y2 = int(x2), int(y2)
                        
                        cv2.line(frame, (x, y), (x2, y2), (0, 0, 255), 7)

                        self.draw_connections(frame, iterations["face"], self.FACE_CONNECTION)

    def draw_connections(self, frame, keypoints, connections):


        for idx1, idx2 in connections:

            x1, y1, x2, y2 = None, None, None, None
            for x, y, inx in keypoints:
                if inx == idx1:
                    x1 = int(x * frame.shape[1])
                    y1 = int(y * frame.shape[0])
                elif inx == idx2:
                    x2 = int(x * frame.shape[1])
                    y2 = int(y * frame.shape[0])

            if x1 and y1 and x2 and y2:

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

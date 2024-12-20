import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands#By assigning mp.solutions.hands to mp_hands, the code makes it easy to access MediaPipe's hand tracking functionality, which includes initialization, processing images, and working with hand landmarks.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles#Utilities for visualizing hand landmarks, but not used in this snippet.

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)#Initializes MediaPipe for detecting hand landmarks
#Set to process still images rather than a video stream.

DATA_DIR = './data'# It is assumed that there are subdirectories in ./data.

data = []
labels = []
#These lists will hold the extracted features and corresponding labels from the images.
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#In OpenCV, images are read by default in BGR (Blue-Green-Red) format, while most other libraries and frameworks, including MediaPipe, expect images to be in RGB (Red-Green-Blue) format.

        results = hands.process(img_rgb)#Processes the image using MediaPipe to detect hand landmarks.
        if results.multi_hand_landmarks:#Checks if any hand landmarks are detected.
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)
                    #For each detected hand landmark, the code extracts the x and y coordinates.
                    #x_ and y_ lists store all x and y coordinates of the hand landmarks.

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                    #Normalization: The coordinates are normalized by subtracting the minimum x and y values (to standardize the position and scale of the hand in the image). This makes the landmarks relative to the smallest coordinate, effectively positioning them in a consistent reference frame.

            data.append(data_aux)#normalized landmark data
            labels.append(dir_)#The corresponding class/label (directory name) is appended to the labels list.

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

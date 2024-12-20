import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
import pickle
import pyttsx3
from send_msg import sendtoTelegram
from time import sleep

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'hi', 1: 'Good morning', 2:'good luck', 3:'victory', 4:'thank you',5:'see you'}

result=""
flag=""

def sendmessage():

    global result

    sendtoTelegram(result)


def monitor(event):
    global flag
    flag = event.keysym
    
    print(f"flag = {flag},{type(flag)}")


# Placeholder function for gesture recognition using a trained model
def predict_gesture(frame):
    # Preprocess the frame (e.g., resize, normalize) before passing it to the model
    # Here, you would replace this placeholder code with the actual model inference
    # For demonstration purposes, this function currently returns a random gesture label
    
    # cv2.imshow("Demo",frame)
    # cv2.waitKey()
    try: 
        
        data_aux = []
        x_ = []
        y_ = []

        # ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
 
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            
            

            predicted_character = labels_dict[int(prediction[0])]
            
            # print(f"predicted char is : {predicted_character}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            return predicted_character
    except Exception as e :
        pass 


def text_to_speech(rate=100):
    # Initialize the engine
    engine = pyttsx3.init()

    engine.setProperty('rate', rate)
    
    # Speak the text
    engine.say(result)
    
    # Run and wait for the speech to finish
    engine.runAndWait()


def check_gesture():
    ret, frame = cap.read()
    global result,flag
    
    print(f"value of flag in check _gesture : {flag}")
    if ret:
        # Perform gesture recognition prediction on the frame

 
            
        
        if True:
            res = predict_gesture(frame)
            
            print(f"{res},,,{type(res)}")
            # result_label.config(text="Predicted Gesture: " + str(predicted_gesture))
            # result=result+str(predict_gesture)
            
            flag=None
            
            if res is not None:
                result=result+str(res)+" "
                strt1.set(result)

                sleep(2)
            

    # Update the video feed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=image)
    video_feed.imgtk = imgtk
    video_feed.config(image=imgtk)
    video_feed.after(500, check_gesture)  # Update every 10 milliseconds

# Create the main application window
root = tk.Tk()
root.title("Gesture Recognition")

root.geometry('700x700')
# Create a button to initiate gesture recognition
gesture_button = tk.Button(root, text="Check Gesture", command=check_gesture)
gesture_button.pack(pady=10)

message_button = tk.Button(root, text="Send Message", command=sendmessage)
message_button.pack(pady=10)

speech_button = tk.Button(root, text="text to speech", command=text_to_speech)
speech_button.pack(pady=10)


# Label to display the result of gesture recognition
strt1=tk.StringVar()
result_label = tk.Label(root, textvariable=strt1)
result_label.pack(pady=10)

# Create a label for displaying the webcam feed
video_feed = tk.Label(root)
video_feed.pack()

root.bind("<Key>", monitor)
# Open the webcam
cap = cv2.VideoCapture(0)

# Run the Tkinter event loop
root.mainloop()

# Release the webcam
cap.release()

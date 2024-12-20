import os

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)#block checks if the ./data directory exists. If it doesn't, it creates the directory. This ensures that the directory where images will be saved is available.

number_of_classes = 6
dataset_size = 100

cap = cv2.VideoCapture(0)#Opens the webcam feed (camera ID 0). This object, cap, is used to capture frames from the webcam.
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))#The script checks if a folder exists for the class 

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        #The loop breaks when the user presses the 'Q' key (cv2.waitKey(25) == ord('q')), signaling that they are ready to start capturing images for the current class.

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()

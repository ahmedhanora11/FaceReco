import sys
import face_recognition
import os
import cv2
import numpy as np
import math
import dlib
from gtts import gTTS
import time


def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return f"{round(linear_val * 100, 2)}%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return f"{round(value, 2)}%"

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.encode_faces()
        self.name_announced = False  # Initialize the flag to False
        self.last_announcement_time = 0  # Initialize the last announcement time to 0


    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(os.path.join('faces', image))
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('No camera is found!')



        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]  # to get it as RGB

                # Find face locations
                face_locations = face_recognition.face_locations(rgb_small_frame)

                # Convert face locations from small frame to original frame coordinates
                face_locations_original = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]

                # Compute face encodings
                face_encodings = face_recognition.face_encodings(frame, face_locations_original)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                        current_time = time.time()
                        if not self.name_announced or current_time - self.last_announcement_time >= 15:
                            tts = gTTS(text=f"Detected face: {name.split('.')[0]}", lang='en')
                            tts.save('temp.mp3')
                            os.system('start temp.mp3')
                            self.name_announced = True
                            self.last_announcement_time = current_time

                    face_names.append(f'{name} ({confidence})')

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Expand the rectangle around the face by a certain margin
                    margin = 32
                    top -= margin
                    right += margin
                    bottom += margin
                    left -= margin
                    
                    title = f"{name.split('.')[0]}"
                    confidence = f"Confidence: {name.split('(')[1].split(')')[0]}"

                    # Draw a colored rectangle around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Draw a filled rectangle for the name text
                    cv2.rectangle(frame, (left, bottom - 45), (right, bottom), (0, 255, 0), cv2.FILLED)

                    # Choose text colors for name and confidence
                    text_color = (0, 0, 0)  # White text color
                    cv2.putText(frame, title, (left + 6, bottom - 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)

                    # Draw a filled rectangle for the confidence text
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, confidence, (left + 6, bottom - 16), cv2.FONT_HERSHEY_COMPLEX, 0.6, text_color, 1)

                cv2.imshow('Face Reco', frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
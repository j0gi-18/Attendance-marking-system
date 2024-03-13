import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from sklearn.ensemble import RandomForestClassifier
import joblib
import datetime as dt
import pandas as pd

# Create MTCNN object with higher threshold
mtcnn = MTCNN(thresholds=[0.9, 0.9, 0.9])

facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# import the model
clf = joblib.load('model.joblib')

# create a dictionary to map names with key values
label_to_name = {0: 'Akash Khulpe', 1: 'Anuj Gavhane', 2:'Devang Edle', 3:'Deepanshu Gadling', 4:'Gaurav Diwedi', 5:'Parimal Kumar', 6:'Parth Deshpande',7:'Rutuja Doiphode',8:'Sagar Kumbhar'}

# Create empty lists to store student names, date and entry time
attendance_records  = []

# Open webcam use other index number if using multiple camera setup
cap = cv2.VideoCapture(0)

# adjust threshold value as required
threshold = 0.26

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    boxes, _ = mtcnn.detect(frame)

    # records date and time
    current_time = dt.datetime.now().strftime('%I: %M: %S %p')

    # If faces are detected, accumulate them for batch processing
    if boxes is not None:
        faces = []
        face_boxes = []

        # Iterate over the detected faces
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                # Resize the face image to desired dimensions
                face = cv2.resize(face, (160, 160))

                # Convert face image to torch tensor
                face = torch.from_numpy(face).permute(2, 0, 1).float()

                # Normalize the face image
                face = (face - 127.5) / 128.0

                faces.append(face)
                face_boxes.append((x1, y1, x2, y2))

        if len(faces) > 0:
            # Convert faces list to a batch tensor
            faces = torch.stack(faces)

            # Generate embeddings for all faces in the batch
            with torch.no_grad():
                embeddings = facenet_model(faces)

            # Perform predictions on the embeddings
            predictions = clf.predict(embeddings.numpy())
            probabilities = clf.predict_proba(embeddings.numpy())

            # Iterate over the predictions and draw bounding boxes
            for prediction, probability, box in zip(predictions, probabilities, face_boxes):
                if probability.max() >= threshold:
                    predicted_name = label_to_name[int(prediction)]
                    #print("Predicted name:", predicted_name)
                    current_date = dt.datetime.now().strftime('%d-%m-%y')
                    current_confidence = probability.max()

                    if not any(record['Name'] == predicted_name for record in attendance_records):
                        # Append attendance record to the list
                        attendance_records.append({'Name': predicted_name, 'Date': current_date, 'Time': current_time})

                    print(probabilities)

                    cv2.putText(frame, predicted_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'unknown person', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # Print all predicted names together
                predicted_names = [label_to_name[int(prediction)] if probability.max() >= threshold else 'unknown person' for prediction, probability in zip(predictions, probabilities)]
                print("Predicted names:", predicted_names)
                attendance_df = pd.DataFrame(attendance_records)

                # Save the DataFrame to a CSV file
                attendance_df.to_csv('attendance_records.csv', index=False)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

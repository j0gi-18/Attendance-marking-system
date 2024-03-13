from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib

# Load embeddings of people
# use facenet to generate embeddings of people and use those for training the classifier
person1_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Akash Khulpe_embeddings.npy')
person2_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Anuj Gavhane_embeddings.npy')
person3_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Devang Edle_embeddings.npy')
person4_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Dipanshu Gadling_embeddings.npy')
person5_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Gaurav Dwivedi_embeddings.npy')
person6_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Parimal Kumar_embeddings.npy')
person7_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Parth Deshpande_embeddings.npy')
person8_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Rutuja Doiphode_embeddings.npy')
person9_embeddings = np.load('/Users/sagarkumbhar/Documents/ml projects/augmented embeddings/Sagar Kumbhar_embeddings.npy')

# Concatenate embeddings of person1 to person9
X = np.concatenate((person1_embeddings, person2_embeddings, person3_embeddings, person4_embeddings, person5_embeddings, person6_embeddings, person7_embeddings, person8_embeddings, person9_embeddings,), axis=0)

# Create labels for the embeddings
y = np.concatenate(( np.zeros(len(person1_embeddings)), np.ones(len(person2_embeddings)), np.full(len(person3_embeddings), 2), np.full(len(person4_embeddings), 3), np.full(len(person5_embeddings), 4), np.full(len(person6_embeddings), 5), np.full(len(person7_embeddings), 6), np.full(len(person8_embeddings), 7), np.full(len(person9_embeddings), 8)), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Model performance metrics
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("classification report is\n", report)
print("confusion matrix is\n", matrix)

joblib.dump(clf, 'model_randomforest.joblib')

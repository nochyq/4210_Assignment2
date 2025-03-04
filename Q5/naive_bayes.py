#-------------------------------------------------------------------------
# AUTHOR: Cynthia Nguyen
# FILENAME: naive_bayes.py
# SPECIFICATION: Using the Naive Bayes strategy, this program reads weather data to train
#                a model in order to predict whether to play tennis based on weather conditions.
#                The program only prints classifications  with a confidence level of 0.75 or more.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db = []
X = []
Y = []

#Reading the training data in a csv file
with open("weather_training.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
feature_dict = {
    "Outlook": {"Sunny": 1, "Overcast": 2, "Rain": 3},
    "Temperature": {"Hot": 1, "Mild": 2, "Cool": 3},
    "Humidity": {"High": 1, "Normal": 2},
    "Wind": {"Weak": 1, "Strong": 2},
    "PlayTennis": {"Yes": 1, "No": 2}
}

for row in db:
    X.append([feature_dict["Outlook"][row[1]], feature_dict["Temperature"][row[2]],
              feature_dict["Humidity"][row[3]], feature_dict["Wind"][row[4]]])
    Y.append(feature_dict["PlayTennis"][row[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

db_test = []

#Reading the test data in a csv file
with open("weather_test.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            db_test.append(row)

#Printing the header os the solution
print(f"{'Day':<7} {'Outlook':<10} {'Temperature':<12} {'Humidity':<10} {'Wind':<7} {'PlayTennis':<12} {'Confidence':<10}")

# Use your test samples to make probabilistic predictions
for row in db_test:
    testSample = [feature_dict["Outlook"][row[1]], feature_dict["Temperature"][row[2]],
                  feature_dict["Humidity"][row[3]], feature_dict["Wind"][row[4]]]
    probabilities = clf.predict_proba([testSample])[0]
    predicted_class = clf.classes_[probabilities.argmax()]
    confidence = max(probabilities)

    if confidence >= 0.75:
        predicted_label = "Yes" if predicted_class == 1 else "No"
        print(f"{row[0]:<7} {row[1]:<10} {row[2]:<12} {row[3]:<10} {row[4]:<7} {predicted_label:<12} {confidence:.2f}")
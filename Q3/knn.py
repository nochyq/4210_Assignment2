#-------------------------------------------------------------------------
# AUTHOR: Cynthia Nguyen
# FILENAME: knn.py
# SPECIFICATION: This program reads the file email_classification.csv and compute the LOO_CV
#                error rate for a 1NN classifier to decide if emails are spam or ham based on
#                word frequency.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0:
         db.append (row)

num_errors = 0
num_instances = len(db)

#Loop your data to allow each instance to be your test set
for i in range(num_instances):

    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages
    # Transform the original training classes to numbers and add them to the vector Y.
    # Do not forget to remove the instance that will be used for testing in this iteration.
    # For instance, Y = [1, 2, ,...].
    # Convert each feature value to float to avoid warning messages
    X = []
    Y = []
    for j in range(num_instances):
        if j != i:
            X.append([float(x) for x in db[j][:-1]])
            Y.append(1 if db[j][-1] == "spam" else 0)

    #Store the test sample of this iteration in the vector testSample
    testSample = [float(x) for x in db[i][:-1]]
    true_label = 1 if db[i][-1] == "spam" else 0

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != true_label:
        num_errors += 1

#Print the error rate
error_rate = num_errors / num_instances
print(f"LOO-CV Error Rate: {error_rate}")
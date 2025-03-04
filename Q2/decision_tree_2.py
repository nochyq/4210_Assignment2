#-------------------------------------------------------------------------
# AUTHOR: Cynthia Nguyen
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program trains decision tree models using different datasets
#                to recommend lens use. The process is repeated 10 times and the average
#                accuracy is calculated for each training set.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

from sklearn.metrics import accuracy_score

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0:
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # Transform the original categorical training classes to numbers and add to the vector Y.
    # For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    age_dict = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
    spectacle_dict = {"Myope": 1, "Hypermetrope": 2}
    astigmatism_dict = {"No": 1, "Yes": 2}
    tear_dict = {"Reduced": 1, "Normal": 2}
    class_dict = {"Yes": 1, "No": 2}

    for row in dbTraining:
        X.append([age_dict[row[0]], spectacle_dict[row[1]], astigmatism_dict[row[2]], tear_dict[row[3]]])
        Y.append(class_dict[row[4]])

    #Loop your training and test tasks 10 times here
    accuracies = []
    for i in range (10):

       #Fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       dbTest = []
       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for i, row in enumerate(reader):
               if i > 0:
                   dbTest.append(row)

       correct_predictions = 0
       total_predictions = len(dbTest)

       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           testSample = [age_dict[data[0]], spectacle_dict[data[1]], astigmatism_dict[data[2]], tear_dict[data[3]]]
           class_predicted = clf.predict([testSample])[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           true_label = class_dict[data[4]]
           if class_predicted == true_label:
               correct_predictions += 1

       # Calculate accuracy for this iteration
       accuracy = correct_predictions / total_predictions
       accuracies.append(accuracy)

    #Find the average of this model during the 10 runs (training and test set)
    avg_accuracy = sum(accuracies) / len(accuracies)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Final accuracy when training on {ds}: {avg_accuracy:.2f}")
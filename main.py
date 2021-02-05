import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
import pandas as pd

np.random.seed(38)

def safelog(x):
    return(np.log(x + 1e-100))
#-------------------------------------------------------------------------------
#--------------------------------Initializing Data------------------------------
def get_data(data_lables):
    data = []
    for i in range(len(data_lables)):
        data.append(data_lables[i][1])
    return data
    
data_labels = np.genfromtxt("hw03_data_set_labels.csv", dtype = str, delimiter = ',')
data_images = np.genfromtxt("hw03_data_set_images.csv", dtype = int, delimiter = ',')

num_of_classes = 5

data_labels = get_data(data_labels)
data_set = np.array([(ord(x) - 65) for x in data_labels])
#-------------------------------------------------------------------------------
#--------------------------------Dividing Data----------------------------------
def get_sets(array):
    train_set, test_set , training_labels, test_labels = [], [], [], []
    count = 0
    for i in range(len(array)):
        if count >= 39:
            count = 0
        if count < 25:
            train_set.append(array[i])
            training_labels.append(data_set[i])
            count += 1
        elif count >= 25 and count < 39:
            test_set.append(array[i])
            test_labels.append(data_set[i])
            count += 1
    return np.array(test_set) ,np.array(train_set),np.array(training_labels), np.array(test_labels)

test_set, training_set, training_labels, test_labels = np.array(get_sets(data_images))

onehot_encoded_lables = np.zeros(shape = (125,5))
onehot_encoded_lables[range(125), training_labels] = 1

onehot_encoded_lables2 = np.zeros(shape = (195, 5))
onehot_encoded_lables2[range(195), data_set] = 1
#-------------------------------------------------------------------------------
#--------------------------------Calculating Priors-----------------------------
def prior(class_data):
    arr = []
    for i in range(len(class_data)):
        arr.append(len(class_data[i])/len(data_set))
    return arr

classes = [data_set[0:39],data_set[39:78],data_set[78:117],data_set[117:156],data_set[156:195]]

priors = prior(classes)
#-------------------------------------------------------------------------------
#--------------------------------Constructing Pij_Estimator Matrix--------------
def r_k():
    arr = []
    for i in range(5):
        arr.append(sum(onehot_encoded_lables[:,i]))
    return arr

pji_matrix = np.dot(training_set.T, onehot_encoded_lables)
rk = r_k()
pji_matrix /= rk
print(pji_matrix[:,4])
#-------------------------------------------------------------------------------
#--------------------------------Prediction Function---------------------------- 
def predict(x, P_matrix, prior):
    return np.dot(x, safelog(P_matrix)) + np.dot((1 - x), safelog(1 - P_matrix)) + safelog(prior)

def y_predicted(X, P_matrix, prior):
    mat = []
    for i in range (len(X)):
        mat.append(predict(X[i], P_matrix, prior))
    return np.array(mat).reshape(len(X),5)
#-------------------------------------------------------------------------------
#--------------------------------Confusion Matrices----------------------------- 
y_pred = y_predicted(training_set, pji_matrix, priors[0]) # priors[0] since all priors are the same
max = np.argmax(y_pred, axis = 1)
confusion_matrix = pd.crosstab(max, training_labels, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)

y_pred = y_predicted(test_set, pji_matrix, priors[0])
max = np.argmax(y_pred, axis = 1)
confusion_matrix = pd.crosstab(max, test_labels, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)
#-------------------------------------------------------------------------------
#--------------------------------Plotting Letters-------------------------------
fig, letters = plt.subplots(1, 5)
pji_matrix = 1 - pji_matrix # Inverted the pji_matrix because the colors are inverted when printing
for i in range(5):
    letters[i].imshow(pji_matrix[:,i].reshape(16, 20).T, cmap = "gray")
plt.show()





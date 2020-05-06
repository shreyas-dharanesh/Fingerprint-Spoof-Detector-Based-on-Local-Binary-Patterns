from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="path to the training images")
ap.add_argument("-e", "--testing", required=True,  help="path to the tesitng images")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

data2 = []
labels2 = []

pridicted_label_set=[]

# loop over the training images
for imagePath in paths.list_images(args["training"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
    
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    
    # extract the label from the image path, then update the
    # original label and data lists
    labels2.append(imagePath.split(os.path.sep)[-2])
    data2.append(hist)
    
    prediction = model.predict(hist.reshape(1, -1))
    
    # store predicted labels in list
    pridicted_label_set.append(prediction[0])

    # display the image and the prediction
    #image=cv2.resize(image, (960, 540))
    #cv2.putText(image, prediction[0]+",", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    #cv2.putText(image, imagePath.split(os.path.sep)[-2]+imagePath.split(os.path.sep)[-1], (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)

# comparing predicted results(pridicted_label_set) and original results(labels2), and generating confusion matrix
con_matrix=confusion_matrix(labels2,pridicted_label_set,labels=["Live","Fake"])
TP=con_matrix[0][0]
FN=con_matrix[0][1]
FP=con_matrix[1][0]
TN=con_matrix[1][1]

#print(con_matrix)
#print(TN,FP,FN,TP)
print("True Positive-->The classifier model predicted "+str(TP)+" Live(Positive) samples as Live(Positive)")
print("False Negative-->The classifier model predicted "+str(FN)+" Live(Positive) samples as Fake(Negative)")
print("True Positive-->The classifier model predicted "+str(FP)+" Fake(Negative) samples as Live(Positive)")
print("True Negative-->The classifier model predicted "+str(TN)+" Fake(Negative) samples as Fake(Negative)")
print("Precision of the Linear SVM:", (TP / (TP+FP)))
print("Recall of the Linear SVM:", (TP / (TP+FN)))
print("Accuracy of the Linear SVM:", ((TP + TN) / (TP + TN + FP + FN)))


print("Precision",precision_score(labels2,pridicted_label_set,labels=["Live","Fake"],pos_label="Live"))
print("Recall",recall_score(labels2,pridicted_label_set,labels=["Live","Fake"],pos_label="Live"))
print("Accuracy",accuracy_score(labels2,pridicted_label_set))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np

import seaborn as sns
df = pd.read_csv(r"C:\Users\moham\Downloads\Book.csv")
#print data frame
print(df.head())


#describe el data
print(df.describe())
print("preprocessing data using standard scaler")
print("-----------------------------------------")
#make preprocessing to data
#1. remove null values
df.dropna(inplace=True)
#2. remove duplicates
df.drop_duplicates(keep='first')

# data using standard scaler

x, y = df.loc[:, df.columns != 'target'], df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
scaler = StandardScaler()
#normalize the data using standard scaler and remove mean and variance from data scale to unit values
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
def Build_Confusion_MAtrix(x_test,y_pred,model):
    # confusion matrix using metaplot
    cm = confusion_matrix(x_test, y_pred)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    display_labels = model.classes_)
    disp.plot()
    plt.show()
def Build_Model_Accurcy(model,x_train,x_test,y_train,y_test):
    # build model
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test data of "+str(model)+": ", accuracy * 100)

    return y_pred
def Train_test_split_Models(model):
    # train test split
    x = df.drop(columns='target', axis=1)
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y,random_state=0)
    # build model
    model = model.fit(x_train, y_train)
    x_test_pred = model.predict(x_test)
    # accuracy
    accuracy = accuracy_score(x_test_pred, y_test)
    print("Accuracy on test data of "+str(model)+": ", accuracy * 100)

    return y_pred


# Decision Tree classifier
classifier = DecisionTreeClassifier()
# build model
prediction=Build_Model_Accurcy(classifier,x_train,x_test,y_train,y_test)
#confusion matrix
Build_Confusion_MAtrix(y_test,prediction,classifier)

#KNN classifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
y_pred=Build_Model_Accurcy(classifier,x_train,x_test,y_train,y_test)
#confusion matrix
Build_Confusion_MAtrix(y_test,y_pred,classifier)

#SVM classifier

classifier = SVC(kernel='rbf', random_state=0)
y_pred = Build_Model_Accurcy(classifier,x_train,x_test,y_train,y_test)
#confusion matrix
Build_Confusion_MAtrix(y_test,y_pred,classifier)

#naive bayes classifier
classifier = GaussianNB()
y_pred = Build_Model_Accurcy(classifier,x_train,x_test,y_train,y_test)
#confusion matrix
Build_Confusion_MAtrix(y_test,y_pred,classifier)
print("\n")

########################################################################################################################

print("\n \n train test split \n ############################################################################################")

#train test split
# 1 - apply decision tree
dtree = DecisionTreeClassifier()
# build model
x_test_pred=Train_test_split_Models(dtree)

# 2 - apply KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
x_test_pred = Train_test_split_Models(knn)

#3 - naive bayes classifier
nb = GaussianNB()
x_test_pred = Train_test_split_Models(nb)

#apply SVM classifier
svm = SVC(kernel='rbf', C=1.0)
x_test_pred = Train_test_split_Models(svm)

print("\n \n cross validation")
print("______________________________________________________________________________________________________________")

#cross validation for decision tree
scores = cross_val_score(dtree, x, y, cv=10)
print("Accuracy of decision tree using cross validation: ",scores.mean()*100)
#cross validation for knn
scores = cross_val_score(knn, x, y, cv=10)
print("Accuracy of knn using cross validation: ",scores.mean()*100)
#cross validation for naive bayes
scores = cross_val_score(nb, x, y, cv=20)
print("Accuracy of naive bayes using cross validation: ",scores.mean()*100)
#cross validation for svm
scores = cross_val_score(svm, x, y, cv=10)
print("Accuracy of svm using cross validation: ",scores.mean()*100)
print("#######################################")




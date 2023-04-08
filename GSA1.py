import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib



# load the dataset
data = pd.read_csv("C:\\Users\ASUS\Desktop\cirrhosis.csv")
data.head()
data.tail()
list(data.columns)
data.describe()

data.isnull().sum()

new_data = data.replace(to_replace = np.nan, value = 0)
new_data = new_data.fillna(new_data.median())
new_data.isnull().sum()

new_data["Age"] = (new_data["Age"] / 365)
new_data["Age"] = new_data["Age"].astype("int64")

data = new_data.iloc[:, :len(new_data.columns) - 1]
labels = new_data.iloc[:, len(new_data.columns)-1]
class_names = labels.unique()

label_encoder = preprocessing.LabelEncoder()

new_data['Status'].unique()
new_data['Status'] = label_encoder.fit_transform(new_data['Status'])

new_data['Drug'].unique()
new_data['Drug'] = pd.to_numeric(new_data['Drug'], errors='coerce')
new_data['Drug'] = label_encoder.fit_transform(new_data['Drug'])

new_data['Sex'].unique()
new_data['Sex'] = label_encoder.fit_transform(new_data['Sex'])

new_data['Ascites'].unique()
new_data['Ascites'] = pd.to_numeric(new_data['Ascites'], errors='coerce')
new_data['Ascites'] = label_encoder.fit_transform(new_data['Ascites'])

new_data['Hepatomegaly'].unique()
new_data['Hepatomegaly'] = pd.to_numeric(new_data['Hepatomegaly'], errors='coerce')
new_data['Hepatomegaly'] = label_encoder.fit_transform(new_data['Hepatomegaly'])

new_data['Spiders'].unique()
new_data['Spiders'] = pd.to_numeric(new_data['Spiders'], errors='coerce')
new_data['Spiders'] = label_encoder.fit_transform(new_data['Spiders'])

new_data['Edema'].unique()
new_data['Edema'] = label_encoder.fit_transform(new_data['Edema'])

new_data.head()

new_data.describe()

plt.figure(figsize=(10,10))
sns.heatmap(new_data.corr(), cmap="RdBu", annot=True)
plt.show()



# Standardization
scaler = StandardScaler()
scaler.fit(new_data)
X_scaled = scaler.transform(new_data)



# set the number of features to be selected
n_features = 5

# set the gravitational constant
G = 100

# calculate the initial velocity
v = np.zeros(new_data.shape[1])

# initialize the position of the features
x = np.zeros(new_data.shape[1])
x[:n_features] = 1

# set the maximum number of iterations
max_iter = 100

# set the damping factor
c = 0.8

# set the time step
dt = 0.01

# initialize the mass of the features
m = np.ones(new_data.shape[1])

# start the GSA loop
for i in range(max_iter):
    # calculate the acceleration
    a = np.zeros(new_data.shape[1])
    for j in range(new_data.shape[1]):
        for k in range(new_data.shape[1]):
            if k != j:
                a[j] += G * m[k] * (x[k] - x[j]) / (np.linalg.norm(x[k] - x[j]) ** 3 + 1e-10)
    # update the velocity
    v = v + a * dt
    # update the position of the features
    x = x + v * dt
    # update the velocity using the damping factor
    v = c * v

# Get the index of the selected features
selected_features = np.argsort(x)[-n_features:]
selected_features_names = new_data.columns[selected_features]
print("Selected Features: ", selected_features)


# extract the features and target
X = new_data.iloc[:, selected_features]
y = new_data.iloc[:, -1]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic Regression, Logistic Regression ADA_Boost, Random forest, Random forest ADA_Boost, Linear SVM CLassifier, Neural Network, K-Nearest Neighbors (KNN) Classifier

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

Lr_pred = lr.predict(X_test)
Lr_cm = confusion_matrix(y_test, Lr_pred)

print(f"Classification Report of Logistic Regression : \n {classification_report(y_test, Lr_pred)}")

print("Confusion Matrix of Logistic Regression: \n")
sns.heatmap(Lr_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

joblib.dump(Lr_pred, "Lr_model.sav")

# Logistic Regression ADA_Boost
base_estimator = LogisticRegression()

ada_lr = AdaBoostClassifier(base_estimator=base_estimator)
ada_lr.fit(X_train, y_train)

Lr_pred_ada = ada_lr.predict(X_test)
Lr_cm_ada = confusion_matrix(y_test, Lr_pred_ada)

print(f"Classification Report of Logistic Regression ADA_Boost : \n {classification_report(y_test, Lr_pred_ada)}")

print("Confusion Matrix of Logistic Regression ADA_Boost: \n")
sns.heatmap(Lr_cm_ada,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

joblib.dump(Lr_pred_ada, "Lr_pred_ada.sav")


"""
# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_pred)

print(f"Classification Report of Random Forest : \n {classification_report(y_test, rf_pred)}")

print("Confusion Matrix of Random Forest: \n")
sns.heatmap(rf_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

joblib.dump(rf_pred, "rf_pred.sav")


# Random Forest Classifier ADA_Boost
base_estimator = RandomForestClassifier()

ada_rf = AdaBoostClassifier(base_estimator=base_estimator)
ada_rf.fit(X_train, y_train)

rf_pred_ada = ada_rf.predict(X_test)
rf_cm_ada = confusion_matrix(y_test, rf_pred_ada)

print(f"Classification Report of Random Forest Classifier : \n {classification_report(y_test, rf_pred_ada)}")

print("Confusion Matrix of Random Forest Classifier: \n")
sns.heatmap(rf_cm_ada,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

"""



# SVM CLassifier
svmpoly = SVC(kernel='poly', degree = 4)
svmpoly.fit(X_train, y_train)

svmpoly_pred = svmpoly.predict(X_test)
svmpoly_cm = confusion_matrix(y_test, svmpoly_pred)

print(f"Classification Report of SVM CLassifier : \n {classification_report(y_test, svmpoly_pred)}")

print("Confusion Matrix of SVM CLassifier: \n")
sns.heatmap(svmpoly_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

joblib.dump(svmpoly_pred, "svmpoly_pred.sav")


# SVM CLassifier Linear
svm = LinearSVC()
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_pred)

print(f"Classification Report of SVM CLassifier : \n {classification_report(y_test, svm_pred)}")

print("Confusion Matrix of SVM CLassifier: \n")
sns.heatmap(svm_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

joblib.dump(svm_pred, "svm_pred.sav")



# Neural Network
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

mlp_pred = mlp.predict(X_test)
mlp_cm = confusion_matrix(y_test, mlp_pred)

print(f"Classification Report of Neural Network : \n {classification_report(y_test, mlp_pred)}")

print("Confusion Matrix of Neural Network: \n")
sns.heatmap(mlp_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

joblib.dump(mlp_pred, "mlp_pred.sav")



# K-Nearest Neighbors (KNN) Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_cm = confusion_matrix(y_test, knn_pred)

print(f"Classification Report of KNN Classifier : \n {classification_report(y_test, knn_pred)}")

print("Confusion Matrix of KNN Classifier: \n")
sns.heatmap(knn_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

joblib.dump(knn_pred, "knn_pred.sav")


# ENSEMBLING MODELS

lr = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC(probability=True)
mlp = MLPClassifier()
knn = KNeighborsClassifier()

ensemble = VotingClassifier(estimators=[('lr', lr),('rf', rf),('svm', svm), ('mlp', mlp), ('knn', knn)], voting='soft')

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_cm = confusion_matrix(y_test, ensemble_pred)

print(f"Classification Report of Ensembling Model : \n {classification_report(y_test, ensemble_pred)}")

print("Confusion Matrix of Ensembling Model: \n")
sns.heatmap(ensemble_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.show()

joblib.dump(ensemble_pred, "ensemble_pred.sav")


# Algorithms Accuracy after using K-Fold
lr = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC()
nn = MLPClassifier()
Knn = KNeighborsClassifier()

k = 5
kf = KFold(n_splits=k, shuffle=True)

# initialize a list to store the accuracy scores
accuracies_lr = []
accuracies_rf = []
accuracies_dt = []
accuracies_svm = []
accuracies_nn = []
accuracies_knn = []


# iterate over the folds
for train_index, test_index in kf.split(X):
    # get the training and testing data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # fit the classifier on the training data
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    nn.fit(X_train, y_train)
    Knn.fit(X_train, y_train)
    
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_svm = svm.predict(X_test)
    y_pred_nn = nn.predict(X_test)
    y_pred_knn = Knn.predict(X_test)
    
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)

    accuracies_lr.append(accuracy_lr)
    accuracies_rf.append(accuracy_rf)
    accuracies_svm.append(accuracy_svm)
    accuracies_nn.append(accuracy_nn)
    accuracies_knn.append(accuracy_knn)
    
average_accuracy_lr = sum(accuracies_lr) / k
average_accuracy_rf = sum(accuracies_rf) / k
average_accuracy_svm = sum(accuracies_svm) / k
average_accuracy_nn = sum(accuracies_nn) / k
average_accuracy_knn = sum(accuracies_knn) / k


print("Average accuracy Linear Regreesion:", average_accuracy_lr)
print("Average accuracy Random Forest:", average_accuracy_rf)
print("Average accuracy SVM:", average_accuracy_svm)
print("Average accuracy NN:", average_accuracy_nn)
print("Average accuracy KNN:", average_accuracy_knn)

import pickle
pickle_out = open("ensemble.pkl", "wb")
pickle.dump(ensemble, pickle_out)
pickle_out.close()

pickle_out = open("Lr_model.pkl", "wb")
pickle.dump(lr, pickle_out)
pickle_out.close()

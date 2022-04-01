import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import statistics
from sklearn.model_selection import train_test_split, cross_validate, StratifiedShuffleSplit, learning_curve, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, power_transform, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

#Load data and clean it
dataset_train = pd.read_csv("../Datasets/Titanic/train.csv")
dataset_test = pd.read_csv("../Datasets/Titanic/test.csv")


def Preprocess(dataset):
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
    conditions = [
        (dataset['Age'] <= 18),
        (dataset['Age'] > 18) & (dataset['Age'] <= 60),
        (dataset['Age'] > 60)
    ]
    values = ['Child', 'Adult', 'Old']
    dataset['Age'] = np.select(conditions, values)
    dataset_cleaned = dataset.drop(['Cabin', 'Ticket', 'Name', 'PassengerId', 'Embarked', 'Parch', 'SibSp'], axis=1)
    if "Survived" in dataset_cleaned.columns :
        dataset_train = dataset_cleaned.drop(['Survived'], axis = 1)
        dataset_labels = dataset_cleaned['Survived']
    else:
        dataset_train = dataset_cleaned
        dataset_labels = []
    # dataset_train["Age"] = power_transform(dataset_train[["Age"]], method='box-cox')

    #Fix missing values in data
    si_mean = SimpleImputer(strategy='mean')
    si_mode = SimpleImputer(strategy='most_frequent')
    dataset_train[['Age']] = si_mode.fit_transform(dataset_train[['Age']]).ravel()
    dataset_train[["Fare"]] = np.log1p(dataset_train[["Fare"]])
    dataset_train[["Fare"]] = power_transform(dataset_train[["Fare"]], method='yeo-johnson')
    dataset_train[['Fare']] = dataset_train[['Fare']][(dataset_train[['Fare']] >= -2)]
    dataset_train[['Fare']] = si_mean.fit_transform(dataset_train[['Fare']]).ravel()
    dataset_train.hist()
    plt.show()

    #Encode Categorical data
    ohe = LabelEncoder()
    dataset_train[['Sex']] = ohe.fit_transform(dataset_train[['Sex']])
    dataset_train[['Age']] = ohe.fit_transform(dataset_train[['Age']])
    # dataset_train_add_1 = pd.DataFrame(Gender)
    # Embark = ohe.fit_transform(dataset_train[['Embarked']])
    # dataset_train_add_2 = pd.DataFrame(Embark)
    # dataset_train = dataset_train.drop(['Sex', 'Embarked'], axis = 1)
    # dataset_train = pd.concat([dataset_train, dataset_train_add_1, dataset_train_add_2], axis = 1, ignore_index=True)
    #Scale data
    ss = StandardScaler()
    dataset_train_scaled = ss.fit_transform(dataset_train.values)
    dataset_train = pd.DataFrame(dataset_train_scaled, columns= dataset_train.columns)
    return dataset_train, dataset_labels

Feature_train, Label_train = Preprocess(dataset_train)

Feature_test, Label_test = Preprocess(dataset_test)

#Train on classifiers
pipeline = Pipeline([
    ('clf', LogisticRegression())
])

score_matrix = {}

def classifer_pipeline(X, y):
    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)
    # X = X.to_numpy()
    y = y.to_numpy()
    skf = StratifiedKFold(n_splits=10)
    # sizes = np.linspace(0.3, 1.0, 10)
    score_dict = {}
    classifiers = [LogisticRegression(), SGDClassifier(), KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 20, criterion = 'entropy'), GaussianNB(), GradientBoostingClassifier(n_estimators=50)]
    for classifier in classifiers:
        score_list = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pipeline.set_params(clf = classifier)
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            scores = accuracy_score(y_test,predictions)
            cm = confusion_matrix(y_test, predictions)
            score_list = score_list + [scores]

        score_dict[classifier] = [statistics.mean(score_list), cm]
    return score_dict

predictions = classifer_pipeline(Feature_train, Label_train)
for key in predictions.keys():
    print("Classifier : {} , \t\t accuracy_score : {}, \n confusion_matix : {}".format(key,predictions[key][0], predictions[key][1]))

poly = PolynomialFeatures(2)
X = poly.fit_transform(Feature_train)

classifier = GradientBoostingClassifier(n_estimators=50)
classifier.fit(X, Label_train)

X = poly.transform(Feature_test)

predictions = classifier.predict(X).reshape(418, 1)
passengers = dataset_test['PassengerId']
submission = pd.DataFrame(list(zip(passengers, predictions)),
               columns =['PassengerId', 'Survived'])

submission.to_csv(r'.\Results.csv', index = False)
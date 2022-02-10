""" 
Written by Hammad Bin Ather and Aliza Lisan
UO ID: 951845958, 951846106
"""

import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from collections import Counter

archive_train = zipfile.ZipFile('/home/hammad/Desktop/UO/Winter 2022/CIS 507/Whats-Cooking/train.json.zip', 'r')
train_data = pd.read_json(archive_train.read('train.json'))

"""
Data Exploration

1. Structure of Data
"""
train_data.head()


"""
2. Most Frequent Cuisines
"""
cuisines_freq = train_data['cuisine'].value_counts()
cuisines_freq.plot(kind = "bar", figsize = (8, 5), title = "Frequency of Cuisines in Dataset")


"""
3. Top 10 Ingredients in each Cuisine
"""
cuisine_ing_counter = {}
for cuisine in train_data['cuisine'].unique():
    cuisine_ing_counter[cuisine] = Counter()
    index = train_data['cuisine'] == cuisine
    for ingredients in train_data[index]['ingredients']:
        cuisine_ing_counter[cuisine].update(ingredients)

ingredients = pd.Series((','.join([','.join(row["ingredients"]) for ind,row in train_data.iterrows()])).split(','))
common_ingredients = Counter(ingredients).most_common(20)

fig, axes = plt.subplots(figsize=(8, 5))
fig.tight_layout(pad = 5.0)
labels = [x[0] for x in common_ingredients]
count = [x[1] for x in common_ingredients]
axes.bar(labels, count, align = 'edge')
axes.set_title("20 most common ingredients", pad = 15.0)
fig.autofmt_xdate()

## Pie Charts
fig, axes = plt.subplots(5, 4, figsize=(20, 20))
fig.tight_layout(pad = 5.0)
index = 0
for cuisine in cuisine_ing_counter:
  ingredients = cuisine_ing_counter[cuisine].most_common(10)
  labels = [x[0] for x in ingredients]
  count = [x[1] for x in ingredients]
  
  axes.ravel()[index].pie(count,labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
  axes.ravel()[index].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  cuisine = cuisine.capitalize()
  cuisine= cuisine.replace('_', ' ')
  axes.ravel()[index].set_title(cuisine, pad = 15.0, fontsize = 16)
  index += 1


train_data['all_ingredients'] = train_data['ingredients'].map(";".join)


"""
Preprocessing
"""
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

countVector = CountVectorizer()
x = countVector.fit_transform(train_data['all_ingredients'].values)

labelEnc = LabelEncoder()
y = labelEnc.fit_transform(train_data.cuisine)
 

"""
Train/Test/Split
"""
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=101)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.875, random_state=101)


""" 
Confusion Matrix
"""
def performanceAnalysis(model, title):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns

    confusionMatrix = confusion_matrix(y_test, model.predict(x_test))
    # Normalise
    confusionMatrixNormalized = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    cuisines = train_data['cuisine'].value_counts().index
    sns.heatmap(confusionMatrixNormalized, annot=True, fmt='.2f', xticklabels=cuisines, yticklabels=cuisines)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    y_pred = logisticRegr.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=cuisines))


"""
Models

1. Logistic Regression
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

predicted = logisticRegr.predict(x_train)
print('\nLogistic Regression training accuracy score: {0:0.4f}'. format(accuracy_score(y_train, predicted)))

predicted = logisticRegr.predict(x_val)
print('Logistic Regression validation accuracy score: {0:0.4f}'. format(accuracy_score(y_val, predicted)))

predicted = logisticRegr.predict(x_test)
print('Logistic Regression testing accuracy score: {0:0.4f}\n'. format(accuracy_score(y_test, predicted)))

performanceAnalysis(logisticRegr, "Logistic Regression Confusion Matrix")

"""
2. SDG
"""
from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier()
SGD.fit(x_train, y_train)

predicted = SGD.predict(x_train)
print('SGD training accuracy score: {0:0.4f}'. format(accuracy_score(y_train, predicted)))

predicted = SGD.predict(x_val)
print('SGD validation accuracy score: {0:0.4f}'. format(accuracy_score(y_val, predicted)))

predicted = SGD.predict(x_test)
print('SGD testing accuracy score: {0:0.4f}\n'. format(accuracy_score(y_test, predicted)))


"""
Validation Curves for SGD
"""

def validationCurves(model):
    from sklearn.model_selection import validation_curve
    parameter_range = ["constant", "optimal", "adaptive"]
    train_score, test_score = validation_curve(model(eta0=0.1), x, y,
                                        param_name = "learning_rate",
                                        param_range = parameter_range, 
                                        scoring = "accuracy")

    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis = 1)
    std_train_score = np.std(train_score, axis = 1)
    
    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis = 1)
    std_test_score = np.std(test_score, axis = 1)
    
    # Plot mean accuracy scores for training and testing scores
    plt.plot(parameter_range, mean_train_score,
        label = "Training Score", color = 'b')
    plt.plot(parameter_range, mean_test_score,
    label = "Cross Validation Score", color = 'g')
    
    # Creating the plot
    plt.title("Validation Curve with SGD Classifier")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.xticks(rotation=30)

    plt.show()

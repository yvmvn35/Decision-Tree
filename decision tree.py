import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


data = pd.read_csv("winequality-red.csv")
print(data.head())
print(data.tail())

x = data.drop(columns=['fixed acidity'])
y = data['quality']

column_target = 'sulphates'
feature = data.drop(columns=[column_target])
label = data[column_target]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=60)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("F1 score:", f1)

#visualitation
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=feature.columns, class_names=[str(i) for i in clf.classes_])
plt.show()

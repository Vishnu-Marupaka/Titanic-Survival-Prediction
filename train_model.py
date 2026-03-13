import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

data = pd.read_csv("train.csv")

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

features = ["Age", "Sex", "Pclass", "Fare"]
X = data[features]
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=8, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, 'model.joblib')
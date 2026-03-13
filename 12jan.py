import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\vishn\Downloads\train.csv")

survived_by_class = data.groupby('Pclass')['Survived'].mean() * 100
plt.figure(figsize=(6, 4))
survived_by_class.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class (1=1st, 2=2nd, 3=3rd)')
plt.ylabel('Survival Rate (%)')
plt.xticks(rotation=0)
plt.show()

# 2. Fill missing Age values with the average 
data['Age'] = data['Age'].fillna(data['Age'].mean())

# 3. Encode Sex
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})


features = ["Age", "Sex", "Pclass", "Fare"]
X = data[features]
y = data["Survived"]

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(max_depth=8, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy with Imputation & Fare: {model.score(X_test, y_test):.4f}")
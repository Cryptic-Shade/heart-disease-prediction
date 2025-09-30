# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("heart - heart.csv")  # Ensure file path is correct

# 1. Print number of records and number of features
print("Number of Records:", df.shape[0])  # df.shape[0] = number of rows
print("Number of Features:", df.shape[1])  # df.shape[1] = number of columns

# 2. Display information
print("\n--- Data Info ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())

# 3. Display description
print("\n--- Description ---")
print(df.describe())

# 4. Print mean of Age
print("\nMean Age:", df["Age"].mean())

# 5. Histogram for Age
plt.hist(df["Age"], bins=15, color="skyblue", edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Value Counts for Sex
print("\n--- Gender Counts ---")
print(df['Sex'].value_counts())

# 7. Bar chart for Gender Distribution
gender_counts = df["Sex"].value_counts()
plt.bar(gender_counts.index, gender_counts.values, color="lightgreen", edgecolor="black")
plt.title("Gender Distribution")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 8. Bar chart: Heart Disease in patients with RestingBP > 170
high_bp = df[df["RestingBP"] > 170]
hd_counts = high_bp["HeartDisease"].value_counts().sort_index()

plt.bar(["No Disease", "Heart Disease"], hd_counts.values, color="salmon", edgecolor="black")
plt.title("Heart Disease in Patients with RestingBP > 170")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 9. Histogram for Cholesterol
plt.hist(df["Cholesterol"], bins=20, color="orange", edgecolor="black")
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol (mg/dL)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Split the Data
# Select features
x = df[["ST_Slope", "ExerciseAngina", "ChestPainType", "RestingBP", "RestingECG"]]
y = df["HeartDisease"]

# Convert categorical columns to numeric using get_dummies
x = pd.get_dummies(x, drop_first=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# 11. Scaling the Features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 12. Train the SVM Model
model = SVC()
model.fit(x_train, y_train)

# 13. Predictions & Accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Accuracy ---")
print(f"{accuracy * 100:.2f}%")  # Display accuracy percentage
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
    

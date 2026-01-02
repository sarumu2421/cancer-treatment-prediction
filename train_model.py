import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data_clinical_patient.txt", sep="\t", comment="#")

# print(data.head())
# print(data.shape)
 
data["responder"] = data["RFS_STATUS"].map({
    "0:Not Recurred": 1,  # responder
    "1:Recurred": 0      # non-responder
})  
data = data.dropna(subset=["responder"])
print(data["responder"].value_counts())

features = [
    "AGE_AT_DIAGNOSIS",
    "LYMPH_NODES_EXAMINED_POSITIVE",
    "CELLULARITY",
    "HISTOLOGICAL_SUBTYPE",
    "INFERRED_MENOPAUSAL_STATE",
    "INTCLUST"
]  

X = data[features].copy()
y = data["responder"] 

#encode categorical features as numeric values so they can be used by the model
encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

X = X.fillna(X.median())

# #data visualization
# sns.countplot(x="responder", data=data)
# plt.title("Responder vs Non-responder Distribution")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
) 

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train) 

y_pred = model.predict(X_test) 

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%") 

# cm = confusion_matrix(y_test, y_pred) 

# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix: Treatment Response Prediction")

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show() 

# print(classification_report(y_test, y_pred))

# importances = pd.DataFrame({
#     "Feature": X.columns,
#     "Importance": model.feature_importances_
# }).sort_values(by="Importance", ascending=False)

# sns.barplot(x="Importance", y="Feature", data=importances)
# plt.title("Feature Importance")
# plt.show() 

joblib.dump(model, "model.joblib")
joblib.dump(encoders, "encoders.joblib")
joblib.dump(features, "features.joblib")
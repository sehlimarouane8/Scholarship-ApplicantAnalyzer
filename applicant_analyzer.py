import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_FILE = "applicants.csv"

def load_data():
    df = pd.read_csv(DATA_FILE)
    # Convert categorical data
    df["Degree"] = df["Degree"].map({"High School":0, "Bachelor":1, "Master":2})
    df["English_Level"] = df["English_Level"].map({"Beginner":0, "Intermediate":1, "Advanced":2})
    return df

def train_model(df):
    X = df[["GPA", "Degree", "English_Level", "Extra_Curriculars"]]
    y = df["Accepted"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    
    return model

def predict_applicant(model, gpa, degree, english, extracurriculars):
    mapping = {"High School":0, "Bachelor":1, "Master":2}
    english_map = {"Beginner":0, "Intermediate":1, "Advanced":2}
    features = [[gpa, mapping[degree], english_map[english], extracurriculars]]
    pred = model.predict(features)
    return "Accepted ✅" if pred[0]==1 else "Rejected ❌"

if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    
    print("\nPredict your scholarship chance:")
    gpa = float(input("GPA (0-4): "))
    degree = input("Degree (High School/Bachelor/Master): ")
    english = input("English Level (Beginner/Intermediate/Advanced): ")
    extra = int(input("Extra Curricular Activities (number): "))
    
    result = predict_applicant(model, gpa, degree, english, extra)
    print(f"\nYour predicted result: {result}")

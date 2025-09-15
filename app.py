import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Load or create dataset
# -----------------------------
@st.cache_data
def load_data():
    # Small demo dataset
    data = {
        "Age": [18, 20, 21, 22, 23, 24, 25, 26],
        "Gender": ["Male", "Female", "Female", "Male", "Male", "Female", "Male", "Female"],
        "GPA": [2.1, 3.4, 1.8, 2.7, 3.1, 2.5, 1.9, 3.6],
        "Parental_Education": ["HighSchool", "College", "HighSchool", "Masters",
                               "College", "College", "HighSchool", "PhD"],
        "DroppedOut": [1, 0, 1, 0, 0, 0, 1, 0]
    }
    return pd.DataFrame(data)

df = load_data()

# -----------------------------
# 2. Split features/target
# -----------------------------
X = df.drop("DroppedOut", axis=1)
y = df["DroppedOut"]

categorical = X.select_dtypes(include="object").columns.tolist()
numeric = X.select_dtypes(exclude="object").columns.tolist()

# -----------------------------
# 3. Preprocessing + Model
# -----------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("🎓 Student Dropout Prediction")

st.write("This app predicts whether a student is likely to drop out based on simple features.")

st.sidebar.header("Enter Student Info")

age = st.sidebar.slider("Age", 16, 30, 20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gpa = st.sidebar.slider("GPA", 0.0, 4.0, 2.5)
parent_edu = st.sidebar.selectbox("Parental Education", ["HighSchool", "College", "Masters", "PhD"])

input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "GPA": [gpa],
    "Parental_Education": [parent_edu]
})

prediction = pipeline.predict(input_df)[0]
proba = pipeline.predict_proba(input_df)[0][1]

st.subheader("📊 Prediction Result")
st.write("**Likely to Dropout**" if prediction == 1 else "**Likely to Continue**")
st.write(f"Probability of Dropout: {proba:.2f}")

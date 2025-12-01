# train_models.py
# Trains regression + classification models on student_data.csv (or creates synthetic data),
# evaluates them and saves best models and result CSVs in ./models/

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ------------- Config -------------
DATA_PATH = "student_data.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
RANDOM_STATE = 42
# ----------------------------------

# Create synthetic dataset if not present
if not os.path.exists(DATA_PATH):
    np.random.seed(RANDOM_STATE)
    n = 100
    df = pd.DataFrame({
        "StudentID": range(1,n+1),
        "Attendance": np.random.randint(50,100,n),
        "StudyHours": np.random.randint(1,6,n),
        "PastScore": np.random.randint(40,95,n),
        "SleepHours": np.random.randint(4,9,n),
        "Assignments": np.random.randint(40,100,n),
        "InternetUse": np.random.randint(1,4,n),
        "LabScore": np.random.randint(40,100,n),
        "ParentEdu": np.random.randint(0,3,n),
        "ParentIncome": np.random.randint(20000,70000,n),
        "Tuition": np.random.randint(0,2,n),
    })
    df["FinalScore"] = (
        0.3*df["PastScore"] +
        0.2*df["Attendance"] +
        0.2*df["StudyHours"]*10 +
        0.1*df["Assignments"] +
        0.1*df["LabScore"] +
        np.random.randint(-5,6,n)
    ).astype(int)
    df["PassFail"] = (df["FinalScore"]>=60).astype(int)
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset created at {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# Define features and targets
FEATURES = ["Attendance","StudyHours","PastScore","SleepHours","Assignments","InternetUse","LabScore","ParentEdu","ParentIncome","Tuition"]
TARGET_REG = "FinalScore"
TARGET_CLS = "PassFail"

X = df[FEATURES]
y_reg = df[TARGET_REG]
y_cls = df[TARGET_CLS]

# Train/test split (same split for reproducibility)
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.20, random_state=RANDOM_STATE
)

print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

# Standard scaler for algorithms that require scaling
scaler = StandardScaler()

# Regression models
regressors = {
    "LinearRegression": Pipeline([("scaler", scaler), ("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", scaler), ("model", Ridge(random_state=RANDOM_STATE))]),
    "Lasso": Pipeline([("scaler", scaler), ("model", Lasso(random_state=RANDOM_STATE))]),
    "KNNRegressor": Pipeline([("scaler", scaler), ("model", KNeighborsRegressor())]),
    "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_STATE),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE)
}

# try to include xgboost if available
xgb_available = False
try:
    from xgboost import XGBRegressor
    regressors["XGBoostRegressor"] = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    xgb_available = True
except Exception:
    pass

reg_results = []
best_reg_score = -float("inf")
best_reg_model = None
best_reg_name = None

for name, model in regressors.items():
    print("Training regressor:", name)
    model.fit(X_train, y_reg_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_reg_test, preds)
    mse = mean_squared_error(y_reg_test, preds)
    mae = mean_absolute_error(y_reg_test, preds)
    print(f" -> R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
    reg_results.append((name, r2, mse, mae))
    if r2 > best_reg_score:
        best_reg_score = r2
        best_reg_model = model
        best_reg_name = name

# Save best regression model
reg_path = os.path.join(MODELS_DIR, f"best_regression_{best_reg_name}.pkl")
joblib.dump(best_reg_model, reg_path)
print("Saved best regression:", reg_path)

# Classification models
classifiers = {
    "LogisticRegression": Pipeline([("scaler", scaler), ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "KNNClassifier": Pipeline([("scaler", scaler), ("model", KNeighborsClassifier())]),
    "SVC": Pipeline([("scaler", scaler), ("model", SVC(probability=True, random_state=RANDOM_STATE))]),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "GaussianNB": GaussianNB(),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

if xgb_available:
    try:
        from xgboost import XGBClassifier
        classifiers["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    except Exception:
        pass

cls_results = []
best_cls_score = -float("inf")
best_cls_model = None
best_cls_name = None

for name, model in classifiers.items():
    print("Training classifier:", name)
    model.fit(X_train, y_cls_train)
    preds = model.predict(X_test)
    # ROC-AUC handling
    roc = float("nan")
    try:
        probs = model.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_cls_test, probs)
    except Exception:
        try:
            probs = model.decision_function(X_test)
            roc = roc_auc_score(y_cls_test, probs)
        except Exception:
            roc = float("nan")
    acc = accuracy_score(y_cls_test, preds)
    prec = precision_score(y_cls_test, preds, zero_division=0)
    rec = recall_score(y_cls_test, preds, zero_division=0)
    f1 = f1_score(y_cls_test, preds, zero_division=0)
    print(f" -> Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc if not pd.isna(roc) else 'NA'}")
    cls_results.append((name, acc, prec, rec, f1, roc))
    if f1 > best_cls_score:
        best_cls_score = f1
        best_cls_model = model
        best_cls_name = name

# Save best classifier
cls_path = os.path.join(MODELS_DIR, f"best_classifier_{best_cls_name}.pkl")
joblib.dump(best_cls_model, cls_path)
print("Saved best classifier:", cls_path)

# Feature importance (if available)
def show_feature_importance(model, name):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print(f"\nFeature importances for {name}:\n{imp}")
    else:
        print(f"\n{name} has no feature_importances_")

print("\nFeature importance checks:")
show_feature_importance(best_reg_model, best_reg_name)
show_feature_importance(best_cls_model, best_cls_name)

# Save results to CSV
reg_df = pd.DataFrame(reg_results, columns=["Model","R2","MSE","MAE"])
cls_df = pd.DataFrame(cls_results, columns=["Model","Accuracy","Precision","Recall","F1","ROC_AUC"])
reg_df.to_csv(os.path.join(MODELS_DIR, "regression_results.csv"), index=False)
cls_df.to_csv(os.path.join(MODELS_DIR, "classification_results.csv"), index=False)
print("\nSaved evaluation CSVs in", MODELS_DIR)

# Example prediction
example = X_test.iloc[[0]]
print("\nExample features:", example.to_dict(orient='records')[0])
reg_loaded = joblib.load(reg_path)
cls_loaded = joblib.load(cls_path)
print("Predicted FinalScore:", reg_loaded.predict(example)[0])
print("Predicted Pass/Fail:", cls_loaded.predict(example)[0])

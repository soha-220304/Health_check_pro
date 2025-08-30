import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

DATASET_PATH = "health_dataset.csv"
RF_PATH = "rf_model.joblib"
DL_PATH = "dl_model.h5"
SCALER_PATH = "scaler.joblib"
LE_PATH = "label_encoder.joblib"

CUSTOM_STATIC_FOLDER = 'static'
app = Flask(__name__, template_folder='TEMPLETE', static_folder=CUSTOM_STATIC_FOLDER)
CORS(app)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------
# 1) Load or create dataset
# -------------------------
if not os.path.exists(DATASET_PATH):
    # Create demo dataset (same distribution you provided)
    # print("Dataset not found — creating demo dataset 'health_dataset.csv' (500 rows).")
    demo = pd.DataFrame({
        "Age": np.random.randint(20, 80, 500),
        "BP": np.random.randint(90, 180, 500),
        "Weight": np.random.randint(45, 120, 500),
        "HeartRate": np.random.randint(60, 110, 500),
        "Smoking": np.random.randint(0, 2, 500),
        "PastDiagnosis": np.random.randint(0, 3, 500),
        "Disease": np.random.choice(["Healthy", "Heart Disease", "Hypertension", "Obesity", "Diabetes"], 500)
    })
    demo.to_csv(DATASET_PATH, index=False)

df = pd.read_csv(DATASET_PATH)
# print(f"Loaded dataset: {DATASET_PATH}  —  shape: {df.shape}")
# print(df.head())

# -------------------------
# 2) Preprocess
# -------------------------
FEATURE_COLS = ["Age", "BP", "Weight", "HeartRate", "Smoking", "PastDiagnosis"]
TARGET_COL = "Disease"

# Drop rows with missing required columns
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

# Label encode Disease
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL].astype(str))
X = df[FEATURE_COLS].astype(float).values

# Save label encoder for future inverse transform
joblib.dump(le, LE_PATH)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

num_classes = len(le.classes_)
# print(f"Classes ({num_classes}): {list(le.classes_)}")

# -------------------------
# 3) Train / Load RandomForest (ML)
# -------------------------
if os.path.exists(RF_PATH):
    # print("Loading saved RandomForest model...")
    rf = joblib.load(RF_PATH)
else:
    # print("Training RandomForest (with a small grid search)...")
    rf_base = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    # small grid to keep runtime reasonable
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    }
    grid = GridSearchCV(rf_base, param_grid, cv=3, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    # print("RF best params:", grid.best_params_)
    joblib.dump(rf, RF_PATH)
    # print("Saved RF to", RF_PATH)

# Evaluate RF
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
# print(f"RandomForest accuracy on test: {rf_acc*100:.2f}%")

# -------------------------
# 4) Train / Load DL model (Keras)
# -------------------------
if os.path.exists(DL_PATH):
    # print("Loading saved DL model...")
    dl_model = tf.keras.models.load_model(DL_PATH)
else:
    # print("Training Keras ANN (DL)...")
    dl_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
    # We use sparse labels for training convenience
    history = dl_model.fit(
        X_train, y_train,
        validation_split=0.12,
        epochs=200,
        batch_size=32,
        callbacks=[early],
        verbose=0
    )
    dl_model.save(DL_PATH)
    # print("Saved DL model to", DL_PATH)

# Evaluate DL
dl_eval = dl_model.evaluate(X_test, y_test, verbose=0)
dl_acc = float(dl_eval[1])
# print(f"DL model accuracy on test: {dl_acc*100:.2f}%")

# -------------------------
# 5) Compute ensemble weights
# -------------------------
# Use accuracies as weights (add small eps to avoid zeros)
eps = 1e-6
w_rf = rf_acc + eps
w_dl = dl_acc + eps
# print(f"Weights before normalization: RF={w_rf:.4f}, DL={w_dl:.4f}")

# Normalize so they sum to 1
w_sum = w_rf + w_dl
w_rf /= w_sum
w_dl /= w_sum
# print(f"Normalized weights: RF={w_rf:.3f}, DL={w_dl:.3f}")

# -------------------------
# 6) Function to get ensemble prediction
# -------------------------
def ensemble_predict_proba(user_array_scaled):
    """
    user_array_scaled: numpy array shape (n_samples, n_features) already scaled
    returns: (final_probs, final_label_index)
    """
    # RF predict_proba (returns shape (n_samples, n_classes))
    rf_proba = rf.predict_proba(user_array_scaled)  # shape (n, K)
    # DL predict (probabilities)
    dl_proba = dl_model.predict(user_array_scaled)  # shape (n, K)
    # Check that columns align with label encoder classes_
    # Most likely both are in the same class order because both were trained from same y (0..K-1)
    # Weighted average
    final_proba = w_rf * rf_proba + w_dl * dl_proba
    final_label_idx = np.argmax(final_proba, axis=1)
    return final_proba, final_label_idx

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = data['Age']
    bp = data['BP']
    weight = data['Weight']
    hr = data['HeartRate']
    smoking = data['Smoking']
    past_diag = data['PastDiagnosis']

    # Prepare numeric array and scale
    user_row = np.array([[age, bp, weight, hr, smoking, past_diag]], dtype=float)
    user_row_scaled = scaler.transform(user_row)

    # Ensemble prediction
    final_proba, final_idx = ensemble_predict_proba(user_row_scaled)
    final_idx = int(final_idx[0])
    final_label = le.inverse_transform([final_idx])[0]
    final_confidence = float(final_proba[0, final_idx])

    # Append to dataset
    new_row = {
        "Age": int(age),
        "BP": int(bp),
        "Weight": int(weight),
        "HeartRate": int(hr),
        "Smoking": int(smoking),
        "PastDiagnosis": int(past_diag),
        "Disease": final_label
    }
    global df
    df_appended = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df_appended
    df_appended.to_csv(DATASET_PATH, index=False)

    return jsonify({
        'prediction': final_label,
        'confidence': f"{final_confidence*100:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)

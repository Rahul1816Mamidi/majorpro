# Data & Project Architecture Documentation

## 6. Datasets Used in the Project

This project fuses **four distinct types of data** (Quad-Modal) to create a comprehensive profile of maternal and fetal health.

### A. Clinical & Physiological Data (Tabular)
*   **Dataset Name:** Maternal Health Risk Data Set (Enhanced)
*   **Source:** UCI Machine Learning Repository / IoT Health collection.
*   **Type:** Structured CSV (Tabular data).
*   **Content:**
    *   **Vitals:** Systolic BP, Diastolic BP, Blood Sugar (BS), Body Temperature, Heart Rate.
    *   **Demographics:** Age.
    *   **Socio-Economic/Lifestyle (Augmented):** Education, Income, Urban/Rural status, Sleep Hours, Physical Activity Level, Stress Score, Diet Quality.
*   **Contribution:** Provides the **baseline physiological context**. It tells us the "General Health State" of the mother (e.g., Is she hypertensive? Diabetic?).
*   **Processing:**
    *   **Cleaning:** Missing values imputed with mean.
    *   **Encoding:** Categorical text (e.g., "Urban") converted to numbers (0/1).
    *   **Scaling:** Normalized using Z-score standardization ($z = \frac{x - \mu}{\sigma}$) so all numbers are on the same scale.

### B. Cardiotocography (CTG) Data (Time-Series)
*   **Dataset Name:** UCI Cardiotocography Dataset.
*   **Source:** UCI Machine Learning Repository.
*   **Type:** Electronic Fetal Monitor signals (Time-Series / Signal features).
*   **Content:**
    *   **FHR:** Fetal Heart Rate baseline.
    *   **Variability:** Short-term (STV) and Long-term variability (LTV).
    *   **Events:** Accelerations (AC), Decelerations (Late, Early, Variable), Uterine Contractions (UC).
*   **Contribution:** Provides the **immediate fetal status**. It tells us if the baby is in distress *right now* (e.g., hypoxic).
*   **Processing:**
    *   treated as a sequence of 11 simultaneous signal channels.
    *   **Imputation:** Missing signal segments filled with statistical means.

### C. Physical Activity Data (Sensor Stream)
*   **Dataset Name:** MHEALTH (Mobile Health) Dataset.
*   **Source:** UCI Machine Learning Repository.
*   **Type:** High-frequency Sensor Data (100Hz).
*   **Content:** Tri-axial acceleration (X, Y, Z) from chest-mounted wearable sensors.
*   **Contribution:** Provides **contextual movement data**. It helps distinguish between "High Heart Rate due to Exercise" (Normal) vs. "High Heart Rate due to Distress" (Abnormal).
*   **Processing:**
    *   **Windowing:** Sliced into "windows" of 50 time-steps to capture movement *patterns* rather than single points.

### D. Ultrasound Imagery (Visual)
*   **Dataset Name:** Fetal Plane Ultrasound Dataset (Simulated/Proxy for prototype).
*   **Source:** Medical Imaging Archives.
*   **Type:** Unstructured 2D Images.
*   **Content:** Grayscale scans of fetal anatomy (Head circumference, Abdominal circumference).
*   **Contribution:** Provides **anatomical growth data**. It is crucial for the "Fetal Weight" prediction task (detecting growth restriction).
*   **Processing:**
    *   **Resizing:** All images resized to standard 128x128 pixels.
    *   **Normalization:** Pixel values scaled from [0, 255] to [0, 1].

---

## 7. Algorithms & Techniques Applied

### Phase 1: Preprocessing & Feature Engineering
1.  **StandardScaler (Scikit-Learn):**
    *   *Where:* Clinical Data.
    *   *Why:* Algorithms like Neural Networks fail if inputs vary wildly (e.g., Age=25 vs. BP=120). This scales everything to have Mean=0, Variance=1.
2.  **LabelEncoder:**
    *   *Where:* Categorical features (e.g., Risk Level: Low $\to$ 0, Mid $\to$ 1, High $\to$ 2).
3.  **Sliding Window Technique:**
    *   *Where:* Activity Sensor Data.
    *   *Why:* Converts a continuous stream into discrete "events" (chunks of 50 steps) that an LSTM can analyze.

### Phase 2: Model Architecture (Deep Learning)
4.  **Dense (Fully Connected) Layers:**
    *   *Where:* Clinical Encoder.
    *   *Role:* Pattern matching for simple tabular numbers.
5.  **CNN (Convolutional Neural Networks):**
    *   *Where:* CTG Encoder (1D Conv) and Image Encoder (2D Conv / MobileNetV2).
    *   *Role:* Feature Extraction. It scans the data to find "shapes" (e.g., a dip in heart rate or a curve in an image).
6.  **LSTM (Long Short-Term Memory):**
    *   *Where:* CTG and Activity Encoders.
    *   *Role:* Temporal Learning. It remembers the *sequence* of events (e.g., "Heart rate dropped *after* contraction").
7.  **Attention Mechanism (Custom Algorithm):**
    *   *Where:* On top of LSTMs.
    *   *Role:* "Focusing." It assigns weights to specific time steps, allowing the model to ignore noise and focus on critical events (like a sudden deceleration).
    *   *Math:* $\alpha_t = \text{softmax}(\tanh(W h_t + b))$.

### Phase 3: Training & Optimization
8.  **Adam Optimizer:**
    *   *Algo:* Adaptive Moment Estimation.
    *   *Role:* Adjusts the learning rate automatically for each parameter, converging faster than standard Gradient Descent.
9.  **Class Weighting:**
    *   *Algo:* Inverse Frequency weighting.
    *   *Role:* Handling Imbalance. Since "High Risk" is rare, we tell the model: "Pay 5x more attention to High Risk errors than Low Risk errors."
10. **Early Stopping:**
    *   *Role:* Prevents Overfitting. Stops training if validation accuracy stops improving for 15 epochs.

---

## 8. End-to-End Project Flow (How it was Built)

### Step 1: Problem Definition & Data Collection
*   **Goal:** Build a system to predict Risk (Classification) and Weight (Regression).
*   **Action:** Collected 4 disparate datasets (CSV, Excel, Log files, Images).

### Step 2: Data Ingestion & Cleaning (`data_loader.py` & `preprocessing.py`)
*   **Raw Data:** Loaded using Pandas.
*   **Sanitization:** Removed nulls, dropped duplicate rows.
*   **Fusion Strategy:** Decided on **Late Fusion**. We kept datasets separate initially to process them with specialized algos.

### Step 3: Model Architecture Design (`model.py`)
*   **Design Choice:** **Decoupled Twin-Tower**.
    *   *Tower 1 (Risk):* Inputs = Clinical + CTG + Activity.
    *   *Tower 2 (Weight):* Inputs = Clinical + Image.
*   **Implementation:** Built using Keras Functional API. Included **Attention Blocks** to handle noisy sensor data.

### Step 4: Training (`train.py`)
*   **Split:** 80% Training, 20% Testing (Stratified to keep risk ratios equal).
*   **Weights:** Calculated Class Weights to fix imbalance.
*   **Execution:** Model trained for 70 epochs.
*   **Optimization:**
    *   Monitor `val_loss`.
    *   If stuck, `ReduceLROnPlateau` lowers learning rate by 50%.
    *   Save best model to `best_maternal_model.keras`.

### Step 5: Evaluation & Explainability (`evaluate_graphs.py`, `explainability.py`)
*   **Metrics:** Checked Accuracy (Risk) and Mean Absolute Error (Weight).
*   **Transparency:**
    *   Applied **SHAP** to find global risk factors.
    *   Applied **LIME** to explain individual predictions.

### Step 6: Deployment (`app.py`)
*   **Interface:** Built a Streamlit Web App.
*   **Features:**
    *   User inputs patient data.
    *   Model runs inference in real-time.
    *   **PDF Generation:** Generates a doctor's report.
    *   **Scenario Comparison:** Allows "What-If" analysis.

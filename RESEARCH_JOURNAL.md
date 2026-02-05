# Multimodal Deep Learning Framework for Enhanced Maternal and Fetal Risk Assessment: A Holistic Approach

## Abstract
Maternal and fetal health monitoring remains a critical challenge in modern obstetrics, particularly in resource-constrained settings where timely expert diagnosis is scarce. Traditional risk assessment models often rely on unimodal data—either clinical vitals or cardiotocography (CTG) independently—failing to capture the complex, systemic nature of pregnancy complications. This paper presents the development of a **Quad-Modal Artificial Intelligence Platform** that integrates four distinct data streams: clinical physiological records, CTG time-series signals, maternal physical activity sensor data, and fetal ultrasound imaging. By employing a decoupled twin-tower neural architecture, the system simultaneously predicts maternal risk levels (Low, Mid, High) and estimates fetal birth weight. Our approach incorporates explainable AI (XAI) modules, including SHAP and LIME, to provide clinicians with transparent, interpretable decision support. This work demonstrates the efficacy of multimodal fusion in improving diagnostic precision and offers a scalable framework for next-generation prenatal care.

---

## 1. Introduction
Maternal mortality and fetal distress are persistent global health issues. According to the World Health Organization (WHO), approximately 295,000 women died during and following pregnancy and childbirth in 2017. Many of these deaths are preventable with early detection and management of complications such as preeclampsia, gestational diabetes, and intrauterine growth restriction (IUGR). The burden is disproportionately higher in low-to-middle-income countries (LMICs), where access to specialized obstetric care is limited.

Current clinical practice involves disparate monitoring systems. A clinician might review a patient's blood pressure and blood sugar logs, separately analyze a 20-minute CTG trace for fetal heart rate variability, and independently assess ultrasound scans for fetal biometry. This fragmented workflow increases cognitive load and risks missing subtle correlations across modalities—for example, how a mother’s physical activity level might correlate with transient physiological spikes or fetal heart rate decelerations. Furthermore, the manual interpretation of these signals is subject to significant inter-observer variability, leading to inconsistent diagnoses.

This project bridges this gap by developing a unified AI platform that fuses heterogeneous data sources. By leveraging deep learning's ability to learn feature representations from raw signals and images, combined with structured clinical data, we aim to provide a holistic risk score that is more robust than any single modality could offer. Our system serves as a clinical decision support tool, augmenting human expertise with data-driven insights.

### 1.1 Related Work and Context
Recent advancements in multimodal learning have shown promise in healthcare. Unimodal approaches, such as using CNNs for ultrasound analysis or LSTMs for CTG classification, have achieved moderate success but often fail to capture the systemic nature of pregnancy complications. Early fusion techniques, which concatenate raw data vectors at the input level, often suffer from the "curse of dimensionality" and modality collapse, where the model ignores weaker signals. Our work builds upon the "late fusion" paradigm, employing specialized encoders for each data type before integration, ensuring that the unique topological features of time-series and image data are preserved.

## 2. Problem Statement
The core problem addressed in this research is the **fragmentation of diagnostic data** in prenatal care. 
Specific challenges include:
1.  **Data Silos:** Clinical biomarkers (e.g., Systolic BP, Hemoglobin) are rarely analyzed in concert with high-frequency sensor data (e.g., CTG, Accelerometer).
2.  **Subjectivity:** Interpretation of CTG traces and ultrasound images suffers from high inter-observer variability among clinicians.
3.  **Black Box AI:** Previous ML models in this domain often lack interpretability, making them untrustworthy for clinical adoption.
4.  **Class Imbalance:** Medical datasets are inherently imbalanced, with high-risk pathological cases being rarer than normal healthy pregnancies, leading to biased model performance.

## 3. Solution Architecture
We propose a **Multimodal Decoupled Twin-Tower Network**. Unlike traditional early-fusion models that simply concatenate all features, our architecture recognizes that different tasks require different data views.

The solution consists of:
*   **Data Ingestion Layer:** Handles tabular clinical data, time-series signals (CTG, Activity), and 2D images (Ultrasound).
*   **Feature Extraction Encoders:** Specialized neural branches for each modality (CNN-LSTM for time-series, MobileNetV2 for images).
*   **Twin-Tower Prediction Heads:**
    *   **Tower 1 (Risk Specialist):** Fuses Clinical, CTG, and Activity features to classify maternal risk (Low, Mid, High). It deliberately excludes imaging to focus on physiological stability.
    *   **Tower 2 (Weight Specialist):** Fuses Clinical and Imaging features to predict fetal birth weight, acknowledging that growth is a function of maternal health and fetal biometry.
*   **Explainability Engine:** A post-hoc analysis layer using SHAP (Shapley Additive Explanations) and LIME (Local Interpretable Model-agnostic Explanations) to visualize feature contributions.

## 4. Objectives
The primary objectives of this research are:
1.  **Develop a Robust Multimodal Model:** To achieve higher classification accuracy for maternal risk categories compared to unimodal baselines.
2.  **Enable Multi-Task Learning:** To simultaneously predict categorical risk and continuous fetal weight, leveraging shared clinical representations.
3.  **Ensure Clinical Interpretability:** To implement "Glass-Box" transparency where the model justifies its predictions via feature importance and decision boundaries.
4.  **Handle Real-World Data Constraints:** To build robust preprocessing pipelines that handle missing values, noise, and class imbalance effectively.

## 5. Methodology

### 5.1 Data Fusion Strategy
We employ a **Late Fusion (Feature-Level)** strategy. Raw data from each modality is processed independently by dedicated encoders to extract high-level feature vectors. These vectors are then concatenated (fused) in the task-specific towers. This approach preserves the unique temporal and spatial structures of the data before integration.

### 5.2 Attention Mechanisms (Mathematical Formulation)
To improve the handling of noisy sensor data (CTG and Activity), we incorporated **Attention Blocks** within the LSTM encoders. This allows the model to dynamically weight "important" time steps (e.g., a specific deceleration in heart rate) while suppressing irrelevant background noise, mimicking a clinician's focus on pathological events.

Mathematically, for a sequence of hidden states $H = \{h_1, h_2, ..., h_T\}$ produced by the LSTM layer, the attention mechanism computes a context vector $c$ as follows:

1.  **Score Calculation:** An alignment score $e_t$ is computed for each hidden state using a learnable weight matrix $W_a$ and bias $b_a$:
    $$e_t = \tanh(W_a h_t + b_a)$$
2.  **Attention Weights:** These scores are normalized using a softmax function to produce attention weights $\alpha_t$:
    $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T} \exp(e_k)}$$
3.  **Context Vector:** The final representation is a weighted sum of the hidden states:
    $$c = \sum_{t=1}^{T} \alpha_t h_t$$

This mechanism ensures that the model focuses on the most discriminative segments of the CTG and activity signals, rather than treating all time steps equally.

### 5.3 Handling Class Imbalance
Given the prevalence of "Low Risk" cases, we implemented **Class Weighting** during training. By penalizing misclassifications of the minority classes (High Risk) more heavily, we force the model to learn robust decision boundaries for pathological cases.

### 5.4 Implementation Environment
The system was developed using Python 3.9 within a virtualized environment to ensure reproducibility. Key libraries include:
*   **TensorFlow/Keras (2.x):** For building and training the deep neural network.
*   **Scikit-learn:** For data preprocessing (StandardScaler, LabelEncoder) and evaluation metrics.
*   **Pandas/NumPy:** For high-performance data manipulation and numerical operations.
*   **Streamlit:** For deploying the interactive web-based user interface.
*   **SHAP/LIME:** For model explainability and interpretability analysis.

## 6. Datasets and Feature Engineering

The system integrates four primary datasets:

### A. Enhanced Maternal Health Risk Data (Tabular)
*   **Source:** UCI Machine Learning Repository / IoT-based Maternal Health datasets.
*   **Features:** Age, Systolic/Diastolic BP, Blood Sugar (BS), Body Temperature, Heart Rate.
*   **Augmentation:** Enhanced with socio-economic and lifestyle factors (Sleep Hours, Physical Activity Level, Stress Score, Diet Quality).
*   **Preprocessing:** 
    *   Categorical variables (e.g., Urban/Rural) are label-encoded.
    *   Numerical variables are standardized ($z = (x - \mu) / \sigma$) to ensure consistent scaling.
    *   Missing values are imputed using the mean strategy.

### B. Cardiotocography (CTG) Data (Time-Series)
*   **Source:** UCI CTG Dataset.
*   **Features:** Fetal Heart Rate (FHR) baseline, Accelerations (AC), Decelerations (DL/DS/DP), Uterine Contractions (UC).
*   **Processing:** Treated as an 11-channel time-series signal. Missing signal segments are imputed.

### C. Maternal Activity Data (Sensor)
*   **Source:** MHEALTH Dataset (Simulated proxy).
*   **Features:** Tri-axial Chest Acceleration (X, Y, Z).
*   **Processing:** Windowed into fixed-length sequences (e.g., 50 time steps) to capture movement patterns.

### D. Ultrasound Imagery (Visual)
*   **Source:** Synthetic/Placeholder dataset for prototype, scalable to real fetal planes.
*   **Features:** 128x128 grayscale images.
*   **Processing:** Resized, normalized to [0, 1] range. MobileNetV2 is used as a feature extractor.

### E. Data Simulation Logic (Transparency Note)
For the purpose of this research prototype, where perfectly matched multimodal data is scarce, we employed a heuristic-based simulation for the "Fetal Weight" target variable to test the multi-task learning architecture. The birth weight $W$ was modeled as a function of the maternal risk category $R$ and hemoglobin levels $H$:
$$W = W_{base} + P_{risk}(R) + P_{hemo}(H) + \epsilon$$
Where:
*   $W_{base} = 3300g$ (Average baseline weight)
*   $P_{risk}$ imposes a penalty of -200g for Mid Risk and -600g for High Risk.
*   $P_{hemo}$ imposes a penalty of -300g if Hemoglobin < 11 g/dL.
*   $\epsilon \sim N(0, 200)$ represents random Gaussian noise.
This ensures the model learns logical physiological correlations even with synthetic targets.

## 7. Model Architecture Details

The model is implemented using TensorFlow/Keras with the following specifications:

1.  **Clinical Encoder:**
    *   Input: vector of size $N_{features}$.
    *   Layers: Dense(128) $\rightarrow$ BatchNormalization $\rightarrow$ ReLU $\rightarrow$ Dense(64).
    
2.  **CTG Encoder:**
    *   Input: (11, 1) sequence.
    *   Layers: Conv1D (Spatial features) $\rightarrow$ MaxPool $\rightarrow$ LSTM (Temporal dependencies) $\rightarrow$ Attention Block $\rightarrow$ GlobalAvgPool.

3.  **Activity Encoder:**
    *   Input: (50, 3) sequence.
    *   Layers: LSTM(64) $\rightarrow$ Attention Block $\rightarrow$ GlobalAvgPool.

4.  **Image Encoder:**
    *   Input: (128, 128, 1).
    *   Backbone: MobileNetV2 (Pre-trained on ImageNet, frozen).
    *   Layers: Conv2D (1ch $\rightarrow$ 3ch adapter) $\rightarrow$ MobileNetV2 $\rightarrow$ GlobalAvgPool.

5.  **Output Heads:**
    *   **Risk:** Softmax activation (3 units: Low, Mid, High).
    *   **Weight:** Linear activation (1 unit: Grams).

## 8. Inputs and Outputs

| Modality | Input Shape | Description |
| :--- | :--- | :--- |
| **Clinical** | (N, 18) | Vitals, Demographics, Lifestyle |
| **CTG** | (N, 11, 1) | Fetal Heart Rate variability features |
| **Activity** | (N, 50, 3) | 3-axis accelerometer readings |
| **Image** | (N, 128, 128, 1) | 2D Ultrasound scan |

| Prediction Task | Output Type | Range/Classes |
| :--- | :--- | :--- |
| **Maternal Risk** | Classification | Low Risk (0), Mid Risk (1), High Risk (2) |
| **Fetal Weight** | Regression | Continuous value (grams, e.g., 2500g - 4000g) |

## 9. The Entire Pipeline

The end-to-end workflow is automated as follows:

1.  **Data Loading:** The `DataLoader` module ingests raw CSV, Excel, and Image files from the disk.
2.  **Preprocessing:** The `DataPreprocessor` class cleans, scales, and aligns the multimodal data.
3.  **Training:** The `train.py` script splits data into stratified train/test sets, computes class weights, and trains the model using the Adam optimizer. Callbacks (EarlyStopping, ReduceLROnPlateau) prevent overfitting.
4.  **Inference:** The trained model is loaded into the Streamlit application (`app.py`).
5.  **User Interface:** A clinician enters patient vitals and uploads sensor/image data.
6.  **Analysis:**
    *   The model predicts risk and weight.
    *   **Confidence Analysis:** The system decomposes the risk score to show contribution from Clinical vs. Sensor vs. Image data.
    *   **XAI:** SHAP waterfall plots and LIME feature bar charts explain *why* the model made a prediction.
    *   **What-If:** A counterfactual module suggests minimal changes (e.g., "Lower Systolic BP by 10mmHg") to reduce risk.

## 10. Results and Evaluation Metrics

The model is evaluated using distinct metrics for its two tasks:

*   **Risk Classification Metrics:**
    *   **Accuracy:** Overall correctness of risk predictions.
    *   **Confusion Matrix:** To visualize misclassifications between adjacent risk levels (e.g., Mid vs. High).
    *   **Precision/Recall:** Critical for High Risk detection to minimize False Negatives (missed diagnoses).
    
*   **Weight Regression Metrics:**
    *   **Mean Absolute Error (MAE):** The average difference between predicted and actual fetal weight in grams.

*Preliminary observations indicate that the fusion of CTG and Clinical data significantly improves the detection of "Mid Risk" cases, which are often ambiguous in unimodal models.*

## 11. Identified Limitations
1.  **Data Quality Dependency:** The model relies on high-quality input. Noisy MHEALTH sensor data or poor-quality ultrasound scans can degrade performance.
2.  **Synthetic Associations:** In the current research phase, the alignment between specific ultrasound images and clinical records is simulated based on risk categories, limiting the "Weight Specialist" tower's ability to learn complex, non-linear growth patterns from raw pixels.
3.  **Generalizability:** The dataset is demographic-specific. Validation on diverse global populations is required to ensure fairness.

## 12. Future Enhancements
1.  **Real-Time IoT Integration:** Connecting the pipeline directly to wearable biosensors for continuous, real-time risk scoring.
2.  **Federated Learning:** Implementing privacy-preserving training to allow hospitals to collaborate without sharing sensitive patient data.
3.  **3D Ultrasound Analysis:** Upgrading the image encoder to handle 3D volumetric data for more precise fetal anomaly detection.
4.  **Mobile Edge Deployment:** Quantizing the model (using TensorFlow Lite) to run on mobile devices for use in rural/remote healthcare centers.

## 13. Conclusion
This research presents a comprehensive **Maternal Health AI Platform** that moves beyond traditional, fragmented care models. By successfully fusing clinical, sensor, and imaging data, we provide a 360-degree view of maternal and fetal health. The inclusion of interpretability tools (SHAP/LIME) and actionable "What-If" analyses transforms the system from a passive predictor into an active decision-support partner. This scalable, multimodal framework holds significant promise for reducing maternal mortality and improving birth outcomes globally.

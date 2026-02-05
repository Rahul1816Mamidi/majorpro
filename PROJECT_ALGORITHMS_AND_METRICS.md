# Algorithms, Metrics, and Technical Architecture

## 9. Algorithms Used in Each Stage

### Stage 1: Preprocessing & Data Cleaning
*   **Algorithm: Z-Score Normalization (StandardScaler)**
    *   **Technical Concept:** Transforms data to have a Mean ($\mu$) of 0 and Standard Deviation ($\sigma$) of 1.
    *   **Formula:** $z = \frac{x - \mu}{\sigma}$
    *   **Why used:** Neural networks train faster and more stably when inputs are centered around zero. It prevents large values (like "Systolic BP: 120") from dominating small values (like "Age: 25").
*   **Algorithm: One-Hot / Label Encoding**
    *   **Technical Concept:** Converts categorical text labels into numerical format.
    *   **Why used:** Computers cannot understand "Low Risk". We convert it to `0`.

### Stage 2: Feature Extraction (The "Encoders")
*   **Algorithm: Convolutional Neural Networks (CNN - 1D & 2D)**
    *   **Technical Concept:** A mathematical filter (kernel) slides over the input data performing matrix multiplication.
    *   **How it works:**
        *   **1D Conv (for CTG):** Slides over time. It learns patterns like "sudden drop" (Deceleration).
        *   **2D Conv (for Images):** Slides over pixels. It learns edges, curves, and textures.
    *   **Why used:** It captures *local dependencies* (neighboring pixels/seconds matter) and is *translation invariant* (a heart rate dip is bad regardless of when it happens).
*   **Algorithm: Long Short-Term Memory (LSTM)**
    *   **Technical Concept:** A Recurrent Neural Network (RNN) variant with a "Memory Cell" that can store information for long periods.
    *   **How it works:** It processes data sequentially. It has gates (Input, Forget, Output) to decide what to remember.
    *   **Why used:** Standard networks have no memory of the past. LSTM remembers "The heart rate was high 10 seconds ago", which is crucial for identifying trends in CTG.
*   **Algorithm: Attention Mechanism**
    *   **Technical Concept:** Computes a "Relevance Score" for each time step.
    *   **Formula:** $\alpha_t = \text{softmax}(\tanh(W h_t + b))$
    *   **Why used:** Not all seconds of a CTG trace are important. Attention allows the model to "zoom in" on the 5 seconds of abnormality and ignore the 15 minutes of silence.

### Stage 3: Training & Optimization
*   **Algorithm: Adam Optimizer (Adaptive Moment Estimation)**
    *   **Technical Concept:** It combines Momentum (moving faster in the right direction) with RMSProp (slowing down when the slope is changing wildly).
    *   **Why used:** It converges much faster than standard Stochastic Gradient Descent (SGD) and requires less tuning of the learning rate.
*   **Algorithm: Backpropagation**
    *   **Technical Concept:** The "Learning" phase. It calculates the error (Loss) and propagates it backward through the network to update weights using the Chain Rule of Calculus.

---

## 10. Visualization & Metrics Explanation

### A. Confusion Matrix
*   **What is it?** A grid showing the count of True vs. Predicted classes.
*   **Describes:** Where the model is getting confused.
    *   *Diagonal:* Correct predictions.
    *   *Off-Diagonal:* Errors (e.g., predicting "Low Risk" when it was actually "High Risk" - a dangerous False Negative).
*   **How we got it:** `sklearn.metrics.confusion_matrix(y_true, y_pred)`.

### B. ROC Curve (Receiver Operating Characteristic)
*   **What is it?** A plot of **Sensitivity (True Positive Rate)** vs. **1 - Specificity (False Positive Rate)**.
*   **Describes:** The trade-off. Can we detect *all* sick patients (High Sensitivity) without falsely alarming healthy ones?
*   **Key Metric:** **AUC (Area Under Curve)**. 1.0 = Perfect, 0.5 = Random Guessing.
*   **How we got it:** `sklearn.metrics.roc_curve`.

### C. Training History (Loss Curves)
*   **What is it?** A line graph of Error (Loss) vs. Time (Epochs).
*   **Describes:**
    *   *Blue Line (Train):* How well it learns the study material.
    *   *Orange Line (Validation):* How well it performs on a mock exam.
    *   *Gap:* If Train goes down but Validation goes up, it means **Overfitting**.

### D. Weight Scatterplot
*   **What is it?** A scatter plot with X-axis = Actual Weight, Y-axis = Predicted Weight.
*   **Describes:** Ideally, all dots should lie on the diagonal line ($y=x$). Dots far from the line indicate large errors in weight estimation.

### E. SHAP Summary Plot
*   **What is it?** A "Bee-swarm" plot.
*   **Describes:** Feature Importance.
    *   *Y-axis:* Features (e.g., Age, BP).
    *   *Color:* Red = High Value, Blue = Low Value.
    *   *X-axis:* Impact on Risk.
    *   *Example:* If "High BP" (Red dots) is on the far right, it means High BP pushes the risk UP.

---

## 11. Regression vs. Classification

### Classification (Maternal Risk)
*   **Goal:** Predict a **Category** (Discrete).
*   **Classes:** 0 (Low), 1 (Mid), 2 (High).
*   **Technical Implementation:**
    *   **Output Layer:** `Dense(3, activation='softmax')`.
    *   **Softmax:** Converts raw numbers (logits) into Probabilities summing to 1.0 (e.g., [0.1, 0.1, 0.8] $\to$ 80% High Risk).
    *   **Loss Function:** `SparseCategoricalCrossentropy`. It penalizes the model heavily if it assigns low probability to the correct class.

### Regression (Fetal Birth Weight)
*   **Goal:** Predict a **Quantity** (Continuous).
*   **Value:** A real number (e.g., 3250.5 grams).
*   **Technical Implementation:**
    *   **Output Layer:** `Dense(1, activation='linear')`.
    *   **Linear:** Returns the raw calculated value without squashing it.
    *   **Loss Function:** `MeanAbsoluteError` (MAE). It measures the average gap between prediction and reality (e.g., "Off by 50 grams").

---

## 12. Why Concatenation at Multiple Levels?
*   **The Strategy:** **Late Fusion**.
*   **Level 1 (Encoders):** We do **NOT** concatenate here. We let the CNN process images and the LSTM process signals *separately*.
    *   *Why?* Images are 2D arrays, Vitals are 1D vectors. Concatenating them raw would destroy the spatial structure of the image.
*   **Level 2 (Fusion Layer):** We concatenate the *extracted features*.
    *   *Why?* Now that the CNN has said "I see a head" (Feature Vector) and the Clinical model says "I see diabetes" (Feature Vector), we combine them.
*   **Level 3 (Twin Towers):** We split them again.
    *   *Why?* The "Risk" tower needs the *Activity* data (for heart rate context), but the "Weight" tower ignores Activity and focuses on *Image* data. This selective concatenation prevents noise.

---

## 13. Technology Stack Choices

### Why TensorFlow / Keras?
1.  **Functional API:** We need a non-linear topology (4 Inputs $\to$ Split $\to$ Merge $\to$ Split $\to$ 2 Outputs). Keras Functional API (`Model(inputs=..., outputs=...)`) is the standard industry tool for this. Sequential models (like in simple tutorials) cannot do this.
2.  **Ecosystem:** It has built-in support for TFLite (Mobile deployment) and TF.js (Web deployment), making it "Production Ready".
3.  **Differentiable Layers:** It provides the complex derivatives needed for Backpropagation through LSTM and CNN layers automatically.

### Why Streamlit?
1.  **Pure Python:** No need to learn HTML/CSS/JavaScript. Data Scientists can build UIs directly in Python.
2.  **Rapid Prototyping:** We can add a slider or a file uploader with one line of code (`st.slider`, `st.file_uploader`).
3.  **Data Native:** It has built-in support for displaying Pandas DataFrames (`st.dataframe`) and Matplotlib/SHAP charts (`st.pyplot`), which is exactly what an ML dashboard needs.

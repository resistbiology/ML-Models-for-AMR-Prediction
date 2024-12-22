**Multi-Label Classification with XGBoost**

This project demonstrates a multi-label classification pipeline using XGBoost and MultiOutputClassifier for predicting antimicrobial resistance (AMR) based on gene presence.

---

**Features**

- Multi-label binarization of genes and resistance phenotypes using MultiLabelBinarizer.
- Multi-label classification using XGBoost wrapped in MultiOutputClassifier.
- Model evaluation using:
  - Accuracy
  - Hamming Loss
  - Classification Report
- Cross-validation to estimate model performance.
- Model saving/loading using joblib.
- Model interpretation using SHAP (SHapley Additive exPlanations).

---

**Setup Instructions**

**Prerequisites**

Install the required Python libraries:

```
pip install pandas scikit-learn xgboost joblib shap matplotlib
```

**Dataset Preparation**

The dataset should be in Excel format with the following structure:

- **Sheet Name:** Sheet1
- **Columns:**
  - **GENE**: A semicolon-separated list of gene names.
  - **RESISTANCE**: A semicolon-separated list of resistance phenotypes.

For example:

| GENE               | RESISTANCE          |
|--------------------|---------------------|
| aac(3)-IId;blaTEM  | Amikacin;Ampicillin |

---

**Workflow**

1. **Load Data**

   Use the `load_data` function to load data from an Excel file.

   ```python
   amr_abricate_df = load_data(file_path='path_to_your_dataset.xlsx')
   ```

2. **Transform Data**

   The `transform_data` function binarizes the GENE and RESISTANCE columns using MultiLabelBinarizer.

   ```python
   X, y, mlb_gene, mlb_resistance = transform_data(amr_abricate_df)
   ```

   - `X`: Binarized gene features.
   - `y`: Binarized resistance labels.
   - `mlb_gene` and `mlb_resistance`: Encoders for genes and resistance phenotypes.

3. **Split Data**

   Split the dataset into training and test sets using `train_test_split`:

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

4. **Train Model**

   Train an XGBoost-based multi-label classifier with the `train_model` function:

   ```python
   model = train_model(X_train, y_train, learning_rate=0.1, n_estimators=100)
   ```

5. **Evaluate Model**

   Evaluate the model using accuracy, Hamming loss, and a classification report:

   ```python
   accuracy, hamming_loss_value, report = evaluate_model(model, X_test, y_test)
   print(f"Accuracy: {accuracy}")
   print(f"Hamming Loss: {hamming_loss_value}")
   print("Classification Report:\n", report)
   ```

6. **Save and Load Model**

   Save the trained model using:

   ```python
   save_model(model, 'XGBoost_model.pkl')
   ```

   Load the saved model using:

   ```python
   model = load_model('XGBoost_model.pkl')
   ```

7. **Predict Resistance**

   Predict resistance phenotypes for a new sample of genes:

   ```python
   new_sample_genes = ['aac(3)-IId']
   predicted_resistance = predict_resistance(model, mlb_gene, mlb_resistance, new_sample_genes)
   print(f"Predicted Resistance: {predicted_resistance}")
   ```

8. **Explain Predictions**

   Use SHAP to explain the model's prediction for a specific resistance type:

   ```python
   explain_model_prediction(
       model=model,
       X_train=X_train,
       X_sample=pd.DataFrame(mlb_gene.transform([new_sample_genes]), columns=mlb_gene.classes_),
       feature_names=mlb_gene.classes_,
       resistance_type="Amikacin",
       mlb_resistance=mlb_resistance
   )
   ```

   This generates a SHAP waterfall plot saved as `shap_summary.svg`.

9. **Cross-Validation**

   Perform multi-label cross-validation using `multi_label_cross_validation`:

   ```python
   cv_accuracies, cv_hamming_losses_values = multi_label_cross_validation(X, y, model)
   print(f"Cross-Validation Accuracies: {cv_accuracies}")
   print(f"Average Accuracy: {np.mean(cv_accuracies)}")
   print(f"Cross-Validation Hamming Losses: {cv_hamming_losses_values}")
   print(f"Average Hamming Loss: {np.mean(cv_hamming_losses_values)}")
   ```

---

**Key Functions**

1. **`load_data(file_path)`**:
   - Loads Excel data from the specified path.

2. **`transform_data(amr_abricate_df)`**:
   - Binarizes the GENE and RESISTANCE columns.

3. **`train_model(X_train, y_train)`**:
   - Trains a multi-label classifier using XGBoost.

4. **`evaluate_model(model, X_test, y_test)`**:
   - Evaluates the model using accuracy, Hamming loss, and a classification report.

5. **`save_model(model, filename)`**:
   - Saves the model to a file.

6. **`load_model(filename)`**:
   - Loads a saved model.

7. **`predict_resistance(model, mlb_gene, mlb_resistance, gene_list)`**:
   - Predicts resistance phenotypes for a given gene list.

8. **`explain_model_prediction(...)`**:
   - Explains model predictions using SHAP.

9. **`multi_label_cross_validation(X, y, model)`**:
   - Performs cross-validation for multi-label classification.

---

**Example Outputs**

- **Accuracy**: 0.85
- **Hamming Loss**: 0.12
- **Predicted Resistance**: [['Amikacin']]
- **SHAP Explanation**: Saved as `shap_summary.svg`

---

**Notes**

- Ensure the dataset matches the required structure for successful processing.
- SHAP analysis requires a valid `feature_names` parameter.

---

**CNN-Based Multi-Label Classification for Antimicrobial Resistance**

This project demonstrates a multi-label classification pipeline using a Convolutional Neural Network (CNN) for predicting antimicrobial resistance (AMR) based on encoded gene data.

---

**Features**

- Data preprocessing using OneHotEncoder and MultiLabelBinarizer.
- Multi-label classification with a custom CNN model.
- Model evaluation using:
  - Accuracy
  - Classification Report
- Resampling to balance the dataset.
- Model saving/loading using TensorFlow/Keras.
- Visualization of training and validation accuracy/loss.
- Learning rate adjustment using ReduceLROnPlateau.

---

**Setup Instructions**

**Prerequisites**

Install the required Python libraries:

```
pip install pandas scikit-learn tensorflow matplotlib openpyxl
```

---

**Workflow**

1. **Load and Preprocess Data**

   Use the `load_and_preprocess_data` function to encode genes and resistance phenotypes:

   ```python
   gene_encoded, resistance_encoded, onehot_encoder, antibiotic_names = load_and_preprocess_data(file_path='path_to_your_dataset.xlsx')
   ```

   - `gene_encoded`: Encoded features for genes.
   - `resistance_encoded`: Encoded labels for resistance phenotypes.
   - `onehot_encoder`: Fitted encoder for gene data.
   - `antibiotic_names`: List of antibiotics (classes).

2. **Reshape Data for CNN**

   Reshape the data for compatibility with the CNN model:

   ```python
   X_reshaped = gene_encoded.reshape((gene_encoded.shape[0], gene_encoded.shape[1], 1))
   ```

3. **Split Data**

   Split the dataset into training and test sets using `train_test_split`:

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X_reshaped, resistance_encoded, test_size=0.2, random_state=42)
   ```

4. **Balance Dataset (Optional)**

   If resampling is required, flatten and balance the data before reshaping back:

   ```python
   X_train_flattened = X_train.reshape((X_train.shape[0], -1))
   X_train_resampled_df = pd.DataFrame(X_train_flattened)
   y_train_resampled_df = pd.DataFrame(y_train)

   X_train_resampled = X_train_resampled_df.to_numpy().reshape((-1, X_train.shape[1], 1))
   y_train_resampled = y_train_resampled_df.to_numpy()
   ```

5. **Create CNN Model**

   Use the `create_cnn_model` function to define the CNN architecture:

   ```python
   input_shape = X_train_resampled.shape[1:]
   num_classes = y_train_resampled.shape[1]
   cnn_model = create_cnn_model(input_shape, num_classes)
   ```

6. **Train the Model**

   Train the CNN model with `ReduceLROnPlateau` to adjust the learning rate during training:

   ```python
   reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
   history = cnn_model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=64, validation_split=0.2, callbacks=[reduce_lr])
   ```

7. **Visualize Training History**

   Plot training and validation accuracy and loss:

   ```python
   plt.plot(history.history['accuracy'])
   plt.plot(history.history['val_accuracy'])
   plt.title('Model accuracy')
   plt.ylabel('Accuracy')
   plt.xlabel('Epoch')
   plt.legend(['Train', 'Test'], loc='upper left')
   plt.show()

   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.title('Model loss')
   plt.ylabel('Loss')
   plt.xlabel('Epoch')
   plt.legend(['Train', 'Test'], loc='upper left')
   plt.show()
   ```

8. **Evaluate the Model**

   Predict and evaluate on the test set:

   ```python
   y_pred = cnn_model.predict(X_test)
   y_pred_binary = (y_pred > 0.5).astype(int)
   print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
   print(f"Classification Report:\n{classification_report(y_test, y_pred_binary)}")
   ```

9. **Save the Model**

   Save the trained model:

   ```python
   cnn_model.save('CNN_without_resampling.h5')
   print("Model saved at CNN_without_resampling.h5")
   ```

---

**Key Functions**

1. **`load_and_preprocess_data(file_path)`**
   - Loads and encodes gene and resistance data.
   - Returns encoded features, labels, and metadata.

2. **`create_cnn_model(input_shape, num_classes)`**
   - Creates and compiles a CNN model for multi-label classification.

3. **`ReduceLROnPlateau`**
   - Adjusts the learning rate dynamically based on validation loss.

4. **`cnn_model.fit()`**
   - Trains the CNN on the resampled training data.

5. **`cnn_model.predict(X_test)`**
   - Predicts resistance phenotypes for the test set.

6. **`cnn_model.save(file_path)`**
   - Saves the trained model to the specified file path.

---

**Example Outputs**

- **Training Accuracy**: 0.92
- **Validation Accuracy**: 0.88
- **Test Accuracy**: 0.85
- **Classification Report**:
  ```
  precision    recall  f1-score   support
  ...
  ```

---

**Notes**

- Ensure the dataset matches the required structure with gene and resistance columns.
- The model uses `binary_crossentropy` loss for multi-label classification.
- SHAP analysis can be added for interpretability if needed.

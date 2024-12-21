import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
import joblib
import shap
import matplotlib.pyplot as plt

def load_data(file_path):
    amr_abricate_df = pd.read_excel(file_path, sheet_name='Sheet1')
    return amr_abricate_df

def transform_data(amr_abricate_df):
    gene_list = amr_abricate_df['GENE'].str.split(';').apply(lambda x: x if isinstance(x, list) else [])
    resistance_list = amr_abricate_df['RESISTANCE'].str.split(';').apply(lambda x: x if isinstance(x, list) else [])

    mlb_gene = MultiLabelBinarizer()
    gene_binary = mlb_gene.fit_transform(gene_list)
    gene_binary_df = pd.DataFrame(gene_binary, columns=mlb_gene.classes_)

    print(gene_binary_df)

    mlb_resistance = MultiLabelBinarizer()
    resistance_binary = mlb_resistance.fit_transform(resistance_list)
    resistance_binary_df = pd.DataFrame(resistance_binary, columns=mlb_resistance.classes_)
    print(resistance_binary_df)
    return gene_binary_df, resistance_binary_df, mlb_gene, mlb_resistance


from sklearn.utils import shuffle
import numpy as np

def train_model(X_train, y_train, learning_rate=0.1, n_estimators=100):
    xgb_model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, eval_metric='logloss', use_label_encoder=False)
    multi_label_model = MultiOutputClassifier(xgb_model)
    multi_label_model.fit(X_train, y_train)
    return multi_label_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    hamming_loss_value = hamming_loss(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, hamming_loss_value, report

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def predict_resistance(model, mlb_gene, mlb_resistance, gene_list):
    new_sample_binary = mlb_gene.transform([gene_list])
    new_sample_df = pd.DataFrame(new_sample_binary, columns=mlb_gene.classes_)
    new_sample_prediction = model.predict(new_sample_df)
    predicted_antibiotics = mlb_resistance.inverse_transform(new_sample_prediction)
    return predicted_antibiotics

def explain_model_prediction(model, X_train, X_sample, feature_names, resistance_type, mlb_resistance):
    label_index = list(mlb_resistance.classes_).index(resistance_type)
    estimator = model.estimators_[label_index]
    explainer = shap.Explainer(estimator, X_train.iloc[:100], feature_names=feature_names)
    shap_values = explainer(X_sample)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig("shap_summary.svg", dpi=700)
    plt.show()

def multi_label_cross_validation(X, y, model, n_splits=5):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    hamming_losses_values = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_resampled, y_train_resampled = X_train, y_train
        fitted_model = model.fit(X_train_resampled, y_train_resampled)
        y_pred = fitted_model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        hamming_losses_values.append(hamming_loss(y_test, y_pred))
    return accuracies, hamming_losses_values

# Main execution
file_path = '../Salmonella_HK_ASEAN_AMR_20231114.xlsx'
amr_abricate_df = load_data(file_path)
X, y, mlb_gene, mlb_resistance = transform_data(amr_abricate_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply custom resampling for balancing the dataset
X_train_resampled, y_train_resampled = X_train, y_train

multi_label_model = train_model(X_train_resampled, y_train_resampled)
accuracy, hamming_loss_value, report = evaluate_model(multi_label_model, X_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Hamming Loss: {hamming_loss_value}")
print("Classification Report:\n", report)

model_filename = 'XGBoost_without_resampling.pkl'
save_model(multi_label_model, model_filename)

# Predicting with a new sample
new_sample_genes = ['aac(3)-IId']
predicted_resistance = predict_resistance(multi_label_model, mlb_gene, mlb_resistance, new_sample_genes)
print(f"Predicted Resistance: {predicted_resistance}")

# Explain the model's prediction for this sample
explain_model_prediction(multi_label_model, X_train, pd.DataFrame(mlb_gene.transform([new_sample_genes]), columns=mlb_gene.classes_), mlb_gene.classes_, "Amikacin", mlb_resistance)

# Cross-Validation
cv_accuracies, cv_hamming_losses_values = multi_label_cross_validation(X, y, multi_label_model)
print(f"Cross-Validation Accuracies: {cv_accuracies}")
print(f"Average Accuracy: {np.mean(cv_accuracies)}")
print(f"Cross-Validation Hamming Losses: {cv_hamming_losses_values}")
print(f"Average Hamming Loss: {np.mean(cv_hamming_losses_values)}")

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    try:
        amr_df = pd.read_excel(file_path, sheet_name='Sheet1')
        onehot_encoder = OneHotEncoder(sparse=False)
        gene_encoded = onehot_encoder.fit_transform(amr_df[['GENE']].fillna('Unknown'))
        mlb = MultiLabelBinarizer()
        resistance_list = amr_df['RESISTANCE'].str.split(';').apply(lambda x: x if isinstance(x, list) else [])
        resistance_encoded = mlb.fit_transform(resistance_list)

        # Print or return the actual names of the antibiotics
        antibiotic_names = mlb.classes_
        print("Antibiotic names:", antibiotic_names)

        return gene_encoded, resistance_encoded, onehot_encoder, antibiotic_names
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {e}")
        raise


# Function to create CNN model
def create_cnn_model(input_shape, num_classes):
    try:
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(num_classes, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        logger.error(f"Error in create_cnn_model: {e}")
        raise

from sklearn.utils import shuffle
import numpy as np



# Main execution flow
def main():
    try:
        file_path = '../Salmonella_HK_ASEAN_AMR_20231114.xlsx'
        X, y, onehot_encoder, classes = load_and_preprocess_data(file_path)
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)


        # Flatten X_train for resampling
        num_samples, num_timesteps, num_features = X_train.shape
        X_train_flattened = X_train.reshape((num_samples, num_timesteps * num_features))

        # Convert X_train and y_train to DataFrame for resampling
        X_train_df = pd.DataFrame(X_train_flattened)
        y_train_df = pd.DataFrame(y_train)

        # Apply custom resampling to balance the dataset
        X_train_resampled_df, y_train_resampled_df = X_train_df, y_train_df
        print(X_train_resampled_df)
        print(y_train_resampled_df)

        file_path = 'encoded gene.csv'

        X_train_resampled_df.to_csv(file_path, sep=',', index=False)

        file_path = 'encoded resistance.csv'

        y_train_resampled_df.to_csv(file_path, sep=',', index=False)


        # Reshape X_train back to 3D for CNN
        X_train_resampled = X_train_resampled_df.to_numpy().reshape((-1, num_timesteps, num_features))
        y_train_resampled = y_train_resampled_df.to_numpy()



        input_shape = X_train_resampled.shape[1:]
        num_classes = y_train_resampled.shape[1]
        cnn_model = create_cnn_model(input_shape, num_classes)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
        history = cnn_model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=64, validation_split=0.2, callbacks=[reduce_lr])

        # Plot training history
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

        # Evaluate the model
        y_pred = cnn_model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        logger.info(f'Accuracy: {accuracy_score(y_test, y_pred_binary)}')
        logger.info(f'Classification Report:\n{classification_report(y_test, y_pred_binary)}')

        # Save the model
        model_save_path = 'CNN_without_resampling.h5'
        cnn_model.save(model_save_path)
        logger.info(f"Model saved at {model_save_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()

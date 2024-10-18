# Transformer-Assisted Bug Detection and Diagnosis in Deep Neural Networks

## Project Description

This project focuses on automating the detection and diagnosis of bugs in deep neural networks (DNNs) using transformer models for feature extraction and deep learning models for classification. Our approach leverages powerful transformer models like PLBART, CodeBERT, and GraphCodeBERT to extract semantic features from code, and then uses a CNN-LSTM model for accurate bug detection and classification.

## Repository Structure

- `README.md`: This file.
- [data](https://github.com/AyanT01/REU-2024/tree/origin/data): Directory containing the dataset (not included in the repository).

## Prerequisites

- [Kaggle notebook for binary classification](https://www.kaggle.com/code/abdulayantayo/binary-classification)
- [Kaggle notebook for categorical classification](https://www.kaggle.com/code/abdulayantayo/categorical-classification)

### Running the Model

All the necessary code is available in the Kaggle notebook. To run the model:

1. **Load and Use the Pre-trained Model**

   ```python
   from tensorflow.keras.models import load_model

   model_path = "cnn_lstm_best_model.keras"
   classification_model = load_model(model_path)
2. **Define your code snippets**
```
code_snippets = [
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    """,
    # Add more code snippets as needed
]
# Extract features using PLBART
features = np.array([extract_features(plbart_model, plbart_tokenizer, snippet) for snippet in code_snippets])
features = np.expand_dims(features, axis=-1)
```
# Binary Classification
```
predictions = classification_model.predict(features)
predicted_labels = (predictions > 0.5).astype(int)

for snippet, label in zip(code_snippets, predicted_labels):
    print(f"Code Snippet: {snippet}\nBug Detected: {'Yes' if label else 'No'}\n")
```

# Categorical Classification
```
predictions = classification_model.predict(features)
predicted_labels = np.argmax(predictions, axis=1)

for snippet, label in zip(code_snippets, predicted_labels):
    print(f"Code Snippet: {snippet}\nBug Type: {label}\n")
```

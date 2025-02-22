# Protodeep

Protodeep is a lightweight, educational deep learning library based on numpy and numba

## Overview

Protodeep provides the essential building blocks required for constructing, training, and evaluating neural networks. It supports a variety of layers, activation functions, loss functions, optimizers, and metrics – all wrapped in a modular, easy-to-use design.

Protodeep also includes helpful utilities such as callbacks (e.g., early stopping) and debugging tools (like a timer class decorator) to streamline model development and experimentation.

## Features

- **Layers**  
  - Convolutional `Conv2D`
  - Dense `Dense`  
  - Long Short Time Memory `LSTM`
  - Utility layers: `Flatten`, `Input`, `MaxPool2D`  
  - Base layer abstraction: `Layer`

- **Activations**  
  - Common functions: `Linear`, `Relu`, `Sigmoid`, `Softmax`, `Tanh`  
  - Base activation class: `Activation`

- **Callbacks**   
  - Early stopping for efficient training: `EarlyStopping`

- **Initializers**  
  - Weight initialization strategies: `GlorotNormal`, `GlorotUniform`, `HeNormal`, `RandomNormal`, `Zeros`

- **Losses**  
  - Loss functions for regression and classification: `BinaryCrossentropy`, `MeanSquaredError`

- **Metrics**  
  - Evaluation metrics: `Accuracy`, `CategoricalAccuracy`, `BinaryAccuracy`

- **Optimizers**  
  - Gradient descent variants: `Adagrad`, `Adam`, `RMSProp`, `SGD`

- **Regularizers**  
  - Regularization techniques: `L1`, `L2`, `L1L2`

- **Model Architecture**  
  - Sequential model class: `Model`

- **Debugging**  
  - Timer class decorator for performance monitoring

- **Unified API**  
  - Aggregates all components (activations, callbacks, initializers, layers, losses, metrics, model, optimizers, regularizers) into one easy-to-use package

## Installation

### Prerequisites
- Python 3.x
- pip (Python package installer)
- Git LFS (Large File Storage)

### Steps
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/protodeep.git
    cd protodeep
    ```

2. Install Git LFS and pull LFS files:
    ```sh
    git lfs install
    git lfs pull
    ```

3. Setup the project and install dependencies:
    ```sh
    make setup
    ```
## Basic Usage

For a simple demo you can run
```sh
make example
```

Below is an example demonstrating how to parse options, load a dataset, build a sequential model, train it, and visualize the results:

```python

# Load dataset
dataset = Dataset('Examples/data.csv', 0.2)

# Build the model
model = Model()
model.add(Protodeep.layers.Dense(64, activation=Protodeep.activations.Relu()))
model.add(Protodeep.layers.Dense(32, activation=Protodeep.activations.Relu()))
model.add(Protodeep.layers.Dense(2, activation=Protodeep.activations.Sigmoid()))
model.compile(
    epochs=30,
    metrics=[Protodeep.metrics.CategoricalAccuracy(), Protodeep.metrics.Accuracy()],
    optimizer=Protodeep.optimizers.Adam()
)
model.summary()

# Display dataset shapes
print(dataset.features.shape)
print(dataset.test_features.shape)

# Train the model
history = model.fit(
    features=dataset.features,
    targets=dataset.targets,
    epochs=500,
    batch_size=32,
    validation_data=(dataset.test_features, dataset.test_targets),
    callbacks=[EarlyStopping(monitor="val_loss", patience=10)]
)

# Evaluate the model
model.evaluate(validation_data=(dataset.test_features, dataset.test_targets))

# Plot performance metrics
import matplotlib.pyplot as plt

plt.plot(history['categorical_accuracy'])
plt.plot(history['val_categorical_accuracy'])
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend([
    'Train Categorical Accuracy',
    'Validation Categorical Accuracy',
    'Train Accuracy',
    'Validation Accuracy'], loc='lower right')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
plt.show()
```

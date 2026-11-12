# CIFAR-10 Image Classification Using Custom CNN

## 1. Project Overview

This project implements an image classification system using a custom Convolutional Neural Network (CNN) developed with TensorFlow and Keras.

The model is trained on the CIFAR-10 dataset to classify color images into 10 different categories. The project covers dataset loading, image preprocessing, normalization, data augmentation, CNN architecture development, model training, evaluation, and deployment through an interactive Gradio interface.

The model was trained for **30 epochs** and achieved **80.50% test accuracy**.

## 2. Dataset Description

The CIFAR-10 dataset contains 60,000 color images of size 32 × 32 pixels, distributed across 10 classes.

- Total images: 60,000
- Training images: 50,000
- Testing images: 10,000
- Training split: 45,000 images
- Validation split: 5,000 images
- Image dimensions: 32 × 32 × 3
- Number of classes: 10

### CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The original CIFAR-10 Python archive is stored in the project's `data/` directory.

## 3. Technologies and Libraries

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Pillow
- Gradio
- Google Colab
- Jupyter Notebook

## 4. Project Structure

```text
cifar10/
│
├── data/
│   └── cifar-10-python.tar.gz
│
├── model/
│   └── cifar10modal.keras
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── training_and_validation_loss.png
│   └── training_and_validation_accuracy.png
│
├── src/
│   └── cifar_10_updated.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Directory Description

| File/Folder | Description |
|---|---|
| `data/` | Contains the original CIFAR-10 compressed dataset. |
| `model/` | Contains the trained CNN model in Keras format. |
| `outputs/` | Contains the confusion matrix and training/validation performance graphs. |
| `src/` | Contains the Jupyter Notebook with the complete implementation. |
| `requirements.txt` | Lists the required Python libraries. |
| `README.md` | Provides project documentation and execution instructions. |
| `.gitignore` | Specifies files and folders excluded from Git tracking. |

## 5. Data Loading and Preprocessing

The CIFAR-10 dataset was loaded from its Python batch files after extracting the downloaded archive.

The following preprocessing steps were performed:

1. Extracted the CIFAR-10 dataset archive.
2. Loaded the training and testing batches using Python Pickle.
3. Reshaped the images into 32 × 32 × 3 format.
4. Displayed 10 randomly selected images with their corresponding class labels.
5. Divided the original training dataset into training and validation sets.
6. Converted image data into floating-point format.
7. Calculated the mean and standard deviation using the training dataset.
8. Normalized training, validation, and testing images using the training statistics.
9. Converted class labels into one-hot encoded format.

### Dataset Split

| Dataset | Number of Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |

## 6. Data Augmentation

Data augmentation was applied to increase image variation and improve the model's ability to generalize to unseen images.

The following techniques were used:

- Rotation: 15 degrees
- Width shifting: 0.12
- Height shifting: 0.12
- Horizontal flipping
- Zoom: 0.1
- Brightness adjustment: 0.9–1.1
- Shear transformation
- Channel shifting

These transformations help the model learn different image variations and reduce overfitting.

## 7. CNN Model Architecture

A custom CNN architecture consisting of four convolutional blocks was developed.

### Convolutional Blocks

| Block | Convolutional Filters | Dropout Rate |
|---|---|---:|
| Block 1 | 32, 32 | 0.2 |
| Block 2 | 64, 64 | 0.3 |
| Block 3 | 128, 128 | 0.4 |
| Block 4 | 256, 256 | 0.5 |

Each convolutional block contains:

- Conv2D layers for feature extraction
- Batch Normalization for stable training
- ReLU activation for non-linearity
- Max Pooling for spatial downsampling
- Dropout for reducing overfitting

### Classification Layer

After the four convolutional blocks:

- Flatten converts feature maps into a one-dimensional vector.
- Dense layer with 10 neurons performs classification.
- Softmax activation generates class probabilities for the 10 CIFAR-10 categories.

L2 regularization was applied to convolutional layers to help control overfitting.

## 8. Model Training Configuration

The model was trained using TensorFlow and Keras with the following configuration:

| Parameter | Value |
|---|---|
| Model | Custom CNN |
| Input Shape | 32 × 32 × 3 |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Loss Function | Categorical Crossentropy |
| Batch Size | 64 |
| Number of Epochs | **30** |
| Weight Decay | 0.0001 |
| Output Classes | 10 |

### Training Callbacks

Three callbacks were implemented to improve the training process.

**1. ModelCheckpoint**

- Monitored validation accuracy.
- Saved the model with the best validation accuracy.
- Used `save_best_only=True`.

**2. ReduceLROnPlateau**

- Monitored validation loss.
- Reduced the learning rate when validation loss stopped improving.
- Reduction factor: 0.5.
- Patience: 4 epochs.
- Minimum learning rate: 0.00001.

**3. EarlyStopping**

- Monitored validation loss.
- Patience: 8 epochs.
- Restored the best model weights.

The training run completed 30 epochs. The highest validation accuracy was recorded at epoch 27.

## 9. Model Evaluation and Results

After training, the best saved model was evaluated on the separate CIFAR-10 test dataset containing 10,000 images.

### Performance Results

| Metric | Result |
|---|---:|
| Final Training Accuracy | 76.56% |
| Final Validation Accuracy | 79.72% |
| Best Validation Accuracy | 81.72% |
| Best Validation Epoch | 27 |
| Test Accuracy | **80.50%** |
| Test Loss | 0.6802 |

### Evaluation Methods

The model was evaluated using:

- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrix
- Classification report
- Test dataset evaluation

The generated training graphs and confusion matrix are stored in the `outputs/` directory.

## 10. Model Saving

The trained CNN model was saved in Keras format:

```text
model/cifar10modal.keras
```

The notebook also saves preprocessing statistics, including the training dataset's mean and standard deviation, for consistent normalization during prediction.

These statistics are required when using the saved model outside the original notebook.

## 11. Gradio Image Classification Interface

An interactive image classification interface was developed using Gradio inside the Jupyter Notebook.

### Prediction Workflow

1. The user uploads an image.
2. The image is converted into RGB format.
3. The image is resized to 32 × 32 pixels.
4. The image is converted into a NumPy array.
5. The same normalization used during training is applied.
6. The trained CNN predicts the image category.
7. The interface displays the predicted class and confidence scores for all 10 classes.

The Gradio interface is launched using:

```python
demo.launch(share=True)
```

This generates a shareable link through which users can test the image classification system.

## 12. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
cd cifar10
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Open the Jupyter Notebook

```bash
jupyter notebook src/cifar_10_updated.ipynb
```

Run the notebook cells to perform dataset loading, preprocessing, model training, evaluation, and Gradio interface execution.

**Note:** Model training was performed in Google Colab using a T4 GPU. The saved model can be reused for predictions without repeating the training process. The preprocessing statistics must also be available to reproduce the same input normalization.

## 13. Conclusion

This project demonstrates the development of a custom CNN-based image classification system using TensorFlow and Keras.

It combines image preprocessing, normalization, data augmentation, convolutional neural networks, batch normalization, dropout, and L2 regularization to classify images into 10 CIFAR-10 categories.

The model was trained for **30 epochs** and achieved **80.50% test accuracy**. An interactive Gradio interface was also implemented to allow users to upload images and view classification predictions.

The project demonstrates the complete workflow of building, training, evaluating, and testing a deep learning image classification model.
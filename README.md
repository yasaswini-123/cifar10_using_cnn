# CIFAR-10 Image Classification Using Custom CNN

## 1. Project Overview

This project implements an image classification system using a custom Convolutional Neural Network (CNN) developed with TensorFlow and Keras.

The model classifies images from the CIFAR-10 dataset into 10 different categories. The project covers dataset loading, preprocessing, normalization, data augmentation, CNN architecture development, model training, evaluation, and an interactive image prediction interface using Gradio.

The CNN was trained for **30 epochs** and achieved **80.50% test accuracy**.

## 2. Dataset Description

The CIFAR-10 dataset consists of 60,000 color images, each with a resolution of 32 × 32 pixels, divided into 10 classes.

| Dataset Details | Value |
|---|---:|
| Total Images | 60,000 |
| Original Training Images | 50,000 |
| Testing Images | 10,000 |
| Training Split | 45,000 |
| Validation Split | 5,000 |
| Image Dimensions | 32 × 32 × 3 |
| Number of Classes | 10 |

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

The original CIFAR-10 Python dataset archive is stored in the `data/` directory.

## 3. Technologies Used

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
cifar10_using_cnn/
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
| `outputs/` | Stores the confusion matrix and training/validation graphs. |
| `src/` | Contains the Jupyter Notebook with the complete implementation. |
| `requirements.txt` | Lists the required Python libraries. |
| `README.md` | Contains project documentation and execution instructions. |
| `.gitignore` | Specifies files and folders excluded from Git tracking. |

## 5. Data Loading and Preprocessing

The CIFAR-10 dataset was loaded from its extracted Python batch files using Python Pickle.

The following preprocessing operations were performed:

1. Uploaded and extracted the CIFAR-10 dataset archive.
2. Loaded the five training batches and the separate test batch.
3. Reshaped the images into 32 × 32 × 3 format.
4. Displayed 10 randomly selected images with their class labels.
5. Divided the original training dataset into training and validation sets.
6. Converted image values into floating-point format.
7. Calculated the mean and standard deviation from the training images.
8. Normalized the training, validation, and testing images.
9. Converted class labels into one-hot encoded format.

### Dataset Split

| Dataset | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |

## 6. Data Augmentation

Data augmentation was applied to improve the model's ability to generalize to unseen images and reduce overfitting.

The following techniques were used:

- Rotation: 15 degrees
- Width shifting: 0.12
- Height shifting: 0.12
- Horizontal flipping
- Zoom: 0.1
- Brightness adjustment: 0.9–1.1
- Shear transformation
- Channel shifting

These transformations introduce variations into training images so that the CNN can learn more robust image features.

## 7. CNN Model Architecture

A custom CNN architecture consisting of four convolutional blocks was developed.

### Convolutional Blocks

| Block | Convolutional Filters | Dropout Rate |
|---|---|---:|
| Block 1 | 32, 32 | 0.2 |
| Block 2 | 64, 64 | 0.3 |
| Block 3 | 128, 128 | 0.4 |
| Block 4 | 256, 256 | 0.5 |

Each convolutional block consists of:

- Conv2D layers for feature extraction.
- Batch Normalization for stable training.
- ReLU activation for introducing non-linearity.
- Max Pooling for spatial downsampling.
- Dropout for reducing overfitting.

### Final Classification Layer

After the convolutional blocks:

- Flatten converts feature maps into a one-dimensional vector.
- Dense layer with 10 neurons performs classification.
- Softmax activation generates probabilities for the 10 CIFAR-10 classes.

L2 regularization was applied to convolutional layers to help control overfitting.

## 8. Model Training

The CNN model was trained using TensorFlow and Keras.

### Training Configuration

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

Three callbacks were used during model training.

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

The best saved model was evaluated using the separate CIFAR-10 test dataset containing 10,000 images.

### Performance Results

| Metric | Result |
|---|---:|
| Final Training Accuracy | 76.56% |
| Final Validation Accuracy | 79.72% |
| Best Validation Accuracy | 81.72% |
| Best Validation Epoch | 27 |
| Test Accuracy | **80.50%** |
| Test Loss | 0.6802 |

### Evaluation Techniques

The following evaluation methods were used:

- Training and validation accuracy curves.
- Training and validation loss curves.
- Confusion matrix.
- Classification report.
- Test dataset evaluation.

The generated performance graphs and confusion matrix are available in the `outputs/` directory.

## 10. Model Saving

The trained CNN model was saved in Keras format:

```text
model/cifar10modal.keras
```

The notebook also saves the preprocessing statistics (mean and standard deviation) in:

```text
cifar10_preprocessing.npz
```

These statistics are required to normalize new images consistently with the training data when using the saved model for predictions.

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
app.launch(share=True)
```

This generates a shareable link through which users can test the image classification system.

## 12. How to Run the Project

The model has already been trained and saved. You do not need to train it again to use the prediction interface.

The following steps explain how to load the dataset, load the saved model, and launch Gradio in Google Colab.

### Step 1: Open the Notebook

Open:

```text
src/cifar_10_updated.ipynb
```

Upload the notebook to Google Colab.

### Step 2: Import Required Libraries

Run the first two import cells to import NumPy, TensorFlow, Matplotlib, Pickle, OS, and the other required libraries.

### Step 3: Upload the CIFAR-10 Dataset

Run the dataset upload cell and select:

```text
cifar-10-python.tar.gz
```

### Step 4: Extract the Dataset

Run the following cells in order:

- Get the uploaded archive filename.
- Extract the dataset archive.
- Set the dataset directory path.
- Define the `load_batch()` function.
- Load and combine the five training batches and the test batch.

### Step 5: Define Class Names

Run the cell containing:

```python
class_names = [...]
```

This defines the ten CIFAR-10 category names.

### Step 6: Prepare the Dataset

Run the following cells:

- Training and validation split.
- Image conversion and normalization.

**Important:** Use the same training mean and standard deviation saved during the original training. If you have saved them in `cifar10_preprocessing.npz`, load those values rather than calculating new statistics.

### Step 7: Upload the Trained Model

Run the model upload cell and select:

```text
cifar10modal.keras
```

### Step 8: Install Gradio

Run:

```python
!pip install -q gradio
```

### Step 9: Import Gradio

Run the cell containing:

```python
import gradio as gr
```

### Step 10: Define the Prediction Function

Run the cell containing:

```python
def predict_image(image):
```

This function preprocesses the uploaded image and generates predictions using the trained CNN model.

### Step 11: Create the Gradio Interface

Run the cell containing:

```python
app = gr.Interface(...)
```

This creates the image-upload interface and prediction outputs.

### Step 12: Launch the Application

Run:

```python
app.launch(share=True)
```

A shareable Gradio link will be generated.

Open the link, upload an image, and view the predicted CIFAR-10 category with its confidence scores.

### Important: Cells to Skip

Since the model has already been trained, **do not run the following training-related cells again**:

- Data augmentation configuration.
- CNN model building.
- Model compilation.
- Training callbacks.
- `model.fit()` training cell.
- Training-history plotting cells, unless you want to regenerate the graphs.

You only need the dataset and preprocessing setup, saved model loading, prediction function, and Gradio interface to demonstrate the project.

## 13. Conclusion

This project demonstrates the development of a custom CNN-based image classification system using TensorFlow and Keras.

It combines image preprocessing, normalization, data augmentation, convolutional neural networks, batch normalization, dropout, and L2 regularization to classify images into 10 CIFAR-10 categories.

The model was trained for **30 epochs** and achieved **80.50% test accuracy**. An interactive Gradio interface was also implemented to allow users to upload images and view classification predictions.

The project demonstrates the complete workflow of developing, training, evaluating, and using a deep learning image classification model.

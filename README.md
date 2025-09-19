 Image Classification with CIFAR-10
This project implements an image classification model using a Convolutional Neural Network (CNN) to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) into one of 10 categories.
 Dataset
The CIFAR-10 dataset consists of:
- 60,000 32x32 color images in 10 classes
- 50,000 training images and 10,000 test images
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
 Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook or Google Colab
Project Features
- Load and visualize the CIFAR-10 dataset
- Normalize image data
- Build and train a Convolutional Neural Network (CNN)
- Evaluate model performance on test data
- Plot training/validation accuracy and loss
- Predict and visualize classification results
Model Architecture
```text
Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Conv2D -> Flatten -> Dense -> Output


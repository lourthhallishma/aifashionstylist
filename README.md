#  AI Fashion Stylist

An AI-powered fashion assistant that classifies clothing items and provides styling suggestions using deep learning. 
Upload an image of a clothing item and get instant fashion advice.


---

## Features

- **Deep Learning Classification** — Uses a CNN trained on Fashion MNIST to classify images into 10 fashion categories.
- **Style Recommendations** — Get smart outfit suggestions for each clothing item.
- **Simple, Responsive UI** — Built with Gradio for an accessible and modern web interface.
- **Deployed on Hugging Face Spaces** — Easily accessible, no installation needed.

---

## Tech Stack

| Area        | Tools/Frameworks                        |
|-------------|------------------------------------------|
| Frontend    | Gradio                                   |
| Backend     | Python, TensorFlow, Keras                |
| Model       | Convolutional Neural Network (CNN)       |
| Dataset     | Fashion MNIST                            |
| Deployment  | Hugging Face Spaces                      |

## Model Details

The core model is a **Convolutional Neural Network (CNN)** trained on the **Fashion MNIST** dataset:

- **Dataset:** 60,000 training + 10,000 test grayscale images (28x28 pixels) of 10 fashion categories.
- **Architecture:**
  - 2 convolutional layers with ReLU and max pooling
  - Dense layer with 128 units
  - Final softmax classification layer
- **Performance:** Achieves ~90% test accuracy
- **Output:** Predicts 1 of 10 classes:
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

The trained model is saved as `fashion_model.h5` and loaded at runtime.

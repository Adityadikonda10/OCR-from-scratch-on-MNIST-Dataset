# OCR From Scratch on MNIST Dataset

Hey everyone! This project is a basic version of Optical Character Recognition (OCR). It's built on the previously uploaded repository [Neural Network from Scratch (nnfs)](https://github.com/Adityadikonda10/Neural-Network-from-Scratch-nnfs-) which is a practical implementation of Neural Networks to identify handwritten numbers from the well-known MNIST Dataset.

## Table of Contents
- Overview.
- Dataset.
- Model Architecture.
- Emphasis on Optimizer (Adam)
- Training.
- Testing.
- Dependencies
- How to Run.


## Overview

The project demonstrates the implementation of Neural Networks from Scratch for classifying handwritten digits using Forward Propagation, Backward Propagation, Activation Function, Loss Function, and Optimization methods. The dataset utilized is the MNIST Dataset.

## Dataset 

The well-known MNIST dataset was created in 1994 to train Artificial Neural Networks. This dataset comprises 60,000 training-labelled images and 10,000 testing-labelled images of handwritten digits, each 28 $\times$ 28 pixels. These images are classified into 10 classes ranging from 0 to 9.

## Model Architecture

The neural network architecture used for this project consists of a fully connected network.

- Input Layer: 128 neurons gets 784 inputs, with ReLU activaton function. 
    - 784 inputs are the vector format of the image. since the image is 28 $\times$ 28 ```image.shape()``` gives ```(28, 28)``` pixels in dimentions. when vectorised ```image.shape()``` gives ```(1, 784)```.
- Hidden Layer-1: 64 neurons gets 128 inputs, with ReLU activaton function. 
- Hidden Layer-2: 64 neurons gets 64 inputs, with ReLU activaton function. 
- Output Layer: 10 neurons gets 64 inputs, with SoftMax activaton function. 

## Emphasis on Adam Optimizier
The previous optimizer used for the NNFS repository was SGD. The issue with using SGD was the slow learning process and high computational requirements. Switching to Adam (Adaptive Moment Estimation) has made the process faster and lighter in terms of resource consumption. This optimizer operates based on the concept of velocity, adaptively adjusting the step size to find the local minima.

### Compute the biased first moment estimate $m_t$
$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t 
$$ 

- where $g_t$ is the gradient at time step $t$.

### Compute the biased second raw moment estimate $v_t$:

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 
$$
### Compute bias-corrected first moment estimate $\hat{m}^t$:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

### Compute bias-corrected second raw moment estimate $\hat{v}^t$:
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$

### Update the parameters $θ$:
$$\theta_t = \theta_{t-1} - \frac{\alpha \cdot \hat{v}_t}{\sqrt{\hat{m}_t} + \epsilon}$$


## Training

The file _**training_OCR.py**_ is used to train and save the model. The training process involves classifying the input data to the correct class through forward propagation and updating the weights and biases of the network after each epoch through backpropagation.

- Learning Rate is set to 0.01.
- number of epochs is set to 301

- Training file contains classes:
    - ```Layer_Dense``` 
    - ```ActivationReLU```
    - ```ActivationSoftMax```
    - ```Loss```
    - ```LossfunctionLossCategoricalCrossEntropy```
    - ```Optimizer_Adam```

The trained model is saved as **OCR_Model_128,64,64,10.pkl**.



## Testing

The trained model can be used to test the 10,000 test images from the MNIST Dataset. In the file _**testing_OCR.py**_, a random batch of 10 examples is tested in series of 5.

## Results

Loss and Accuracy are tracked at every epoch and plottet for visualisation.
#### Perfomance
- Highest achieved Accuracy is 99% at 300<sup>th</sup> epoch.
- 90% Accuracy achieved at 53<sup>rd</sup> epoch.


## Dependencies 
- Numpy
- Matplotlib
- Keras

## How to Run

To run the code and train the neural network, follow these steps:

1. Clone the Repository:
```sh
git clone https://github.com/Adityadikonda10/Neural-Network-from-Scratch-nnfs-
```

2. Install Dependencies:
```sh
pip install numpy
pip install matplotlib
pip install keras
```

3. Training the Model:
```sh
python training_OCR.py
```

4. Testing the Model:
```sh
python testing_OCR.py
```

## Acknowledgements

This project is primarily based on nnfs by harrison Kinsley at Sentdex's YouTube playlist
- [Neural Networks from Scratch](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&pp=iAQB).

Samson Zhang's video on 
- [Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=4s)

Andreas Zinonos's video on
- [Beginner Deep Learning Tutorial | MNIST Digits Classification Neural Network in Python, Keras YouTube video.](https://www.youtube.com/watch?v=BfCPxoYCgo0&t=1012s)

CampusX's video on 
- [Adam Optimizer Explained in detail](https://www.youtube.com/watch?v=N5AynalXD9g)

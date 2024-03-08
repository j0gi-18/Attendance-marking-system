This repository is an application of high performance face recognition using FaceNet, 
Multi-Task Cascaded Convolutional Neural Networks(MTCNN) and Random Forest classifier.

Required setup 

<B>Camera</B>

camera selection is a very important step as they are available in various resolutions such as 

1MP 1280\*720 pixels (720p HD) </br>
2MP 1980\*1080 pixels (1080p Full HD) </br>
3MP 2048\*1536 pixels </br>
4MP 2688\*1520 pixels</br>
5MP 2592\*1944 pixels</br>
8MP 3840\*2160 pixels (4K Ultra HD)</br>

As the pixel count increases, we get better video quality so the camera can capture more information or detailing in every frame and range of the camera increases as well, but the CPU/GPU now has to evaluate so many pixels in the frame. 

<B>OpenCv</br>

OpenCV provides functions to read and write images and videos in various formats, making it easy to work with different media sources. In our recognition algorithm we used OpenCV for reading input from the camera and the input from the camera is displayed in frames.

<B>PyTorch</br>

PyTorch is a open-source deep learning framework. It provides a flexible and efficient way to build and train neural networks for various machine learning tasks. Basically a machine learning framework to execute the models.

<B>facenet-pytorch</br>

Facenet-pytorch is a Python library that provides an implementation of the FaceNet model using the PyTorch deep learning framework.

<B>MTCNN</br>

Multi-task Cascaded Convolutional Networks is a deep learning based library for face detection and facial landmark detection. It has very high  its accuracy, speed, and capable to handle challenging conditions of various poses, and occlusions. 

<B>Scikit-learn </br>

Sk learn provides a comprehsive set of algorithms, tools, and utilities for data preparation, model fitting, and evaluation. 

<B>Random Forest</br>

Random Forest is an ensemble learning algorithm which combines multiple simpler models to make a single most accurate prediction. It uses multiple decision trees each focusing a set of features and each tree divides based on the features. 






















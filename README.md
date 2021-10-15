# NUACCESS-autoencoder
Autoencoder that can reconstruct hyperspectral imaging data. Written in Python. 

In this project, we take the first step toward solving the problem of separating the mixed X-ray Fluorescence (XRF) image of an overpainted oil painting that has top and bottom layers. In order to obtain distinctive XRF images for the different layers, an autoencoder is needed first to extract features from the hyperspectral imaging (HSI) data of the painting. We propose using a convolutional neural network, which is a type of artificial neural network in machine learning, for the architecture of the autoencoder. An artificial neural network has multiple layers, each holds a collection of neurons and can learn information of a certain problem. It is an unsupervised learning method, meaning that the input and output are not labeled and the algorithm needs to discover the patterns in the data by itself. Our results also show this autoencoder’s potential to be adopted as a denoiser for processing hyperspectral images. 
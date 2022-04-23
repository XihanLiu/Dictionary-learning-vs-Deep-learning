# Dictionary-learning-vs-Deep-learning
Brief Description:
	We proposed to compare the three approaches between dictionary learning, deep learning and the combination of sparse coding and deep learning, which we call deep sparse neural network(DSNN). The proposed DSNN has most of the standard deep learning layers, including convolutional layer, activation layer, and fully connected layer. ResNet will be adopted as the backbone of the DSNN, and the difference between DSNN and traditional ResNet will be the input features. For traditional ResNet, the input will be the full size images. As for DSNN, the input will be the processed images after the layer of sparse coding, which has a smaller size compared to the full size images. 
Goals:
We would use YaleB as our dataset to perform the evaluations with 1000 learning images and 800 testing images. 
To compare the three proposed approaches, we achieve error rates:
Dictionary learning method with random sensing matrix of sensing rates:R = 0.01, R= 0.05,and R = 0.1. (sensing rate  =  #Number of measurements / #Number of training images)
	Deep learning network of 18 layers ResNet with a learning rate of 0.0025 and 100 epochs.
	DSNN of 18 layers with a learning rate of 0.0025 and 100 epochs and sensing rates R = 0.01, R = 0.05, and R = 0.1.
	The DSNN we implement is intended to have an error rate lower than 10%.

Reference:
Tang, Hao, et al. "When dictionary learning meets deep learning: Deep dictionary learning and coding network for image recognition with limited data." IEEE transactions on neural networks and learning systems 32.5 (2020): 2129-2141.
He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
Robert Calderbank, Sina Jafarpour, and Robert Schapire, “Compressed learning: Universal sparse dimensionality reduction and learning in the measurement domain,” Tech. Rep., 2009.


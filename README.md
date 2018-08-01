# MachineLearning MNIST
Academic project for the *Machine Learning* course consisting in creating a Neural Network **from scratch** to recognise hand-written digits stored in the [**MNIST** database](http://yann.lecun.com/exdb/mnist/). The NN is dynamic: it's possible to specify an arbitrary number of layers and nodes per layer.

The project is divided into 2 parts. The main differences are the algorithm used for the weights update and the error function. Respectively:
- Part A uses the *gradient descent* algorithm and the *square sum* error function;
- Part B uses the *resilient backpropagation* algorithm and the *cross entropy* error function.

The entry point is the *main.m* file in each part.

**NB**: the MNIST database is not included in this repository. You need to download it and modify rows 17-19 of main.m as appropriate.

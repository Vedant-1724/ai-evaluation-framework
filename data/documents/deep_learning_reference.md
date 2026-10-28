# Deep Learning Fundamentals

## What Is a Neural Network?

A neural network is a machine learning model inspired by the way biological neurons process information. It consists of interconnected layers of artificial neurons that transform input data into predictions or useful representations.

A typical neural network contains an input layer, one or more hidden layers, and an output layer. Each neuron receives values, applies weights and a bias, and passes the result through an activation function. The network learns by adjusting its weights and biases during training to reduce prediction errors.

## Main Components of a Neural Network

- **Input layer:** Receives the input features.
- **Hidden layers:** Learn patterns and relationships in the data.
- **Output layer:** Produces the final prediction.
- **Weights and biases:** Trainable parameters that influence the network's output.
- **Activation functions:** Introduce non-linearity so the network can learn complex patterns.
- **Loss function:** Measures the difference between predictions and target values.
- **Optimizer:** Updates model parameters to reduce the loss.

## How Are Neural Networks Trained?

During training, the network performs a forward pass to generate predictions. A loss function measures prediction error. Backpropagation calculates gradients, and an optimizer uses those gradients to update the weights and biases. This process repeats over multiple training iterations until the model learns useful patterns.
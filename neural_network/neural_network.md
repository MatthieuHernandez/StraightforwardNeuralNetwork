---
layout: default
title: Neural network
nav_order: 5
permalink: /neural_network
has_children: true
---

# Neural network
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/neural_network.png" att="Neural network" height="300px" class="center"/>
</p>

## Presentation
**StraightforwardNeuralNetwork** is the main class of the library. 
## Declaration 
This is the constructor used to create a neural network.
```cpp
StraightforwardNeuralNetwork::StraightforwardNeuralNetwork(vector<LayerModel> models);
```
**Arguments**
 * **models**: The vector of all layer of function of neurons. 
 [See list of all available layer types.]({{site.baseurl}}/layer/layer.html)

 Here is an example of neural networks.
```cpp
StraightforwardNeuralNetwork neuralNetwork({
        Input(4),
        FullyConnected(15),
        FullyConnected(5),
        FullyConnected(3)
    });
```
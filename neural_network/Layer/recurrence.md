---
layout: default
title: Recurrence
parent: Layers
grand_parent: Neural network
nav_order: 3
---

# Recurrent layer
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/recurrence.png" att="GRU neuron" width="500px" class="center"/>
</p>

## Presentation
This layer is a simple fully connected layer with recurrence, where each neuron of the layer receive in additional input the output of this same neuron but at t-1. This allows it to exhibit temporal dynamic behavior. the RNNs can use this recurrence like a kind of memory to process [temporal]({{site.baseurl}}/temporal) or [sequential]({{site.baseurl}}/sequential) data.

## Declaration
This is the function used to declare a Recurrent layer.
```cpp
LayerModel Recurrence(int numberOfNeurons, activation activation = activation::tanh);
```
Here is an example of neural networks using a GRU layer.
```cpp
 StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        Recurrence(12),
        FullyConnected(6),
        FullyConnected(1, activation::tanh)
    });
```
[See an example of GRU layer on dataset]({{site.baseurl}}/examples/audio_cats_and_dogs.html)

## Algorithms and References
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/gru2.png" att="GRU neuron" width="320px" class="left"/>
</p>
GRU implementation is based on **Fully Gated Unit** schema on [Gated recurrent unit Wikipadia page](https://en.wikipedia.org/wiki/Gated_recurrent_unit).
Also used [Back propagation through time](https://cran.r-project.org/web/packages/rnn/vignettes/GRU_units.html).
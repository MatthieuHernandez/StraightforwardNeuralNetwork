---
layout: default
title: GRU layer
parent: Layers
grand_parent: Neural network
nav_order: 4
---

# Gated Recurrent Units layer
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/gru1.png" att="GRU neuron" width="320px" class="center"/>
</p>

## Presentation
This layer is a simple fully connected layer with Gated recurrent units instead of simple neurons. Gated recurrent units (GRUs) are improved version of standard recurrent neural network. The GRU is like a long short-term memory (LSTM) but with fewer parameters. This is really useful when predicting time series or classifying sequential data.

## Declaration
This is the function used to declare a GRU layer.
```cpp
LayerModel GruLayer(int numberOfNeurons);
```
**Arguments**
 * **numberOfNeurons**: The number of neurons in the layer.

Here is an example of neural networks using a GRU layer.
```cpp
StraightforwardNeuralNetwork neuralNetwork({
        Input(1),
        GRULayer(10),
        FullyConnected(1)
    });
```
[See an example of GRU layer on dataset]({{site.baseurl}}/examples/audio_cats_and_dogs.html)

## Algorithms and References
<p>
    <img src="{{site.baseurl}}/assets/images/neural_network/gru2.png" att="GRU neuron" width="320px" class="left"/>
</p>
GRU implementation is based on **Fully Gated Unit** schema on [Gated recurrent unit Wikipadia page](https://en.wikipedia.org/wiki/Gated_recurrent_unit).
Also used [Back propagation through time](https://cran.r-project.org/web/packages/rnn/vignettes/GRU_units.html).
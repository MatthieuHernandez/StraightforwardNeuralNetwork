---
layout: default
title: Evaluation
parent: Neural network
nav_order: 4
---

# Evaluation &#127919;
{: .no_toc }

This is the list of metrics used by `StraighforwardNeuralNetwork` to evaluate the neural network from `Data`.

* TOC
{:toc}

## Accuracy
Mainly used for [classification]({{site.baseurl}}/data/classification.html) and [multiple classification]({{site.baseurl}}/data/multiple_classification.html).<br/>
The accuracy corresponds to the number of well-classified examples in the testing set of dataset. It is equivalent to precision.

Here the function to retreive the accuracy:
```c++
void StraightforwardNeuralNetwork::getGlobalClusteringRate() const;
```
[See more detail on precision](https://en.wikipedia.org/wiki/Precision_and_recall)

## Weighted Accuracy
Mainly used for [classification]({{site.baseurl}}/data/classification.html).<br/>
The weighted accuracy corresponds to the number of well-classified examples in the testing set of dataset. It is equivalent to recall.

Here the function to retreive the weighted accuracy:
```c++
void StraightforwardNeuralNetwork::getWeightedClusteringRate() const;
```
[See more detail on recall](https://en.wikipedia.org/wiki/Precision_and_recall)

## F1 score
Mainly used for binary [classification]({{site.baseurl}}/data/classification.html).<br/>
The F1 score is calculated from the precision and recall.

Here the function to retreive the weighted F1 score:
```c++
void StraightforwardNeuralNetwork::getF1Score() const;
```
[See more detail on F1 score](https://en.wikipedia.org/wiki/Precision_and_recall)

## Mean Absolute Error
Mainly used for [regression]({{site.baseurl}}/data/regression.html).<br/>
The MAE is a measure of errors between the output of neural network and the expected output.

Here the function to retreive the weighted MAE:
```c++
void StraightforwardNeuralNetwork::getMeanAbsoluteError() const;
```
[See more detail on MAE](https://en.wikipedia.org/wiki/Mean_absolute_error)

## Root-Mean-Square Error
RMSE is very similar to MAE except that examples with a large error have a greater impact on the value of RMSE due to the square.
Mainly used for [regression]({{site.baseurl}}/data/regression.html). It is equivalent to root-mean-square deviation (RMSE).

Here the function to retreive the weighted RMSE:
```c++
void StraightforwardNeuralNetwork::getRootMeanSquaredError() const;
```
[See more detail on RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
#include <boost/serialization/export.hpp>
#include "Layer.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Layer<Perceptron>)

template <class N>
Layer<N>::Layer(LayerModel& model, StochasticGradientDescent* optimizer)
{
    this->numberOfInputs = model.numberOfInputs;
    this->neurons.reserve(model.numberOfNeurons);
    for (int n = 0; n < model.numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(model.numberOfInputsByNeurons, model.activation, optimizer);
    }
}

template <class N>
void Layer<N>::train(vector<float>& inputErrors)
{
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        neurons[n].train(inputErrors[n]);
    }
}

template <class N>
int Layer<N>::isValid() const
{
    if (this->neurons.size() != this->getNumberOfNeurons()
        || this->getNumberOfNeurons() < 1
        || this->getNumberOfNeurons() > 1000000)
        return 201;

    int numberOfOutput = 1;
    auto shape = this->getShapeOfOutput();
    for (int s : shape)
        numberOfOutput *= s;

    if (numberOfOutput != this->getNumberOfNeurons())
        return 202;

    for (auto& neuron : this->neurons)
    {
        int err = neuron.isValid();
        if (err != 0)
            return err;
    }
    return 0;
}

template <class N>
int Layer<N>::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

template <class N>
int Layer<N>::getNumberOfNeurons() const
{
    return this->neurons.size();
}

template <class N>
int Layer<N>::getNumberOfParameters() const
{
    int sum = 0;
    for (auto& neuron : this->neurons)
    {
        sum += neuron.getNumberOfParameters();
    }
    return sum;
}

template <class N>
bool Layer<N>::operator==(const Layer& layer) const
{
    return this->numberOfInputs == layer.numberOfInputs
        && this->errors == layer.errors
        && this->neurons == layer.neurons;
}

template <class T>
bool Layer<T>::operator!=(const Layer& layer) const
{
    return !(*this == layer);
}

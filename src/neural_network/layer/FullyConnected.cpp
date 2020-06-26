#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "FullyConnected.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(FullyConnected)

FullyConnected::FullyConnected(LayerModel& model, StochasticGradientDescent* optimizer)
     : Layer(model, optimizer)
{
}

inline
unique_ptr<Layer> FullyConnected::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<FullyConnected>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

vector<float> FullyConnected::output(const vector<float>& inputs, bool temporalReset)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = neurons[n].output(inputs);
    }
    return outputs;
}

vector<float> FullyConnected::backOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = neurons[n].backOutput(inputErrors[n]);
        for(int n = 0; n < errors.size(); ++n)
            errors[n] += error[n];
    }
    return errors;
}

std::vector<int> FullyConnected::getShapeOfOutput() const
{
    return {this->getNumberOfNeurons()};
}

int FullyConnected::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->getNumberOfInputs())
            return 203;
    }
    return this->Layer::isValid();
}

bool FullyConnected::operator==(const FullyConnected& layer) const
{
    return this->Layer::operator==(layer);
}

bool FullyConnected::operator!=(const FullyConnected& layer) const
{
    return !(*this ==layer);
}
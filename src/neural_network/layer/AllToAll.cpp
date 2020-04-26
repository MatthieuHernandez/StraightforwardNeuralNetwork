#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "AllToAll.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(AllToAll)

AllToAll::AllToAll(LayerModel& model, StochasticGradientDescent* optimizer)
     : Layer(model, optimizer)
{
}

inline
unique_ptr<Layer> AllToAll::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<AllToAll>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

inline
vector<float> AllToAll::createInputsForNeuron(int neuronNumber, const vector<float>& inputs, bool temporalReset) const
{
    return inputs;
}

inline
 void AllToAll::insertBackOutputForNeuron(int neuronNumber, const std::vector<float>& error, std::vector<float>& errors) const
{
    for(int n = 0; n < errors.size(); ++n)
    {
        errors[n] += error[n];
    }
}

std::vector<int> AllToAll::getShapeOfOutput() const
{
    return {this->getNumberOfNeurons()};
}

int AllToAll::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->getNumberOfInputs())
            return 203;
    }
    return this->Layer::isValid();
}

bool AllToAll::operator==(const AllToAll& layer) const
{
    return this->Layer::operator==(layer);
}

bool AllToAll::operator!=(const AllToAll& layer) const
{
    return !(*this ==layer);
}
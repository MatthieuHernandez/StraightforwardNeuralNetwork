#include <boost/serialization/export.hpp>
#include "Filter.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Convolution)

Filter::Filter(LayerModel& model, StochasticGradientDescent* optimizer)
     : Layer(model, optimizer)
{
    this->numberOfFilters = model.numberOfFilters;
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

vector<float> Filter::output(const vector<float>& inputs, bool temporalReset)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto neuronInputs = this->createInputsForNeuron(n, inputs);
        outputs[n] = neurons[n].output(neuronInputs);
    }
    return outputs;
}

vector<float> Filter::backOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = neurons[n].backOutput(inputErrors[n]);
        this->insertBackOutputForNeuron(n, error, errors);
    }
    return errors;
}

int Filter::isValid() const
{
    return this->Layer::isValid();
}

inline 
bool Filter::operator==(const Filter& layer) const
{
    return this->Layer::operator==(layer)
    && this->numberOfFilters == layer.numberOfFilters
    && this->sizeOfFilterMatrix == layer.sizeOfFilterMatrix
    && this->shapeOfInput == layer.shapeOfInput;
}

inline 
bool Filter::operator!=(const Filter& layer) const
{
    return !(*this == layer);
}
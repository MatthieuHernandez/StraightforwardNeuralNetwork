#include <boost/serialization/export.hpp>
#include "FilterLayer.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(FilterLayer)

FilterLayer::FilterLayer(LayerModel& model, StochasticGradientDescent* optimizer)
     : Layer(model, optimizer)
{
    this->numberOfFilters = model.numberOfFilters;
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

vector<float> FilterLayer::output(const vector<float>& inputs, bool temporalReset)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < (int)this->neurons.size(); ++n)
    {
        auto neuronInputs = this->createInputsForNeuron(n, inputs);
        outputs[n] = this->neurons[n].output(neuronInputs);
    }
    return outputs;
}

vector<float> FilterLayer::backOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < (int)this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        this->insertBackOutputForNeuron(n, error, errors);
    }
    return errors;
}

int FilterLayer::isValid() const
{
    return this->Layer::isValid();
}

bool FilterLayer::operator==(const BaseLayer& layer) const
{
   try
    {
        const auto& f = dynamic_cast<const FilterLayer&>(layer);
        return this->Layer::operator==(layer)
            && this->numberOfInputs == f.numberOfInputs
            && this->neurons == f.neurons;
    }
    catch (bad_cast&)
    {
        return false;
    }
}

bool FilterLayer::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
#include <boost/serialization/export.hpp>
#include "FilterLayer.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(FilterLayer)

FilterLayer::FilterLayer(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Layer(model, optimizer)
{
    this->numberOfFilters = model.numberOfFilters;
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

inline
vector<float> FilterLayer::computeOutput(const vector<float>& inputs, [[maybe_unused]] bool temporalReset)
{
    vector<float> outputs(this->neurons.size());
    for (int i = 0, n = 0; n < (int)this->neurons.size(); ++i)
    {
        auto neuronInputs = this->createInputsForNeuron(i, inputs);
        for (int f = 0; f < this->numberOfFilters; ++f, ++n)
        {
            outputs[n] = this->neurons[n].output(neuronInputs);
        }
    }
    return outputs;
}

inline
vector<float> FilterLayer::computeBackOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < (int)this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        const int neuronIndex = n / this->numberOfFilters;
        this->insertBackOutputForNeuron(neuronIndex, error, errors);
    }
    return errors;
}

std::vector<int> FilterLayer::getShapeOfInput() const
{
    return this->shapeOfInput;
}

std::vector<int> FilterLayer::getShapeOfOutput() const
{
    return this->shapeOfOutput;
}

int FilterLayer::getSizeOfFilterMatrix() const
{
    return this->sizeOfFilterMatrix;
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
            && this->sizeOfFilterMatrix == f.sizeOfFilterMatrix
            && this->shapeOfInput == f.shapeOfInput
            && this->shapeOfOutput == f.shapeOfOutput;
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

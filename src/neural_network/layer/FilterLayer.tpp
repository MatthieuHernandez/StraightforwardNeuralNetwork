#include "FilterLayer.hpp"

template<NeuronInput I>
FilterLayer<I>::FilterLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
     : Layer(model, optimizer)
{
    this->numberOfFilters = model.numberOfFilters;
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

template<NeuronInput I>
std::vector<float> FilterLayer<I>::computeOutput(const std::vector<float>& inputs, bool temporalReset)
{
    std::vector<float> outputs(this->neurons.size());
    for (int n = 0; n < (int)this->neurons.size(); ++n)
    {
        auto neuronInputs = this->createInputsForNeuron(n, inputs);
        outputs[n] = this->neurons[n].output(neuronInputs);
    }
    return outputs;
}

template<NeuronInput I>
std::vector<float> FilterLayer<I>::computeBackOutput(std::vector<float>& inputErrors)
{
    std::vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < (int)this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        this->insertBackOutputForNeuron(n, error, errors);
    }
    return errors;
}

template<NeuronInput I>
std::vector<int> FilterLayer<I>::getShapeOfOutput() const
{
    return this->shapeOfOutput;
}

template<NeuronInput I>
int FilterLayer<I>::isValid() const
{
    return Layer<SimpleNeuron>::isValid();
}

template<NeuronInput I>
bool FilterLayer<I>::operator==(const BaseLayer& layer) const
{
   try
    {
        const auto& f = dynamic_cast<const FilterLayer<I>&>(layer);
        return Layer<SimpleNeuron>::operator==(layer)
            && this->numberOfInputs == f.numberOfInputs
            && this->neurons == f.neurons;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

template<NeuronInput I>
bool FilterLayer<I>::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
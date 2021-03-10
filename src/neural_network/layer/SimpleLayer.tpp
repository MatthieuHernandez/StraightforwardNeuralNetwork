
template <class N>
SimpleLayer<N>::SimpleLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Layer<N>(model, optimizer)
{
}

template <class N>
std::unique_ptr<BaseLayer> SimpleLayer<N>::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = std::make_unique<SimpleLayer<N>>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

template <class N>
std::vector<float> SimpleLayer<N>::computeOutput(const std::vector<float>& inputs, bool temporalReset)
{
    std::vector<float> outputs(this->neurons.size());
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = this->neurons[n].output(inputs);
    }
    return outputs;
}

template <class N>
std::vector<float> SimpleLayer<N>::computeBackOutput(std::vector<float>& inputErrors)
{
    std::vector<float> errors(this->numberOfInputs, 0);
    for (size_t n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = this->neurons[n].backOutput(inputErrors[n]);
        for(size_t e = 0; e < errors.size(); ++e)
            errors[e] += error[e];
    }
    return errors;
}

template <class N>
std::vector<int> SimpleLayer<N>::getShapeOfOutput() const
{
    return {this->getNumberOfNeurons()};
}

template <class N>
int SimpleLayer<N>::isValid() const
{
    for (auto& neuron : this->neurons)
    {
        if (neuron.getNumberOfInputs() != this->getNumberOfInputs())
            return 203;
    }
    return this->Layer<N>::isValid();
}

template <class N>
bool SimpleLayer<N>::operator==(const BaseLayer& layer) const
{
    return this->Layer<N>::operator==(layer);
}

template <class N>
bool SimpleLayer<N>::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}
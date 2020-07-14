
template <class N>
SimpleLayer<N>::SimpleLayer(LayerModel& model, StochasticGradientDescent* optimizer)
    : Layer(model, optimizer)
{
}

template <class N>
std::unique_ptr<BaseLayer> SimpleLayer<N>::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<SimpleLayer<N>>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

template <class N>
std::vector<float> SimpleLayer<N>::output(const std::vector<float>& inputs, bool temporalReset)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = neurons[n].output(inputs);
    }
    return outputs;
}

template <class N>
std::vector<float> SimpleLayer<N>::backOutput(std::vector<float>& inputErrors)
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

template <class N>
std::vector<int> SimpleLayer<N>::getShapeOfOutput() const
{
    return {this->getNumberOfNeurons()};
}

template <class N>
int SimpleLayer<N>::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->getNumberOfInputs())
            return 203;
    }
    return this->Layer::isValid();
}

template <class N>
bool SimpleLayer<N>::operator==(const BaseLayer& layer) const
{
    return this->Layer::operator==(layer);
}

template <class N>
bool SimpleLayer<N>::operator!=(const BaseLayer& layer) const
{
    return !(*this ==layer);
}
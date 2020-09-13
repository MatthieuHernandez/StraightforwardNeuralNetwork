
template <class N>
Layer<N>::Layer(LayerModel& model, StochasticGradientDescent* optimizer)
{
    this->numberOfInputs = model.numberOfInputs;
    this->neurons.reserve(model.numberOfNeurons);
    for (int n = 0; n < model.numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(model.neuron, optimizer);
    }
}

template <class N>
void Layer<N>::train(std::vector<float>& inputErrors)
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
BaseNeuron* Layer<N>::getNeuron(int index)
{
    return static_cast<BaseNeuron*>(&this->neurons[index]);
}

template <class N>
int Layer<N>::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

template <class N>
int Layer<N>::getNumberOfNeurons() const
{
    return (int)this->neurons.size();
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
bool Layer<N>::operator==(const BaseLayer& layer) const
{
    try
    {
        const Layer& l = dynamic_cast<const Layer&>(layer);
        return typeid(*this).hash_code() == typeid(layer).hash_code()
            && this->numberOfInputs == l.numberOfInputs
            && this->errors == l.errors
            && this->neurons == l.neurons;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

template <class N>
bool Layer<N>::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}

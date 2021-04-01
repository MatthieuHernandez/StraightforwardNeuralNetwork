template <BaseNeuron2 N>
Layer<N>::Layer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
{
    this->numberOfInputs = model.numberOfInputs;
    this->neurons.reserve(model.numberOfNeurons);
    for (int n = 0; n < model.numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(model.neuron, optimizer);
    }
    LayerOptimizerFactory::build(this->optimizers, model, this);
}

template <BaseNeuron2 N>
Layer<N>::Layer(const Layer& layer)
{
    this->numberOfInputs = layer.numberOfInputs;
    this->neurons = layer.neurons;

    this->optimizers.reserve(layer.optimizers.size());
    for (auto& optimizer : layer.optimizers)
        this->optimizers.emplace_back(optimizer->clone(this));
}

template <BaseNeuron2 N>
std::vector<float> Layer<N>::output(const std::vector<float>& inputs, bool temporalReset)
{
    auto output = this->computeOutput(inputs, temporalReset);
    for (auto& optimizer : this->optimizers)
        optimizer->applyAfterOutputForTesting(output);
    return output;
}

template <BaseNeuron2 N>
std::vector<float> Layer<N>::outputForTraining(const std::vector<float>& inputs, bool temporalReset)
{
    auto output = this->computeOutput(inputs, temporalReset);
    for (auto& optimizer : this->optimizers)
        optimizer->applyAfterOutputForTraining(output, temporalReset);
    return output;
}

template <BaseNeuron2 N>
std::vector<float> Layer<N>::backOutput(std::vector<float>& inputErrors)
{
    for (auto& optimizer : this->optimizers)
        optimizer->applyBeforeBackpropagation(inputErrors);
    auto error = this->computeBackOutput(inputErrors);
    return error;
}

template <BaseNeuron2 N>
void Layer<N>::train(std::vector<float>& inputErrors)
{
    for (auto& optimizer : this->optimizers)
        optimizer->applyBeforeBackpropagation(inputErrors);
    for (size_t n = 0; n < this->neurons.size(); ++n)
        neurons[n].train(inputErrors[n]);
}

template <BaseNeuron2 N>
int Layer<N>::isValid() const
{
    if (this->getNumberOfNeurons() != (int)this->neurons.size()
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

template <BaseNeuron2 N>
void* Layer<N>::getNeuron(int index)
{
    return static_cast<void*>(&this->neurons[index]);
}

template <BaseNeuron2 N>
float Layer<N>::getAverageOfAbsNeuronWeights() const 
{
    auto sum = 0.0f;
    for (auto& n : this->neurons)
        for (auto w : n.getWeights())
            sum += abs(w);
    sum /= static_cast<float>(this->neurons.size());
    return sum;
}

template <BaseNeuron2 N>
float Layer<N>::getAverageOfSquareNeuronWeights() const 
{
    auto sum = 0.0f;
    for (auto& n : this->neurons)
        for (auto w : n.getWeights())
            sum += w*w;
    sum /= static_cast<float>(this->neurons.size());
    return sum;
}

template <BaseNeuron2 N>
int Layer<N>::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

template <BaseNeuron2 N>
int Layer<N>::getNumberOfNeurons() const
{
    return (int)this->neurons.size();
}

template <BaseNeuron2 N>
int Layer<N>::getNumberOfParameters() const
{
    int sum = 0;
    for (auto& neuron : this->neurons)
    {
        sum += neuron.getNumberOfParameters();
    }
    return sum;
}

template <BaseNeuron2 N>
bool Layer<N>::operator==(const BaseLayer& layer) const
{
    try
    {
        const auto& l = dynamic_cast<const Layer&>(layer);

        return typeid(*this).hash_code() == typeid(layer).hash_code()
            && this->numberOfInputs == l.numberOfInputs
            && this->neurons == l.neurons
            && [this, &l]()
            {
                for (size_t o = 0; o < this->optimizers.size(); ++o)
                {
                    if (*this->optimizers[o] != *l.optimizers[o])
                        return false;
                }
                return true;
            }();
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

template <BaseNeuron2 N>
bool Layer<N>::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}

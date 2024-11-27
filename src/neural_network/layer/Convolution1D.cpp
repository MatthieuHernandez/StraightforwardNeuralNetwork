#include <boost/serialization/export.hpp>
#include "Convolution1D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Convolution1D::Convolution1D(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : FilterLayer(model, optimizer)
{
    this->shapeOfOutput = {
        this->numberOfFilters,
        this->shapeOfInput[X] - (this->kernelSize - 1),
    };
    this->numberOfNeuronsPerFilter = 1;
    this->buildKernelIndexes();
}

void Convolution1D::buildKernelIndexes()
{
    this->kernelIndexes.resize(this->numberOfKernelsPerFilter);
    const int maxC = this->shapeOfInput[C];
    const int kSize = this->kernelSize;
    const int kIndexSize = static_cast<int>(this->kernelIndexes.size());
    for (int k = 0; k < kIndexSize; ++k)
    {
        this->kernelIndexes[k].resize(this->sizeOfNeuronInputs);
        for (int x = 0; x < kSize; ++x)
        {
            const int inputIndexX = (k + x) * maxC;
            const int kernelIndexX = x * maxC;
            for (int c = 0; c < maxC; ++c)
            {
                const int inputIndex = inputIndexX + c;
                const int kernelIndex = kernelIndexX + c;
                this->kernelIndexes[k][kernelIndex] = inputIndex;
            }
        }
    }
}

inline
unique_ptr<BaseLayer> Convolution1D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<Convolution1D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

int Convolution1D::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->kernelSize * this->shapeOfInput[C])
            return 203;
    }
    return this->FilterLayer::isValid();
}

std::string Convolution1D::summary() const
{
    stringstream ss;
    ss << "------------------------------------------------------------" << endl;
    ss << " Convolution1D" << endl;
    ss << "                Input shape:  [" << this->shapeOfInput[0] << ", " << this->shapeOfInput[1] << "]" << endl;
    ss << "                Filters:      " << this->numberOfFilters << endl;
    ss << "                Kernel size:  " << this->kernelSize << endl;
    ss << "                Parameters:   " << this->getNumberOfParameters() << endl;
    ss << "                Activation:   " << this->neurons[0].outputFunction->getName() << endl;
    ss << "                Output shape: [" << this->shapeOfOutput[0] << ", " << this->shapeOfOutput[1] << "]" << endl;
    if (!optimizers.empty())
    {
        ss << "                Optimizers:   " << optimizers[0]->summary() << endl;
    }
    for (size_t o = 1; o < this->optimizers.size(); ++o)
    {
        ss << "                              " << optimizers[o]->summary() << endl;
    }
    return ss.str();
}

inline
vector<float> Convolution1D::computeOutput(const vector<float>& inputs, [[maybe_unused]] bool temporalReset)
{
    vector<float> outputs(this->numberOfKernels);
    vector<float> neuronInputs(this->sizeOfNeuronInputs);
    for (size_t k = 0, o = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (size_t i = 0; i < neuronInputs.size(); ++i)
        {
            const auto& index = kernelIndexes[k][i];
            neuronInputs[i] = inputs[index];
        }
        for (size_t n = 0; n < this->neurons.size(); ++n, ++o)
        {
            outputs[o] = this->neurons[n].output(neuronInputs);
        }
    }
    return outputs;
}

inline
vector<float> Convolution1D::computeBackOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (size_t k = 0, i = 0; k < this->kernelIndexes.size(); ++k)
    {
        for (auto& neuron : this->neurons)
        {
            auto& error = neuron.backOutput(inputErrors[i]);
            for (size_t e = 0; e < error.size(); ++e)
            {
                const auto& index = kernelIndexes[k][e];
                errors[index] += error[e];
            }
            ++i;
        }
    }
    return errors;
}

inline
void Convolution1D::computeTrain(std::vector<float>& inputErrors)
{
    for (int n = 0; n < this->numberOfFilters; ++n)
        this->neurons[n].train(inputErrors[n]);
}
bool Convolution1D::operator==(const BaseLayer& layer) const
{
    return this->FilterLayer::operator==(layer);
}

bool Convolution1D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}

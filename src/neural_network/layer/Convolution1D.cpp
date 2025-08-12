#include "Convolution1D.hpp"

#include <boost/serialization/export.hpp>
#include <utility>

#include "LayerModel.hpp"

namespace snn::internal
{
Convolution1D::Convolution1D(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Convolution(model, std::move(optimizer))
{
    this->shapeOfOutput = {this->numberOfFilters, this->shapeOfInput[X]};
    this->numberOfNeuronsPerFilter = 1;
    this->buildKernelIndexes();
}

void Convolution1D::buildKernelIndexes()
{
    this->kernelIndexes.resize(this->numberOfKernelsPerFilter);
    const int maxC = this->shapeOfInput[C];
    const int maxX = this->shapeOfInput[X];
    const int kSize = this->kernelSize;
    const int pad = (kSize - 1) / 2;
    const int kIndexSize = static_cast<int>(this->kernelIndexes.size());
    for (int k = 0; k < kIndexSize; ++k)
    {
        this->kernelIndexes[k].resize(this->sizeOfNeuronInputs, -1);
        for (int x = 0; x < kSize; ++x)
        {
            const int inputX = (k + x) - pad;
            if (inputX < 0 || inputX >= maxX)
            {
                continue;
            }
            const int inputIndexX = inputX * maxC;
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

inline auto Convolution1D::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer>
{
    auto layer = std::make_unique<Convolution1D>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto Convolution1D::isValid() const -> errorType
{
    for (const auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->kernelSize * this->shapeOfInput[C])
        {
            return errorType::conv1DLayerWrongNumberOfInputs;
        }
    }
    return this->FilterLayer::isValid();
}

auto Convolution1D::summary() const -> std::string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------\n";
    summary << " Convolution1D\n";
    summary << "                Input shape:  [" << this->shapeOfInput[0] << ", " << this->shapeOfInput[1] << "]\n";
    summary << "                Filters:      " << this->numberOfFilters << '\n';
    summary << "                Kernel size:  " << this->kernelSize << '\n';
    summary << "                Parameters:   " << this->getNumberOfParameters() << '\n';
    summary << "                Activation:   " << this->neurons[0].outputFunction->getName() << '\n';
    summary << "                Output shape: [" << this->shapeOfOutput[0] << ", " << this->shapeOfOutput[1] << "]"
            << '\n';
    if (!optimizers.empty())
    {
        summary << "                Optimizers:   " << optimizers[0]->summary() << '\n';
    }
    for (size_t o = 1; o < this->optimizers.size(); ++o)
    {
        summary << "                              " << optimizers[o]->summary() << '\n';
    }
    return summary.str();
}

inline auto Convolution1D::operator==(const BaseLayer& layer) const -> bool
{
    return this->FilterLayer::operator==(layer);
}
}  // namespace snn::internal

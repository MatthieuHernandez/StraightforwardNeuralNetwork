#include "FilterLayer.hpp"

#include <boost/serialization/export.hpp>

#include "LayerModel.hpp"

namespace snn::internal
{

FilterLayer::FilterLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Layer(model, optimizer),
      numberOfFilters(model.numberOfFilters),
      numberOfKernels(model.numberOfKernels),
      numberOfKernelsPerFilter(model.numberOfKernelsPerFilter),
      kernelSize(model.kernelSize),
      sizeOfNeuronInputs(model.neuron.numberOfInputs),
      shapeOfInput(model.shapeOfInput)
{
}

auto FilterLayer::getShapeOfInput() const -> std::vector<int> { return this->shapeOfInput; }

auto FilterLayer::getShapeOfOutput() const -> std::vector<int> { return this->shapeOfOutput; }

auto FilterLayer::getKernelSize() const -> int { return this->kernelSize; }

auto FilterLayer::isValid() const -> errorType
{
    int numberOfOutput = 1;
    auto shape = this->getShapeOfOutput();
    for (const int s : shape)
    {
        numberOfOutput *= s;
    }
    if (numberOfOutput != this->numberOfKernels)
    {
        return errorType::filterLayerWrongNumberOfOutputs;
    }

    return this->Layer::isValid();
}

auto FilterLayer::operator==(const BaseLayer& layer) const -> bool
{
    try
    {
        const auto& f = dynamic_cast<const FilterLayer&>(layer);
        return this->Layer::operator==(layer) && this->numberOfFilters == f.numberOfFilters &&
               this->numberOfKernels == f.numberOfKernels &&
               this->numberOfKernelsPerFilter == f.numberOfKernelsPerFilter &&
               this->numberOfNeuronsPerFilter == f.numberOfNeuronsPerFilter && this->kernelSize == f.kernelSize &&
               this->sizeOfNeuronInputs == f.sizeOfNeuronInputs && this->shapeOfInput == f.shapeOfInput &&
               this->shapeOfOutput == f.shapeOfOutput && this->kernelIndexes == f.kernelIndexes;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}
}  // namespace snn::internal

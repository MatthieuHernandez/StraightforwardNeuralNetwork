#include "FilterLayer.hpp"

#include <boost/serialization/export.hpp>

#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

FilterLayer::FilterLayer(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : Layer(model, optimizer)
{
    this->numberOfFilters = model.numberOfFilters;
    this->numberOfKernels = model.numberOfKernels;
    this->numberOfKernelsPerFilter = model.numberOfKernelsPerFilter;
    this->kernelSize = model.kernelSize;
    this->shapeOfInput = model.shapeOfInput;
    this->sizeOfNeuronInputs = model.neuron.numberOfInputs;
}

std::vector<int> FilterLayer::getShapeOfInput() const { return this->shapeOfInput; }

std::vector<int> FilterLayer::getShapeOfOutput() const { return this->shapeOfOutput; }

int FilterLayer::getKernelSize() const { return this->kernelSize; }

int FilterLayer::isValid() const
{
    int numberOfOutput = 1;
    auto shape = this->getShapeOfOutput();
    for (int s : shape) numberOfOutput *= s;

    if (numberOfOutput != this->numberOfKernels) return 202;

    return this->Layer::isValid();
}

bool FilterLayer::operator==(const BaseLayer& layer) const
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
    catch (bad_cast&)
    {
        return false;
    }
}

bool FilterLayer::operator!=(const BaseLayer& layer) const { return !(*this == layer); }

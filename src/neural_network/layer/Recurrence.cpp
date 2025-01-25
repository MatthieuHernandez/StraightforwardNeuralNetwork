#include "Recurrence.hpp"

#include <boost/serialization/export.hpp>

namespace snn::internal
{
Recurrence::Recurrence(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : SimpleLayer(model, optimizer)
{
}

auto Recurrence::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer>
{
    auto layer = std::make_unique<Recurrence>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto Recurrence::summary() const -> string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------" << endl;
    summary << " Recurrence" << endl;
    summary << "                Input shape:  [" << this->getNumberOfNeurons() << "]" << endl;
    summary << "                Neurons:      " << this->getNumberOfNeurons() << endl;
    summary << "                Parameters:   " << this->getNumberOfParameters() << endl;
    summary << "                Activation:   " << this->neurons[0].outputFunction->getName() << endl;
    summary << "                Output shape: [" << this->getNumberOfInputs() << "]" << endl;
    if (!optimizers.empty())
    {
        summary << "                Optimizers:   " << optimizers[0]->summary() << endl;
    }
    for (size_t o = 1; o < this->optimizers.size(); ++o)
    {
        summary << "                              " << optimizers[o]->summary() << endl;
    }
    return summary.str();
}
}  // namespace snn::internal

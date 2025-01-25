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

auto Recurrence::summary() const -> std::string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------\n";
    summary << " Recurrence\n";
    summary << "                Input shape:  [" << this->getNumberOfNeurons() << "]\n";
    summary << "                Neurons:      " << this->getNumberOfNeurons() << '\n';
    summary << "                Parameters:   " << this->getNumberOfParameters() << '\n';
    summary << "                Activation:   " << this->neurons[0].outputFunction->getName() << '\n';
    summary << "                Output shape: [" << this->getNumberOfInputs() << "]\n";
    if (!optimizers.empty())
    {
        summary << "                Optimizers:   " << optimizers[0]->summary() << '\n';
        for (size_t opti = 1; opti < this->optimizers.size(); ++opti)
        {
            summary << "                              " << optimizers[opti]->summary() << '\n';
        }
        return summary.str();
    }
}
}  // namespace snn::internal
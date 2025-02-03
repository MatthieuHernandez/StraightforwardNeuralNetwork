#include "FullyConnected.hpp"

#include <boost/serialization/export.hpp>
#include <utility>

namespace snn::internal
{
FullyConnected::FullyConnected(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : SimpleLayer(model, std::move(optimizer))
{
}

auto FullyConnected::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer>
{
    auto layer = std::make_unique<FullyConnected>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto FullyConnected::summary() const -> std::string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------\n";
    summary << " FullyConnected\n";
    summary << "                Input shape:  [" << this->getNumberOfInputs() << "]\n";
    summary << "                Neurons:      " << this->getNumberOfNeurons() << '\n';
    summary << "                Parameters:   " << this->getNumberOfParameters() << '\n';
    summary << "                Activation:   " << this->neurons[0].outputFunction->getName() << '\n';
    summary << "                Output shape: [" << this->getNumberOfNeurons() << "]\n";
    if (!optimizers.empty())
    {
        summary << "                Optimizers:   " << optimizers[0]->summary() << '\n';
    }
    for (size_t opti = 1; opti < this->optimizers.size(); ++opti)
    {
        summary << "                              " << optimizers[opti]->summary() << '\n';
    }
    return summary.str();
}
}  // namespace snn::internal
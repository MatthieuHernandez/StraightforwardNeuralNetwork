#include "GruLayer.hpp"

#include <boost/serialization/export.hpp>

namespace snn::internal
{
GruLayer::GruLayer(LayerModel& model, std::shared_ptr<NeuralNetworkOptimizer> optimizer)
    : SimpleLayer(model, optimizer)
{
}

auto GruLayer::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> std::unique_ptr<BaseLayer>
{
    auto layer = std::make_unique<GruLayer>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto snn::internal::GruLayer::summary() const -> std::string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------\n";
    summary << " GruLayer\n";
    summary << "                Input shape:  [" << this->getNumberOfNeurons() << "]\n";
    summary << "                Neurons:      " << this->getNumberOfNeurons() << '\n';
    summary << "                Parameters:   " << this->getNumberOfParameters() << '\n';
    summary << "                Output shape: [" << this->getNumberOfInputs() << "]\n";
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

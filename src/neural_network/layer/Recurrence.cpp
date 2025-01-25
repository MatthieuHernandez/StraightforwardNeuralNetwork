#include "Recurrence.hpp"

#include <boost/serialization/export.hpp>

using namespace std;
using namespace snn;
using namespace internal;

Recurrence::Recurrence(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : SimpleLayer(model, optimizer)
{
}

auto Recurrence::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const -> unique_ptr<BaseLayer>
{
    auto layer = make_unique<Recurrence>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto Recurrence::summary() const -> string
{
    stringstream ss;
    ss << "------------------------------------------------------------" << endl;
    ss << " Recurrence" << endl;
    ss << "                Input shape:  [" << this->getNumberOfNeurons() << "]" << endl;
    ss << "                Neurons:      " << this->getNumberOfNeurons() << endl;
    ss << "                Parameters:   " << this->getNumberOfParameters() << endl;
    ss << "                Activation:   " << this->neurons[0].outputFunction->getName() << endl;
    ss << "                Output shape: [" << this->getNumberOfInputs() << "]" << endl;
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

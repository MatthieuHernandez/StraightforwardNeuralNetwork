#include "FullyConnected.hpp"

#include <boost/serialization/export.hpp>

using namespace std;
using namespace snn;
using namespace internal;

FullyConnected::FullyConnected(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
    : SimpleLayer(model, optimizer)
{
}

auto FullyConnected::clone(shared_ptr<NeuralNetworkOptimizer> optimizer) const -> unique_ptr<BaseLayer>
{
    auto layer = make_unique<FullyConnected>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

auto FullyConnected::summary() const -> string
{
    stringstream ss;
    ss << "------------------------------------------------------------" << endl;
    ss << " FullyConnected" << endl;
    ss << "                Input shape:  [" << this->getNumberOfInputs() << "]" << endl;
    ss << "                Neurons:      " << this->getNumberOfNeurons() << endl;
    ss << "                Parameters:   " << this->getNumberOfParameters() << endl;
    ss << "                Activation:   " << this->neurons[0].outputFunction->getName() << endl;
    ss << "                Output shape: [" << this->getNumberOfNeurons() << "]" << endl;
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

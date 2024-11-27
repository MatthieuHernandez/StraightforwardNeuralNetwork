#include <boost/serialization/export.hpp>
#include "GruLayer.hpp"

using namespace std;
using namespace snn;
using namespace internal;

GruLayer::GruLayer(LayerModel& model, shared_ptr<NeuralNetworkOptimizer> optimizer)
     : SimpleLayer(model, optimizer)
{
}

unique_ptr<BaseLayer> GruLayer::clone(std::shared_ptr<NeuralNetworkOptimizer> optimizer) const
{
    auto layer = make_unique<GruLayer>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].setOptimizer(optimizer);
    }
    return layer;
}

std::string snn::internal::GruLayer::summary() const
{
    stringstream ss;
    ss << "------------------------------------------------------------" << endl;
    ss << " GruLayer" << endl;
    ss << "                Input shape:  [" << this->getNumberOfNeurons() << "]" << endl;
    ss << "                Neurons:      " << this->getNumberOfNeurons() << endl;
    ss << "                Parameters:   " << this->getNumberOfParameters() << endl;
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

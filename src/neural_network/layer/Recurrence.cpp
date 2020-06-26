#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "LayerModel.hpp"
#include "Recurrence.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(Recurrence)

Recurrence::Recurrence(LayerModel& model, StochasticGradientDescent* optimizer)
     : Layer(model, optimizer), numberOfRecurrences(model.numberOfRecurrences), sizeToCopy(sizeof(float) * model.numberOfInputs)
{
    this->allInputs.resize(model.numberOfInputsByNeurons);
}

inline
unique_ptr<Layer> Recurrence::clone(StochasticGradientDescent* optimizer) const
{
    auto layer = make_unique<Recurrence>(*this);
    for (int n = 0; n < layer->getNumberOfNeurons(); ++n)
    {
        layer->neurons[n].optimizer = optimizer;
    }
    return layer;
}

vector<float> Recurrence::output(const vector<float>& inputs, bool temporalReset)
{
    this->addNewInputs(inputs, temporalReset);

    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = neurons[n].output(this->allInputs);
    }
    return outputs;
}

void Recurrence::addNewInputs(std::vector<float> inputs, bool temporalReset)
{
    if(temporalReset)
    {
        fill(this->allInputs.begin()+inputs.size(), this->allInputs.end(), -1);
    }
    else
    {
        for(int i = this->numberOfRecurrences; i > 0; --i)
        {
            const int index = i * this->numberOfInputs;
            memcpy(&this->allInputs[index], &this->allInputs[index - this->numberOfInputs], this->sizeToCopy);
        }
    }
    copy(inputs.begin(), inputs.end(), this->allInputs.begin());
}


vector<float> Recurrence::backOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& error = neurons[n].backOutput(inputErrors[n]);
        for(int n = 0; n < errors.size(); ++n)
            errors[n] += error[n];
    }
    return errors;
}

std::vector<int> Recurrence::getShapeOfOutput() const
{
    return {this->getNumberOfNeurons()};
}

int Recurrence::isValid() const
{
    for (auto& neuron : neurons)
    {
        if (neuron.getNumberOfInputs() != this->getNumberOfInputs() * (numberOfRecurrences + 1))
            return 203;
    }
    return this->Layer::isValid();
}

bool Recurrence::operator==(const Recurrence& layer) const
{
    return this->Layer::operator==(layer);
}

bool Recurrence::operator!=(const Recurrence& layer) const
{
    return !(*this ==layer);
}
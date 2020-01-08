#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include "AllToAll.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(AllToAll)

AllToAll::AllToAll(const int numberOfInputs,
                   const int numberOfNeurons,
                   activationFunction activation,
                   float* learningRate,
                   float* momentum)
     : Layer(allToAll, numberOfInputs, numberOfNeurons)
{
    for (int n = 0; n < numberOfNeurons; ++n)
    {
        this->neurons.emplace_back(numberOfInputs, activation, learningRate, momentum);
    }
}

unique_ptr<Layer> AllToAll::clone() const
{
    return make_unique<AllToAll>(*this);
}

vector<float> AllToAll::output(const vector<float>& inputs)
{
    vector<float> outputs(this->neurons.size());
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        outputs[n] = neurons[n].output(inputs);
    }
    return outputs;
}

vector<float> AllToAll::backOutput(vector<float>& inputsError)
{
    vector<float> errors(this->numberOfInputs);
    for (int n = 0; n < numberOfInputs; ++n)
    {
        errors[n] = 0;
    }

    for (int n = 0; n < this->neurons.size(); ++n)
    {
        auto& result = neurons[n].backOutput(inputsError[n]);
        for (int r = 0; r < numberOfInputs; ++r)
            errors[r] += result[r];
    }
    return errors;
}

void AllToAll::train(vector<float>& inputsError)
{
    for (int n = 0; n < this->neurons.size(); ++n)
    {
        neurons[n].backOutput(inputsError[n]);
    }
}

int AllToAll::isValid() const
{
    return this->Layer::isValid();
}

bool AllToAll::operator==(const AllToAll& layer) const
{
    return this->Layer::operator==(layer);
}

bool AllToAll::operator!=(const AllToAll& layer) const
{
    return !(*this ==layer);
}
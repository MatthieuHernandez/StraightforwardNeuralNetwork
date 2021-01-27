#include <boost/serialization/export.hpp>
#include "MaxPooling1D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(MaxPooling1D)

MaxPooling1D::MaxPooling1D(LayerModel& model)
    : NoNeuronLayer(model)
{
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

inline
unique_ptr<BaseLayer> MaxPooling1D::clone(shared_ptr<NeuralNetworkOptimizer>) const
{
    return make_unique<MaxPooling1D>(*this);
}


std::vector<float> MaxPooling1D::computeOutput(const std::vector<float>& inputs, bool temporalReset)
{
    auto output = vector<float>(this->shapeOfInput[0], numeric_limits<float>::lowest());
    for (int i = 0; i < inputs.size(); ++i)
    {
        const int indexOutput = i / this->sizeOfFilterMatrix;
        if (output[indexOutput] <= inputs[i])
        {
            output[indexOutput] = inputs[i];
        }
    }
    return output;
}


vector<float> MaxPooling1D::output(const vector<float>& inputs, bool temporalReset)
{
    return this->computeOutput(inputs, temporalReset);
}

vector<float> MaxPooling1D::outputForBackpropagation(const vector<float>& inputs, bool temporalReset)
{
    return this->computeOutput(inputs, temporalReset);
}

std::vector<float> MaxPooling1D::backOutput(std::vector<float>& inputErrors)
{
    std::vector<float> errors;
    errors.reserve(this->numberOfOutputs);
    for (int i = 0, k = 0; i < this->numberOfOutputs; ++i)
    {
        for(int j = 0; k < this->numberOfInputs && j < this->sizeOfFilterMatrix; ++j, ++k)
            errors.push_back(inputErrors[i]);
    }
    return errors;
}

void MaxPooling1D::train(std::vector<float>& inputErrors)
{
}

int MaxPooling1D::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

vector<int> MaxPooling1D::getShapeOfOutput() const
{
    const int rest = this->shapeOfInput[0] % this->sizeOfFilterMatrix == 0 ? 0 : 1;

    return {
        this->shapeOfInput[0] / this->sizeOfFilterMatrix + rest,
        1
    };
}

int MaxPooling1D::isValid() const
{
    return 0;
}

inline
bool MaxPooling1D::operator==(const BaseLayer& layer) const
{
    try
    {
        const auto& l = dynamic_cast<const MaxPooling1D&>(layer);

        return typeid(*this).hash_code() == typeid(layer).hash_code()
            && this->sizeOfFilterMatrix == l.sizeOfFilterMatrix
            && this->shapeOfInput == l.shapeOfInput;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

inline
bool MaxPooling1D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}

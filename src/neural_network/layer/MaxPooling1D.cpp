#include <boost/serialization/export.hpp>
#include "MaxPooling1D.hpp"
#include "LayerModel.hpp"
#include "../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(MaxPooling1D)

MaxPooling1D::MaxPooling1D(LayerModel& model)
    : NoNeuronLayer(model)
{
    this->kernelSize = model.kernelSize;
    this->shapeOfInput = model.shapeOfInput;
}

inline
unique_ptr<BaseLayer> MaxPooling1D::clone(shared_ptr<NeuralNetworkOptimizer>) const
{
    return make_unique<MaxPooling1D>(*this);
}

std::vector<float> MaxPooling1D::output(const std::vector<float>& inputs, [[maybe_unused]] bool temporalReset)
{
    auto output = vector<float>(this->numberOfOutputs, numeric_limits<float>::lowest());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const size_t indexOutput = i / this->kernelSize;
        if (output[indexOutput] <= inputs[i])
        {
            output[indexOutput] = inputs[i];
        }
    }
    return output;
}

vector<float> MaxPooling1D::outputForTraining(const vector<float>& inputs, bool temporalReset)
{
    return this->output(inputs, temporalReset);
}

std::vector<float> MaxPooling1D::backOutput(std::vector<float>& inputErrors)
{
    std::vector<float> errors;
    errors.reserve(this->numberOfInputs);
    for (int i = 0, k = 0; i < this->numberOfOutputs; ++i)
    {
        for (int j = 0; k < this->numberOfInputs && j < this->kernelSize; ++j, ++k)
            errors.push_back(inputErrors[i]);
    }
    return errors;
}

void MaxPooling1D::train([[maybe_unused]] std::vector<float>& inputErrors)
{
}

int MaxPooling1D::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

std::vector<int> MaxPooling1D::getShapeOfInput() const
{
    return this->shapeOfInput;
}

vector<int> MaxPooling1D::getShapeOfOutput() const
{
    const int rest = this->shapeOfInput[X] % this->kernelSize == 0 ? 0 : 1;

    return {
        1,
        this->shapeOfInput[X] / this->kernelSize + rest
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
            && this->kernelSize == l.kernelSize
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

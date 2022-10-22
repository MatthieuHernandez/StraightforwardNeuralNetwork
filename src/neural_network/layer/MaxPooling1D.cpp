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

Tensor MaxPooling1D::output(const Tensor& inputs, [[maybe_unused]] bool temporalReset)
{
    auto output = Tensor(this->numberOfOutputs, numeric_limits<float>::lowest());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const size_t indexOutput = i / this->sizeOfFilterMatrix;
        if (output[indexOutput] <= inputs[i])
        {
            output[indexOutput] = inputs[i];
        }
    }
    return output;
}

Tensor MaxPooling1D::outputForTraining(const Tensor& inputs, bool temporalReset)
{
    return this->output(inputs, temporalReset);
}

Tensor MaxPooling1D::backOutput(Tensor& inputErrors)
{
    Tensor errors;
    errors.reserve(this->numberOfInputs);
    for (int i = 0, k = 0; i < this->numberOfOutputs; ++i)
    {
        for (int j = 0; k < this->numberOfInputs && j < this->sizeOfFilterMatrix; ++j, ++k)
            errors.push_back(inputErrors[i]);
    }
    return errors;
}

void MaxPooling1D::train([[maybe_unused]] Tensor& inputErrors)
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

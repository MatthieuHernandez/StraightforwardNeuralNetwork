#include <boost/serialization/export.hpp>
#include "MaxPooling2D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(MaxPooling2D)

MaxPooling2D::MaxPooling2D(LayerModel& model)
    : NoNeuronLayer(model)
{
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;

    const int restX = shapeOfInput[0] % this->sizeOfFilterMatrix == 0 ? 0 : 1;
    const int restY = shapeOfInput[1] % this->sizeOfFilterMatrix == 0 ? 0 : 1;

    this->shapeOfOutput = {
        this->shapeOfInput[0] / this->sizeOfFilterMatrix + restX,
        this->shapeOfInput[1] / this->sizeOfFilterMatrix + restY,
        1
    };
}

inline
unique_ptr<BaseLayer> MaxPooling2D::clone(shared_ptr<NeuralNetworkOptimizer>) const
{
    return make_unique<MaxPooling2D>(*this);
}


std::vector<float> MaxPooling2D::computeOutput(const std::vector<float>& inputs, bool temporalReset)
{
    auto output = vector<float>(this->numberOfOutputs, numeric_limits<float>::lowest());
    const int rest = this->shapeOfInput[0] % sizeOfFilterMatrix == 0 ? 0 : 1;
    for (int i = 0; i < (int)inputs.size(); ++i)
    {
        const int indexOutputX = (i % this->shapeOfInput[0]) / this->sizeOfFilterMatrix;
        const int indexOutputY = (i / this->shapeOfInput[0]) / this->sizeOfFilterMatrix;
        const int indexOutput = indexOutputY * (this->shapeOfInput[0] / this->sizeOfFilterMatrix + rest) + indexOutputX;
        if (output[indexOutput] <= inputs[i])
        {
            output[indexOutput] = inputs[i];
        }
    }
    return output;
}

vector<float> MaxPooling2D::output(const vector<float>& inputs, bool temporalReset)
{
    return this->computeOutput(inputs, temporalReset);
}

vector<float> MaxPooling2D::outputForTraining(const vector<float>& inputs, bool temporalReset)
{
    return this->computeOutput(inputs, temporalReset);
}

vector<float> MaxPooling2D::backOutput(std::vector<float>& inputErrors)
{
    std::vector<float> errors;
    errors.reserve(this->numberOfInputs);
    for (int x = 0; x < this->shapeOfInput[0]; ++x)
    {
        for (int y = 0; y < this->shapeOfInput[0]; ++y)
        {
            const int outputX = x / this->sizeOfFilterMatrix;
            const int outputY = y / this->sizeOfFilterMatrix;
            const int indexOutput = outputY * this->shapeOfOutput[0] + outputX;

            errors.push_back(inputErrors[indexOutput]);
        }
    }
    return inputErrors;
}

void MaxPooling2D::train(std::vector<float>& inputErrors)
{
}

int MaxPooling2D::getNumberOfInputs() const
{
    return this->numberOfInputs;
}

vector<int> MaxPooling2D::getShapeOfOutput() const
{
    return this->shapeOfOutput;
}

int MaxPooling2D::isValid() const
{
    return 0;
}

inline
bool MaxPooling2D::operator==(const BaseLayer& layer) const
{
    try
    {
        const auto& l = dynamic_cast<const MaxPooling2D&>(layer);

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
bool MaxPooling2D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}

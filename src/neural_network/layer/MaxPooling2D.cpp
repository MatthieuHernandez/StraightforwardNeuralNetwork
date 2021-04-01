#include <boost/serialization/export.hpp>
#include "MaxPooling2D.hpp"
#include "LayerModel.hpp"
#include "../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;
using namespace tools;

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


std::vector<float> MaxPooling2D::output(const std::vector<float>& inputs, bool temporalReset)
{
    auto output = vector<float>(this->numberOfOutputs, numeric_limits<float>::lowest());
    for (int i = 0; i < (int)inputs.size(); ++i)
    {
        const int outputX = roughenX(i, this->shapeOfInput[0], this->shapeOfInput[1]) / this->sizeOfFilterMatrix;
        const int outputY = roughenY(i, this->shapeOfInput[0], this->shapeOfInput[1]) / this->sizeOfFilterMatrix;
        const int indexOutput = flatten(outputX, outputY, this->shapeOfOutput[0]);

        if (output[indexOutput] <= inputs[i])
            output[indexOutput] = inputs[i];
    }
    return output;
}

vector<float> MaxPooling2D::outputForTraining(const vector<float>& inputs, bool temporalReset)
{
    return this->output(inputs, temporalReset);
}

vector<float> MaxPooling2D::backOutput(std::vector<float>& inputErrors)
{
    std::vector<float> errors;
    errors.reserve(this->numberOfInputs);

    for (int z = 0; z < this->shapeOfInput[2]; ++z)
    {
        for (int y = 0; y < this->shapeOfInput[1]; ++y)
        {
            for (int x = 0; x < this->shapeOfInput[0]; ++x)
            {
                const int outputX = x / this->sizeOfFilterMatrix;
                const int outputY = y / this->sizeOfFilterMatrix;
                const int i = flatten(outputX, outputY, this->shapeOfOutput[0]);

                errors.push_back(inputErrors[i]);
            }
        }
    }
    return errors;
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

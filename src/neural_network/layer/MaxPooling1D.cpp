#include <boost/serialization/export.hpp>
#include "MaxPooling1D.hpp"
#include "LayerModel.hpp"
#include "../../tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(MaxPooling1D)

MaxPooling1D::MaxPooling1D(LayerModel& model)
    : FilterLayer(model, nullptr)
{
    const int restX = shapeOfInput[X] % this->kernelSize == 0 ? 0 : 1;

    this->shapeOfOutput = {
        this->shapeOfInput[C],
        this->shapeOfInput[X] / this->kernelSize + restX,
    };
    this->sizeOfNeuronInputs = this->kernelSize * 1;
    this->numberOfNeuronsPerFilter = 0;
    this->numberOfOutputs = model.numberOfOutputs;
    this->maxValueIndexes.resize(this->numberOfOutputs);
    this->buildKernelIndexes();
}

void MaxPooling1D::buildKernelIndexes()
{
    this->kernelIndexes.resize(this->numberOfOutputs);
    const int maxC = this->shapeOfInput[C];
    const int kSize = this->kernelSize;
    for (int k = 0; k < this->kernelIndexes.size(); ++k)
    {
        this->kernelIndexes[k].resize(this->sizeOfNeuronInputs);
        const int kernelPosX = k / maxC;
        for (int x = 0; x < kSize; ++x)
        {
            const int inputIndexX = (kernelPosX * kSize + x) * maxC;

            const int c = k % maxC;
            const int inputIndex = inputIndexX + c;
            const int kernelIndex = x;
            if (inputIndex < this->numberOfInputs)
                this->kernelIndexes[k][kernelIndex] = inputIndex;
            else
                this->kernelIndexes[k][kernelIndex] = -1;
           
        }
    }
}

inline
unique_ptr<BaseLayer> MaxPooling1D::clone(shared_ptr<NeuralNetworkOptimizer>) const
{
    return make_unique<MaxPooling1D>(*this);
}

int MaxPooling1D::isValid() const
{
    if (this->maxValueIndexes.size() != this->numberOfOutputs
        && this->numberOfKernels != this->numberOfOutputs)
        return 204;
    return 0;
}

inline
vector<float> MaxPooling1D::computeOutput(const vector<float>& inputs, [[maybe_unused]] bool temporalReset)
{
    vector<float> outputs(this->numberOfKernels);
    for (size_t k = 0; k < this->kernelIndexes.size(); ++k)
    {
        this->maxValueIndexes[k] = -1;
        for (size_t i = 0; i < this->sizeOfNeuronInputs; ++i)
        {
            const auto& index = this->kernelIndexes[k][i];
            if (index >= 0) [[likely]]
                if (this->maxValueIndexes[k] == -1 || inputs[index] >= inputs[this->maxValueIndexes[k]])
                    this->maxValueIndexes[k] = index;
        }
        outputs[k] = inputs[this->maxValueIndexes[k]];
    }
    return outputs;
}

inline
vector<float> MaxPooling1D::computeBackOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (size_t e = 0; e < inputErrors.size(); ++e)
    {
        errors[this->maxValueIndexes[e]] = inputErrors[e];
    }
    return errors;
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

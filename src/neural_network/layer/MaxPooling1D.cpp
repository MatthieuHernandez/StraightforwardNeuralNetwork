#include "MaxPooling1D.hpp"

#include <boost/serialization/export.hpp>

#include "LayerModel.hpp"
#include "Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

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
    const int kIndexSize = static_cast<int>(this->kernelIndexes.size());
    for (int k = 0; k < kIndexSize; ++k)
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

inline unique_ptr<BaseLayer> MaxPooling1D::clone(shared_ptr<NeuralNetworkOptimizer>) const
{
    return make_unique<MaxPooling1D>(*this);
}

auto MaxPooling1D::isValid() const -> ErrorType
{
    if (this->maxValueIndexes.size() != static_cast<size_t>(this->numberOfOutputs) &&
        this->numberOfKernels != this->numberOfOutputs)
    {
        return ErrorType::maxPooling1DWrongNumberOfInputs;
    }
    return ErrorType::noError;
}

std::string MaxPooling1D::summary() const
{
    stringstream ss;
    ss << "------------------------------------------------------------" << endl;
    ss << " MaxPooling1D" << endl;
    ss << "                Input shape:  [" << this->shapeOfInput[0] << ", " << this->shapeOfInput[1] << "]" << endl;
    ss << "                Kernel size:  " << this->kernelSize << endl;
    ss << "                Output shape: [" << this->shapeOfOutput[0] << ", " << this->shapeOfOutput[1] << "]" << endl;
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

inline vector<float> MaxPooling1D::computeOutput(const vector<float>& inputs, [[maybe_unused]] bool temporalReset)
{
    vector<float> outputs(this->numberOfKernels);
    for (size_t k = 0; k < this->kernelIndexes.size(); ++k)
    {
        this->maxValueIndexes[k] = -1;
        for (int i = 0; i < this->sizeOfNeuronInputs; ++i)
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

inline vector<float> MaxPooling1D::computeBackOutput(vector<float>& inputErrors)
{
    vector<float> errors(this->numberOfInputs, 0);
    for (size_t e = 0; e < inputErrors.size(); ++e)
    {
        errors[this->maxValueIndexes[e]] = inputErrors[e];
    }
    return errors;
}

inline bool MaxPooling1D::operator==(const BaseLayer& layer) const
{
    try
    {
        const auto& l = dynamic_cast<const MaxPooling1D&>(layer);

        return typeid(*this).hash_code() == typeid(layer).hash_code() && this->kernelSize == l.kernelSize &&
               this->shapeOfInput == l.shapeOfInput;
    }
    catch (std::bad_cast&)
    {
        return false;
    }
}

inline bool MaxPooling1D::operator!=(const BaseLayer& layer) const { return !(*this == layer); }

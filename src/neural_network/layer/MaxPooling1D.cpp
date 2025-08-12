#include "MaxPooling1D.hpp"

#include <boost/serialization/export.hpp>

#include "LayerModel.hpp"
#include "Tools.hpp"

namespace snn::internal
{
MaxPooling1D::MaxPooling1D(LayerModel& model)
    : FilterLayer(model, nullptr),
      numberOfOutputs(model.numberOfOutputs)
{
    const int restX = shapeOfInput[X] % this->kernelSize == 0 ? 0 : 1;

    this->shapeOfOutput = {
        this->shapeOfInput[C],
        this->shapeOfInput[X] / this->kernelSize + restX,
    };
    this->sizeOfNeuronInputs = this->kernelSize * 1;
    this->numberOfNeuronsPerFilter = 0;

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
            {
                this->kernelIndexes[k][kernelIndex] = inputIndex;
            }
            else
            {
                this->kernelIndexes[k][kernelIndex] = -1;
            }
        }
    }
}

inline auto MaxPooling1D::clone(std::shared_ptr<NeuralNetworkOptimizer>) const -> std::unique_ptr<BaseLayer>
{
    return std::make_unique<MaxPooling1D>(*this);
}

auto MaxPooling1D::isValid() const -> errorType
{
    if (this->maxValueIndexes.size() != static_cast<size_t>(this->numberOfOutputs) &&
        this->numberOfKernels != this->numberOfOutputs)
    {
        return errorType::maxPooling1DWrongNumberOfInputs;
    }
    return errorType::noError;
}

auto MaxPooling1D::summary() const -> std::string
{
    std::stringstream summary;
    summary << "------------------------------------------------------------\n";
    summary << " MaxPooling1D\n";
    summary << "                Input shape:  [" << this->shapeOfInput[0] << ", " << this->shapeOfInput[1] << "]\n";
    summary << "                Kernel size:  " << this->kernelSize << '\n';
    summary << "                Output shape: [" << this->shapeOfOutput[0] << ", " << this->shapeOfOutput[1] << "]\n";
    if (!optimizers.empty())
    {
        summary << "                Optimizers:   " << optimizers[0]->summary() << '\n';
    }
    for (size_t opti = 1; opti < this->optimizers.size(); ++opti)
    {
        summary << "                              " << optimizers[opti]->summary() << '\n';
    }
    return summary.str();
}

inline auto MaxPooling1D::computeOutput(const std::vector<float>& inputs, [[maybe_unused]] bool temporalReset)
    -> std::vector<float>
{
    std::vector<float> outputs(this->numberOfKernels);
    for (size_t k = 0; k < this->kernelIndexes.size(); ++k)
    {
        this->maxValueIndexes[k] = -1;
        for (int i = 0; i < this->sizeOfNeuronInputs; ++i)
        {
            const auto& index = this->kernelIndexes[k][i];
            if (index >= 0)
            {
                [[likely]]
                if (this->maxValueIndexes[k] == -1 || inputs[index] >= inputs[this->maxValueIndexes[k]])
                {
                    this->maxValueIndexes[k] = index;
                }
            }
        }
        outputs[k] = inputs[this->maxValueIndexes[k]];
    }
    return outputs;
}

inline auto MaxPooling1D::computeBackOutput(std::vector<float>& inputErrors) -> std::vector<float>
{
    std::vector<float> errors(this->numberOfInputs, 0);
    for (size_t e = 0; e < inputErrors.size(); ++e)
    {
        errors[this->maxValueIndexes[e]] = inputErrors[e];
    }
    return errors;
}

inline auto MaxPooling1D::operator==(const BaseLayer& layer) const -> bool
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
}  // namespace snn::internal

#include <boost/serialization/export.hpp>
#include "MaxPooling2D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(MaxPooling2D)

MaxPooling2D::MaxPooling2D(LayerModel& model)
{
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

inline
unique_ptr<BaseLayer> MaxPooling2D::clone() const
{
    auto layer = make_unique<MaxPooling2D>(*this);
}

vector<float> MaxPooling2D::output(const vector<float>& inputs, bool temporalReset)
{
    auto output = vector<float>(inputs.size(), numeric_limits<float>::lowest());
    const int rest = this->shapeOfInput[0] % sizeOfFilterMatrix == 0 ? 0 : 1;
    for(int i = 0 ;i < inputs.size(); ++i)
    {
        const int indexOutputX = (i % this->shapeOfInput[0]) / this->sizeOfFilterMatrix;
        const int indexOutputY = (i / this->shapeOfInput[0]) / this->sizeOfFilterMatrix;
        const int indexOutput = indexOutputY * (this->shapeOfInput[0]/ (this->sizeOfFilterMatrix) + rest) + indexOutputX;
        if (output[indexOutput] <= inputs[i])
        {
            output[indexOutput] = inputs[i];
        }
    }
    return output;
}

vector<float> MaxPooling2D::outputForBackpropagation(const vector<float>& inputs, bool temporalReset)
{
    return this->output(inputs, temporalReset);
}

vector<int> MaxPooling2D::getShapeOfOutput() const
{
    const int restX = shapeOfInput[0] % this->sizeOfFilterMatrix == 0 ? 0 : 1;
    const int restY = shapeOfInput[1] % this->sizeOfFilterMatrix == 0 ? 0 : 1;

    return {
        this->shapeOfInput[0] / this->sizeOfFilterMatrix + restX,
        this->shapeOfInput[1] / this->sizeOfFilterMatrix + restY,
        1
    };
}

int MaxPooling2D::isValid() const
{
    return this->BaseLayer::isValid();
}

inline
bool MaxPooling2D::operator==(const BaseLayer& layer) const
{
    return this->BaseLayer::operator==(layer);
}

inline
bool MaxPooling2D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}

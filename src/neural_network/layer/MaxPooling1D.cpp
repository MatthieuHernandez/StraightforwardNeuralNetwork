#include <boost/serialization/export.hpp>
#include "MaxPooling1D.hpp"
#include "LayerModel.hpp"

using namespace std;
using namespace snn;
using namespace internal;

BOOST_CLASS_EXPORT(MaxPooling1D)

MaxPooling1D::MaxPooling1D(LayerModel& model)
{
    this->sizeOfFilterMatrix = model.sizeOfFilerMatrix;
    this->shapeOfInput = model.shapeOfInput;
}

inline
unique_ptr<BaseLayer> MaxPooling1D::clone() const
{
    auto layer = make_unique<MaxPooling1D>(*this);
}

vector<float> MaxPooling1D::output(const vector<float>& inputs, bool temporalReset)
{
    auto output = vector<float>(this->shapeOfInput[0], numeric_limits<float>::lowest());
    for(int i = 0 ;i < inputs.size(); ++i)
    {
        const int indexOutput = i / this->sizeOfFilterMatrix;
        if (output[indexOutput] <= inputs[i])
        {
            output[indexOutput] = inputs[i];
        }
    }
    return output;
}

vector<float> MaxPooling1D::outputForBackpropagation(const vector<float>& inputs, bool temporalReset)
{
    return this->output(inputs, temporalReset);
}

vector<int> MaxPooling1D::getShapeOfOutput() const
{
    const int rest = this->shapeOfInput[0] % this->sizeOfFilterMatrix == 0 ? 0 : 1;

    return {this->shapeOfInput[0] / this->sizeOfFilterMatrix + rest};
}

int MaxPooling1D::isValid() const
{
    return this->BaseLayer::isValid();
}

inline
bool MaxPooling1D::operator==(const BaseLayer& layer) const
{
    return this->BaseLayer::operator==(layer);
}

inline
bool MaxPooling1D::operator!=(const BaseLayer& layer) const
{
    return !(*this == layer);
}

#pragma once
#include <vector>
#include "../../../../tools/Tools.hpp"

using namespace std;
using namespace snn;
//using namespace internal;
using namespace tools;

struct NeuronInputFromConvolution2D
{
    const vector<float>& inputs;
    const vector<int>& shapeOfInput;
    const int sizeOfFilterMatrix;

    int neuronNumber;

    const float& operator[](int index) const
    {
        const int indexX = roughenX(index, this->sizeOfFilterMatrix, this->sizeOfFilterMatrix);
        const int indexY = roughenY(index, this->sizeOfFilterMatrix, this->sizeOfFilterMatrix);
        const int indexZ = roughenZ(index, this->sizeOfFilterMatrix, this->sizeOfFilterMatrix);

        const int neuronX = roughenX(neuronNumber, this->shapeOfInput[0]);
        const int neuronY = roughenY(neuronNumber, this->shapeOfInput[0]);

        const int i = flatten(neuronX + indexX, neuronY + indexY, indexZ, this->shapeOfInput[0], this->shapeOfInput[1]);
        return this->inputs[i];
    }

    size_t size() const
    {
        return static_cast<size_t>(this->sizeOfFilterMatrix * this->sizeOfFilterMatrix * this->shapeOfInput[2]);
    }
};

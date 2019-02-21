#include <algorithm>
#include <vector>
#include "Data.h"

using namespace std;
using namespace data;

void Data::loadData()
{
	unshuffle();
}

void Data::clearData()
{
	this->sets[training].labels.clear();
	this->sets[training].data.clear();
	this->sets[testing].labels.clear();
	this->sets[testing].data.clear();
	this->sets[training].size = 0;
	this->sets[testing].size = 0;
}

void Data::shuffle()
{
	rand();
	if (indexes.empty())
	{
		indexes.resize(sets[training].size);
		for (int i = 0; i < static_cast<int>(indexes.size()); i++)
			indexes[i] = i;
	}

	random_shuffle(indexes.begin(), indexes.end());
}

void Data::unshuffle()
{
	indexes.resize(sets[training].size);
	for (int i = 0; i < static_cast<int>(indexes.size()); i++)
		indexes[i] = i;
}

vector<float>& Data::getTrainingData(const int index)
{
	return this->sets[training].data[indexes[index]];
}

vector<float>& Data::getTestingData(const int index)
{
	return this->sets[testing].data[index];
}

vector<float>& Data::getTrainingOutputs(const int index)
{
	return this->sets[training].labels[indexes[index]];
}

std::vector<float>& Data::getData(set set, const int index)
{
	if (set == training)
		return this->getTrainingData(index);

	return this->getTestingData(index);
}

std::vector<float>& Data::getOutputs(set set, const int index)
{
	if (set == training)
		return this->getTrainingOutputs(index);

	return this->getTestingOutputs(index);
}

int Data::getLabel(set set, const int index)
{
	if (set == training)
		return this->getTrainingLabel(index);

	return this->getTestingLabel(index);
}

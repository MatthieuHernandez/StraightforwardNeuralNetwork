#include "DataForClassification.h"

using namespace std;

DataForClassification::DataForClassification()
{
	problem = classification;
}

int DataForClassification::getTrainingLabel(const int index)
{
	for (int i = 0; i < this->numberOfLabel; i++)
	{
		if (this->sets[training].labels[indexes[index]][i] == 1)
			return i;
	}
	throw exception("wrong label");
}

int DataForClassification::getTestingLabel(const int index)
{
	for (int i = 0; i < this->numberOfLabel; i++)
	{
		if (this->sets[testing].labels[index][i] == 1)
			return i;
	}
	throw exception("wrong label");
}
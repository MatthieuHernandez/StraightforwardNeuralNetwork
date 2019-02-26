#include "DataForRegression.h"

using namespace std;

DataForRegression::DataForRegression()
{
	problem = regression;
}

vector<float>& DataForRegression::getTestingOutputs(const int index)
{
	return this->sets[testing].labels[index];
}

#include "DataForRegression.h"
using namespace std;

vector<float>& DataForRegression::getTestingOutputs(const int index)
{
	return this->sets[testing].labels[index];
}

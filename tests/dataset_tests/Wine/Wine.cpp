#include <fstream>
#include <vector>
#include "Wine.hpp"
#include "data/DataForClassification.hpp"
#include "tools/Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Wine::Wine()
{
    this->loadData();
}

void Wine::loadData()
{
    vector2D<float> data;
    vector2D<float> labels;

    string line;
    ifstream file("./Wine/wine.data");
    
    if (!file.is_open())
        throw FileOpeningFailed();

    data.reserve(178);
    labels.reserve(178);

    while (getline(file, line))
    {
        vector<float> label;
        vector<float> values;

        float value = atof(line.substr(0, line.find(',')).c_str());
        
        label.resize(3, 0);
        label[value] = 1;
        labels.push_back(label);

        while (line != line.substr(line.find(',') + 1))
        {
            value = atof(line.substr(0, line.find(',')).c_str());
            values.push_back(value);
            line = line.substr(line.find(',') + 1);
        }
        value = atof(line.substr(0, line.find(',')).c_str());
        values.push_back(value);
        data.push_back(values);
    }
    file.close();
    this->data = make_unique<DataForClassification>(data, labels);
}

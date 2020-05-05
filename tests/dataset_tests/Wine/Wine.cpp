#include <fstream>
#include <string>
#include <vector>
#include "Wine.hpp"
#include "data/Data.hpp"
#include "tools/Tools.hpp"
#include "tools/ExtendedExpection.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Wine::Wine(string folderPath)
{
    this->loadData(folderPath);
}

void Wine::loadData(string folderPath)
{
    vector2D<float> data;
    vector2D<float> labels;

    string line;
    ifstream file(folderPath + "/wine.data", ios::in);
    
    if (!file.is_open())
        throw FileOpeningFailedException();

    data.reserve(178);
    labels.reserve(178);

    while (getline(file, line))
    {
        vector<float> label;
        vector<float> values;

        float value = static_cast<float>(atof(line.substr(0, line.find(',')).c_str()));

        label.resize(3, 0);
        label[static_cast<int>(value-1.0)] = 1;
        labels.push_back(label);
        line = line.substr(line.find(',') + 1);
        while (line != line.substr(line.find(',') + 1))
        {
            value = static_cast<float>(atof(line.substr(0, line.find(',')).c_str()));
            values.push_back(value);
            line = line.substr(line.find(',') + 1);
        }
        value = static_cast<float>(atof(line.substr(0, line.find(',')).c_str()));
        values.push_back(value);
        data.push_back(values);
    }
    file.close();
    this->data = make_unique<Data>(classification, data, labels);
}

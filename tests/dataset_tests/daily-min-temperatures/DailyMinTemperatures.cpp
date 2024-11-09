#include <fstream>
#include <string>
#include <vector>
#include "DailyMinTemperatures.hpp"
#include <snn/data/Data.hpp>
#include <snn/tools/Tools.hpp>
#include <snn/tools/ExtendedExpection.hpp>

using namespace std;
using namespace snn;
using namespace internal;


DailyMinTemperatures::DailyMinTemperatures(string folderPath, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences)
{
    this->loadData(folderPath);
}

void DailyMinTemperatures::loadData(string folderPath)
{
    vector2D<float> inputs;
    vector2D<float> labels;
    string line;
    ifstream file(folderPath + "/daily-min-temperatures.csv", ios::in);
        
    if (!file.is_open())
        throw FileOpeningFailedException();

    inputs.reserve(3650);

    getline(file, line); // ignore headers
    float previousValue = -273.15f;
    while (getline(file, line))
    {
        string date = line.substr(0, line.find(','));
        line = line.substr(line.find(',') + 1);
        float value = static_cast<float>(atof(line.substr(0, line.find(',')).c_str()));

        if (previousValue != -273.15f)
        {
            inputs.push_back({previousValue});
            labels.push_back({value});
        }
        previousValue = value;
    }
    file.close();
    this->data = make_unique<Data>(problem::regression, inputs, labels, nature::timeSeries, this->numberOfRecurrences);
    this->data->normalize(0, 1);
}

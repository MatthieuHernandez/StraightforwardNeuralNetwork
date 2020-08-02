#include <fstream>
#include <string>
#include <vector>
#include "DailyMinTemperatures.hpp"
#include "data/Data.hpp"
#include "tools/Tools.hpp"
#include "tools/ExtendedExpection.hpp"

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
    vector2D<float> data;
    vector2D<float> labels;
    string line;
    ifstream file(folderPath + "/daily-min-temperatures.csv", ios::in);
        
    if (!file.is_open())
        throw FileOpeningFailedException();

    data.reserve(3650);

    getline(file, line); // ignore headers
    float previousValue = -273.15f;
    while (getline(file, line))
    {
        string date = line.substr(0, line.find(','));
        line = line.substr(line.find(',') + 1);
        float value = static_cast<float>(atof(line.substr(0, line.find(',')).c_str()));

        if (previousValue != -273.15f)
        {
            data.push_back({previousValue});
            labels.push_back({value});
        }
        previousValue = value;
    }
    file.close();
    this->data = make_unique<Data>(problem::regression, data, labels, nature::timeSeries, this->numberOfRecurrences);
}

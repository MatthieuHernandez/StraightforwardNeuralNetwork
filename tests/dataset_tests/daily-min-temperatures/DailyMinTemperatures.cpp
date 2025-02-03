#include "DailyMinTemperatures.hpp"

#include <fstream>
#include <snn/data/Dataset.hpp>
#include <snn/tools/ExtendedExpection.hpp>
#include <snn/tools/Tools.hpp>
#include <string>
#include <vector>

using namespace snn;

DailyMinTemperatures::DailyMinTemperatures(std::string folderPath, int numberOfRecurrences)
    : numberOfRecurrences(numberOfRecurrences)
{
    this->loadData(folderPath);
}

void DailyMinTemperatures::loadData(const std::string& folderPath)
{
    vector2D<float> inputs;
    vector2D<float> labels;
    std::string line;
    std::ifstream file(folderPath + "/daily-min-temperatures.csv", std::ios::in);

    if (!file.is_open()) throw FileOpeningFailedException();

    inputs.reserve(3650);

    getline(file, line);  // ignore headers
    float previousValue = -273.15F;
    while (getline(file, line))
    {
        std::string date = line.substr(0, line.find(','));
        line = line.substr(line.find(',') + 1);
        auto value = static_cast<float>(atof(line.substr(0, line.find(',')).c_str()));

        if (previousValue != -273.15F)
        {
            inputs.push_back({previousValue});
            labels.push_back({value});
        }
        previousValue = value;
    }
    file.close();
    this->dataset =
        std::make_unique<Dataset>(problem::regression, inputs, labels, nature::timeSeries, this->numberOfRecurrences);
    this->dataset->normalize(0, 1);
}

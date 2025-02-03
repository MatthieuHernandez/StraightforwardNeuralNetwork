#include "Wine.hpp"

#include <fstream>
#include <snn/data/Dataset.hpp>
#include <snn/tools/ExtendedExpection.hpp>
#include <snn/tools/Tools.hpp>
#include <string>
#include <vector>

using namespace snn;

Wine::Wine(std::string folderPath) { this->loadData(folderPath); }

void Wine::loadData(std::string folderPath)
{
    vector2D<float> inputs;
    vector2D<float> labels;

    std::string line;
    std::ifstream file(folderPath + "/wine.data", std::ios::in);

    if (!file.is_open()) throw FileOpeningFailedException();

    inputs.reserve(178);
    labels.reserve(178);

    while (getline(file, line))
    {
        std::vector<float> label;
        std::vector<float> values;

        float value = static_cast<float>(atof(line.substr(0, line.find(',')).c_str()));

        label.resize(3, 0);
        label[static_cast<int>(value - 1.0)] = 1;
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
        inputs.push_back(values);
    }
    file.close();
    this->dataset = std::make_unique<Dataset>(problem::classification, inputs, labels);
    this->dataset->normalize(0, 1);
}

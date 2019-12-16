#include <stdexcept>
#include <string>
#include <fstream>
#include "Iris.hpp"
#include "tools/Tools.hpp"
#include "data/DataForClassification.hpp"

using namespace std;
using namespace snn;
using namespace internal;

Iris::Iris()
{
    this->loadData();
}

void Iris::loadData()
{
    vector2D<float> data;
    vector2D<float> labels;

    string line;
    ifstream file("./Iris/iris.data");
    int count = 0;
    int size = 150;
    vector<vector<string>> individuals;
    individuals.reserve(size);
    const vector<string> temp;

    if (!file.is_open())
        throw FileOpeningFailed();

    getline(file, line);
    while (getline(file, line))
    {
        individuals.push_back(temp);
        for (int i = 0; line != line.substr(line.find('\t') + 1); i++)
        {
            individuals[count].push_back(line.substr(0, line.find('\t')));
            line = line.substr(line.find('\t') + 1);
        }
        individuals[count].push_back(line.substr(0, line.find('\t')));
        count++;
    }
    file.close();
    data.resize(size);
    labels.resize(size);
    for (int i = 0; i < size; i++)
    {
        data[i].resize(4);
        labels[i].resize(3);
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < 4; j++)
            data[i][j] = stof(individuals[i][j]);

        if (individuals[i][4] == "setosa")
        {
            labels[i][0] = 1;
            labels[i][1] = 0;
            labels[i][2] = 0;
        }
        else if (individuals[i][4] == "versicolor")
        {
            labels[i][0] = 0;
            labels[i][1] = 1;
            labels[i][2] = 0;
        }
        else if (individuals[i][4] == "virginica")
        {
            labels[i][0] = 0;
            labels[i][1] = 0;
            labels[i][2] = 1;
        }
        else
            throw runtime_error("Cannot load iris data set");
    }
    this->data = make_unique<DataForClassification>(data, labels);
}
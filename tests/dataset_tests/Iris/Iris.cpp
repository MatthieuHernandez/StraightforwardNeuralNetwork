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
    int size = 0;
    vector2D<string> individuals;
    individuals.reserve(150);

    if (!file.is_open())
        throw FileOpeningFailed();

    while (getline(file, line) && line != "")
    {
        const vector<string> temp;
        individuals.push_back(temp);
        for (int i = 0; line != line.substr(line.find(',') + 1); i++)
        {
            individuals[size].push_back(line.substr(0, line.find(',')));
            line = line.substr(line.find(',') + 1);
        }
        individuals[size].push_back(line.substr(0, line.find(',')));
        size++;
    }
    file.close();
    data.resize(size);
    labels.resize(size);
    for (int i = 0; i < size; i++)
    {
        data[i].reserve(4);
        labels[i].resize(3);
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < 4; j++)
            data[i].push_back(stof(individuals[i][j]));

        if (individuals[i][4] == "Iris-setosa")
        {
            labels[i][0] = 1;
            labels[i][1] = 0;
            labels[i][2] = 0;
        }
        else if (individuals[i][4] == "Iris-versicolor")
        {
            labels[i][0] = 0;
            labels[i][1] = 1;
            labels[i][2] = 0;
        }
        else if (individuals[i][4] == "Iris-virginica")
        {
            labels[i][0] = 0;
            labels[i][1] = 0;
            labels[i][2] = 1;
        }
        else
            throw runtime_error("Cannot load iris dataset");
    }
    this->data = make_unique<DataForClassification>(data, labels);
}
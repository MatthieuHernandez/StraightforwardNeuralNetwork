#include "Iris.hpp"

#include <fstream>
#include <snn/data/Data.hpp>
#include <snn/tools/ExtendedExpection.hpp>
#include <snn/tools/Tools.hpp>
#include <stdexcept>
#include <string>

using namespace std;
using namespace snn;
using namespace internal;

Iris::Iris(string folderPath) { this->loadData(folderPath); }

void Iris::loadData(string folderPath)
{
    vector2D<float> inputs;
    vector2D<float> labels;

    string line;
    ifstream file(folderPath + "/bezdekIris.data");
    int size = 0;
    vector2D<string> individuals;
    individuals.reserve(150);

    if (!file.is_open()) throw FileOpeningFailedException();

    while (getline(file, line) && line.size() > 4)
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
    inputs.resize(size);
    labels.resize(size);
    for (int i = 0; i < size; i++)
    {
        inputs[i].reserve(4);
        labels[i].resize(3);
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < 4; j++) inputs[i].push_back(stof(individuals[i][j]));

        if (individuals[i][4][8] == 'o')  // Iris-setosa
        {
            labels[i][0] = 1;
            labels[i][1] = 0;
            labels[i][2] = 0;
        }
        else if (individuals[i][4][8] == 's')  // Iris-versicolor
        {
            labels[i][0] = 0;
            labels[i][1] = 1;
            labels[i][2] = 0;
        }
        else if (individuals[i][4][8] == 'g')  // Iris-virginica
        {
            labels[i][0] = 0;
            labels[i][1] = 0;
            labels[i][2] = 1;
        }
        else
            throw runtime_error("Cannot load iris dataset");
    }
    this->data = make_unique<Data>(problem::classification, inputs, labels);
}
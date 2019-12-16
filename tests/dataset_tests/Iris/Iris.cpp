#include <stdexcept>
#include <string>
#include <fstream>
#include "Iris.hpp"
#include "../../../src/tools/Tools.hpp"

using namespace std;

void Iris::loadData()
{
    vector<vector<float>> data;
    Vector<vector<float>> labels;

    string line;
    ifstream file("./iris.data");
    int count = 0;
    vector<vector<string>> individuals;
    individuals.reserve(this->size);
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
    data.resize(this->size);
    labels.resize(this->size);
    for (int i = 0; i < this->size; i++)
    {
        data[i].resize(this->sizeOfData);
        labels[i].resize(this->numberOfLabel);
    }

    for (int i = 0; i < this->size; i++)
    {
        for (int j = 0; j < this->sizeOfData; j++)
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
    this-> data = new DataForClassification(data, labels);
}
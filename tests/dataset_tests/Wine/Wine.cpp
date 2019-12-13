#include <fstream>
#include <vector>
#include "Wine.hpp"

using namespace std;

void Wine::loadData()
{
    vector<vector<float>> data;
    Vector<vector<float>> labels;

    string line;
    ifstream file("../wine.data.txt");
    
    if (!file.is_open())
        throw FileOpeningFailed();

    data.reserve(178);
    labels.reserve(178);

    while (getline(file, line))
    {
        vector<float> label;
        vector<float> values;

        float value = itoa(line.substr(0, line.find(',')));
        
        label.resize(3, 0);
        label[value] = 1;
        labels.push_back(label);

        while (line != line.substr(line.find(',') + 1))
        {
            value = itoa(line.substr(0, line.find(',')));
            values.push_back(value);
            line = line.substr(line.find(',') + 1);
        }
        values.push_back(line.substr(0, line.find(',')));
        data.push_back(values);
    }
    file.close();
    this-> data = new DataForClassification(data, labels);
}

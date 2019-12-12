void Mnist::loadData()
{
    ifstream imagesTestFile;
    ifstream labelsTestFile;
    ifstream imagesTrainFile;
    ifstream labelsTrainFile;

    imagesTrainFile.open("./train-images.idx3-ubyte", ios::in | ios::binary);
    labelsTrainFile.open("./train-labels.idx1-ubyte", ios::in | ios::binary);
    imagesTestFile.open("./t10k-images.idx3-ubyte", ios::in | ios::binary);
    labelsTestFile.open("./t10k-labels.idx1-ubyte", ios::in | ios::binary);

    vector<vector<float>> trainingInputs = this->readImages();
    vector<vector<float>> trainingLabels = this->readLabes();
    vector<vector<float>> testingInputs = this->readImages();
    vector<vector<float>> testingLabels = this->readLabes();

    this->data = new DataForClassification(trainingInputs,
                                           trainingLabels,
                                           testingInputs,
                                           testingLabels)

}

void Mnist::readImages(ifstream& images)
{
    if (!images.is_open()
        && !labels.is_open())
    {
        throw FileOpeningFailed();
    }
    unsigned char c;
    int shift = 0;

    for (int i = 0; !images.eof(); i++)
    {
        const vector<float> imageTemp;
        sets[set].data.push_back(imageTemp);
        sets[set].data.back().reserve(this->sizeOfData);
        if (!images.eof())
            for (int j = 0; !images.eof() && j < this->sizeOfData;)
            {
                c = images.get();

                if (shift > 15)
                {
                    const float value = static_cast<int>(c) / 255.0f * 2.0f - 1.0f;
                    sets[set].data.back().push_back(value);
                    j++;
                }
                else
                    shift ++;
            }
    }
    images.close();
    labels.close();
}

void Mnist::readSet(const set set, ifstream& images, ifstream& labels)
{
    if (!images.is_open()
        && !labels.is_open())
    {
        throw FileOpeningFailed();
    }
    int i;
    unsigned char c;

    for (i = 0; !labels.eof(); i++)
    {
        c = labels.get();

        const vector<float> labelsTemp(10, 0);
        sets[set].labels.push_back(labelsTemp);

        if (!labels.eof())
            sets[set].labels.back()[c] = 1.0;
        else
            sets[set].labels.resize(sets[set].labels.size() - 1);
    }
    int shift = 0;

    for (i = 0; !images.eof(); i++)
    {
        const vector<float> imageTemp;
        sets[set].data.push_back(imageTemp);
        sets[set].data.back().reserve(this->sizeOfData);
        if (!images.eof())
            for (int j = 0; !images.eof() && j < this->sizeOfData;)
            {
                c = images.get();

                if (shift > 15)
                {
                    const float value = static_cast<int>(c) / 255.0f * 2.0f - 1.0f;
                    sets[set].data.back().push_back(value);
                    j++;
                }
                else
                    shift ++;
            }
    }
    images.close();
    labels.close();
}
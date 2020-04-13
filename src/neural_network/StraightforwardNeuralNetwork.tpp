template <>
inline void StraightforwardNeuralNetwork::startTraining(Data& data)
{
    if (typeid(data) == typeid(DataForRegression))
    {
        this->startTraining(dynamic_cast<DataForRegression&>(data));
    }
    if (typeid(data) == typeid(DataForClassification))
    {
        this->startTraining(dynamic_cast<DataForClassification&>(data));
    }
    if (typeid(data) == typeid(DataForMultipleClassification))
    {
        this->startTraining(dynamic_cast<DataForMultipleClassification&>(data));
    }
}

template <typename TData>
void StraightforwardNeuralNetwork::startTraining(TData& data)
{
    internal::log<complete>("Start training");
    if (!this->validData(data))
        throw std::runtime_error("Data has not the same format as the neural network");
    this->stopTraining();
    this->isIdle = false;
    internal::log<complete>("Start a new thread");
    this->thread = std::thread(&StraightforwardNeuralNetwork::train<TData>, this, std::ref(data));
}


template <typename TData>
void StraightforwardNeuralNetwork::train(TData& data)
{
    this->numberOfTrainingsBetweenTwoEvaluations = data.sets[training].size;
    this->wantToStopTraining = false;

    this->evaluate(data);

    for (this->numberOfIteration = 0; !this->wantToStopTraining; this->numberOfIteration++)
    {
        internal::log<minimal>("Iteration: " + std::to_string(this->numberOfIteration));
        
        data.shuffle();

        for (this->currentIndex = 0; currentIndex < this->numberOfTrainingsBetweenTwoEvaluations && !this->wantToStopTraining;
            this->currentIndex ++)
        {
            this->trainOnce(data.getTrainingData(this->currentIndex),
                            data.getTrainingOutputs(this->currentIndex));
        }
        this->evaluate(data);
    }
}

template <>
inline void StraightforwardNeuralNetwork::evaluate(Data& data)
{
    if (typeid(data) == typeid(DataForRegression))
    {
        this->evaluate(dynamic_cast<DataForRegression&>(data));
    }
    if (typeid(data) == typeid(DataForClassification))
    {
        this->evaluate(dynamic_cast<DataForClassification&>(data));
    }
    if (typeid(data) == typeid(DataForMultipleClassification))
    {
        this->evaluate(dynamic_cast<DataForMultipleClassification&>(data));
    }
}

template <typename TData>
void StraightforwardNeuralNetwork::evaluate(TData& data)
{
    this->startTesting();
    for (currentIndex = 0; currentIndex < data.sets[testing].size; currentIndex++)
    {
        if (this->wantToStopTraining)
            return;
        this->evaluateOnce(data);
    }
    this->stopTesting();
    if (this->autoSaveWhenBetter && this->globalClusteringRateIsBetterThanPreviously)
    {
            this->saveAs(autoSaveFilePath);
    }
}

inline
void StraightforwardNeuralNetwork::evaluateOnce(DataForRegression& data)
{
    this->evaluateOnceForRegression(
        data.getTestingData(this->currentIndex),
        data.getTestingOutputs(this->currentIndex), data.getValue());
}

inline
void StraightforwardNeuralNetwork::evaluateOnce(DataForClassification& data)
{
    this->evaluateOnceForClassification(
        data.getTestingData(this->currentIndex),
        data.getTestingLabel(this->currentIndex));
}

inline
void StraightforwardNeuralNetwork::evaluateOnce(DataForMultipleClassification& data)
{
    this->evaluateOnceForMultipleClassification(
        data.getTestingData(this->currentIndex),
        data.getTestingOutputs(this->currentIndex), data.getValue());
}

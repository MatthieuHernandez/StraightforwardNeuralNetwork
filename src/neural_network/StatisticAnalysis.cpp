#include "StatisticAnalysis.hpp"

#include <cmath>

namespace snn::internal
{
void StatisticAnalysis::initialize(int numberOfCluster)
{
    clusters.resize(numberOfCluster);
    this->startTesting();
}

void StatisticAnalysis::setResultsAsNan()
{
    for (auto& c : clusters)
    {
        c.truePositive = NAN;
        c.trueNegative = NAN;
        c.falsePositive = NAN;
        c.falseNegative = NAN;
        c.totalError = NAN;
    }
    numberOfDataWellClassified = NAN;
    numberOfDataMisclassified = NAN;

    this->globalClusteringRate = NAN;
    this->weightedClusteringRate = NAN;
    this->f1Score = NAN;
    this->meanAbsoluteError = NAN;
    this->rootMeanSquaredError = NAN;
}

void StatisticAnalysis::startTesting()
{
    for (auto& c : clusters)
    {
        c.truePositive = 0;
        c.trueNegative = 0;
        c.falsePositive = 0;
        c.falseNegative = 0;
        c.totalError = 0;
    }
    numberOfDataWellClassified = 0;
    numberOfDataMisclassified = 0;
}

void StatisticAnalysis::stopTesting()
{
    const float newGlobalClusteringRate = this->computeGlobalClusteringRate();
    const float newWeightedClusteringRate = this->computeWeightedClusteringRate();
    const float newF1Score = this->computeF1Score();
    const float newMeanAbsoluteError = this->computeMeanAbsoluteError();
    const float newRootMeanSquaredError = this->computeRootMeanSquaredError();

    this->globalClusteringRateIsBetterThanMax = newGlobalClusteringRate > this->globalClusteringRateMax;
    if (this->globalClusteringRateIsBetterThanMax) this->globalClusteringRateMax = newGlobalClusteringRate;

    this->weightedClusteringRateIsBetterThanMax = newWeightedClusteringRate > this->weightedClusteringRateMax;
    if (this->weightedClusteringRateIsBetterThanMax) this->weightedClusteringRateMax = newWeightedClusteringRate;

    this->f1ScoreIsBetterThanMax = newF1Score > this->f1ScoreMax;
    if (this->f1ScoreIsBetterThanMax) this->f1Score = newF1Score;

    this->meanAbsoluteErrorIsBetterThanMin =
        newMeanAbsoluteError < this->meanAbsoluteErrorMin || this->meanAbsoluteErrorMin < 0;
    if (this->meanAbsoluteErrorIsBetterThanMin) this->meanAbsoluteErrorMin = newMeanAbsoluteError;

    this->rootMeanSquaredErrorIsBetterThanMin =
        newRootMeanSquaredError < this->rootMeanSquaredErrorMin || this->meanAbsoluteErrorMin < 0;
    if (this->rootMeanSquaredErrorIsBetterThanMin) this->rootMeanSquaredErrorMin = newMeanAbsoluteError;

    this->globalClusteringRate = newGlobalClusteringRate;
    this->weightedClusteringRate = newWeightedClusteringRate;
    this->f1Score = newF1Score;
    this->meanAbsoluteError = newMeanAbsoluteError;
    this->rootMeanSquaredError = newRootMeanSquaredError;
}

void StatisticAnalysis::evaluateOnceForRegression(const std::vector<float>& outputs,
                                                  const std::vector<float>& desiredOutputs, float precision)
{
    bool classifiedWell = true;
    for (size_t i = 0; i < clusters.size(); i++)
    {
        if (outputs[i] > desiredOutputs[i] + precision)
        {
            clusters[i].falsePositive++;
            classifiedWell = false;
        }
        else if (outputs[i] < desiredOutputs[i] - precision)
        {
            clusters[i].falseNegative++;
            classifiedWell = false;
        }
        else if (outputs[i] >= desiredOutputs[i])
        {
            clusters[i].trueNegative++;
        }
        else if (outputs[i] <= desiredOutputs[i])
        {
            clusters[i].trueNegative++;
        }

        clusters[i].totalError += abs(desiredOutputs[i] - outputs[i]);
    }
    if (classifiedWell)
    {
        numberOfDataWellClassified++;
    }
    else
    {
        numberOfDataMisclassified++;
    }
}

void StatisticAnalysis::evaluateOnceForMultipleClassification(const std::vector<float>& outputs,
                                                              const std::vector<float>& desiredOutputs, float separator)
{
    bool classifiedWell = true;
    for (size_t i = 0; i < clusters.size(); i++)
    {
        if (outputs[i] > separator && desiredOutputs[i] > separator)
        {
            clusters[i].truePositive++;
        }
        else if (outputs[i] <= separator && desiredOutputs[i] <= separator)
        {
            clusters[i].trueNegative++;
        }
        else if (outputs[i] > separator && desiredOutputs[i] <= separator)
        {
            clusters[i].falsePositive++;
            classifiedWell = false;
        }
        else if (outputs[i] <= separator && desiredOutputs[i] > separator)
        {
            clusters[i].falseNegative++;
            classifiedWell = false;
        }

        clusters[i].totalError += abs(desiredOutputs[i] - outputs[i]);
    }
    if (classifiedWell)
    {
        numberOfDataWellClassified++;
    }
    else
    {
        numberOfDataMisclassified++;
    }
}

void StatisticAnalysis::evaluateOnceForClassification(const std::vector<float>& outputs, int classNumber,
                                                      float separator)
{
    float maxOutputValue = -2;
    int maxOutputIndex = -1;

    for (int i = 0; i < static_cast<int>(clusters.size()); i++)
    {
        if (maxOutputValue < outputs[i])
        {
            maxOutputValue = outputs[i];
            maxOutputIndex = i;
        }
        if (i == classNumber)
        {
            clusters[i].totalError += abs(1 - outputs[i]);

            if (outputs[i] > separator)
            {
                clusters[i].truePositive++;
            }
            else
            {
                clusters[i].falseNegative++;
            }
        }
        else
        {
            clusters[i].totalError += abs(0 - outputs[i]);

            if (outputs[i] > separator)
            {
                clusters[i].falsePositive++;
            }
            else
            {
                clusters[i].trueNegative++;
            }
        }
    }
    if (maxOutputIndex == classNumber)
    {
        numberOfDataWellClassified++;
    }
    else
    {
        numberOfDataMisclassified++;
    }
}

auto StatisticAnalysis::computeGlobalClusteringRate() const -> float
{
    return numberOfDataWellClassified / (numberOfDataWellClassified + numberOfDataMisclassified);
}

auto StatisticAnalysis::computeWeightedClusteringRate() const -> float
{
    float result = 0;
    for (const auto& c : clusters)
    {
        if (c.truePositive > 0) result += c.truePositive / (c.truePositive + c.falseNegative);
    }
    return result / clusters.size();
}

auto StatisticAnalysis::computeF1Score() const -> float
{
    float result = 0;
    for (const auto& c : clusters)
    {
        if (c.truePositive > 0)
        {
            const float precision = c.truePositive / (c.truePositive + c.falsePositive);
            const float recall = c.truePositive / (c.truePositive + c.falseNegative);
            result += (precision * recall) / (precision + recall);
        }
    }
    return 2.0F * result / clusters.size();
}

auto StatisticAnalysis::computeMeanAbsoluteError() const -> float
{
    float totalError = 0;
    for (const auto& c : clusters)
    {
        totalError += c.totalError;
    }
    return totalError / (numberOfDataWellClassified + numberOfDataMisclassified);
}

auto StatisticAnalysis::computeRootMeanSquaredError() const -> float
{
    float totalError = 0;
    for (const auto& c : clusters)
    {
        totalError += c.totalError * c.totalError;
    }
    const float meanSquaredError = totalError / (numberOfDataWellClassified + numberOfDataMisclassified);
    return sqrt(meanSquaredError);
}

auto StatisticAnalysis::getGlobalClusteringRate() const -> float { return this->globalClusteringRate; }

auto StatisticAnalysis::getWeightedClusteringRate() const -> float { return this->weightedClusteringRate; }

auto StatisticAnalysis::getF1Score() const -> float { return this->f1Score; }

auto StatisticAnalysis::getMeanAbsoluteError() const -> float { return this->meanAbsoluteError; }

auto StatisticAnalysis::getRootMeanSquaredError() const -> float { return this->rootMeanSquaredError; }

auto StatisticAnalysis::getGlobalClusteringRateMax() const -> float { return this->globalClusteringRateMax; }

auto StatisticAnalysis::getWeightedClusteringRateMax() const -> float { return this->weightedClusteringRateMax; }

auto StatisticAnalysis::getF1ScoreMax() const -> float { return this->f1ScoreMax; }

auto StatisticAnalysis::getMeanAbsoluteErrorMin() const -> float { return this->meanAbsoluteErrorMin; }

auto StatisticAnalysis::getRootMeanSquaredErrorMin() const -> float { return this->rootMeanSquaredErrorMin; }

auto StatisticAnalysis::operator==(const StatisticAnalysis& sa) const -> bool
{
    return this->clusters == sa.clusters && this->numberOfDataWellClassified == sa.numberOfDataWellClassified &&
           this->numberOfDataMisclassified == sa.numberOfDataMisclassified &&
           this->globalClusteringRate == sa.globalClusteringRate &&
           this->weightedClusteringRate == sa.weightedClusteringRate && this->f1Score == sa.f1Score &&
           this->meanAbsoluteError == sa.meanAbsoluteError && this->rootMeanSquaredError == sa.rootMeanSquaredError &&
           this->globalClusteringRateMax == sa.globalClusteringRateMax &&
           this->weightedClusteringRateMax == sa.weightedClusteringRateMax && this->f1ScoreMax == sa.f1ScoreMax &&
           this->meanAbsoluteErrorMin == sa.meanAbsoluteErrorMin &&
           this->rootMeanSquaredErrorMin == sa.rootMeanSquaredErrorMin &&
           this->globalClusteringRateIsBetterThanMax == sa.globalClusteringRateIsBetterThanMax &&
           this->weightedClusteringRateIsBetterThanMax == sa.weightedClusteringRateIsBetterThanMax &&
           this->f1ScoreIsBetterThanMax == sa.f1ScoreIsBetterThanMax &&
           this->meanAbsoluteErrorIsBetterThanMin == sa.meanAbsoluteErrorIsBetterThanMin &&
           this->rootMeanSquaredErrorIsBetterThanMin == sa.rootMeanSquaredErrorIsBetterThanMin;
}

auto StatisticAnalysis::operator!=(const StatisticAnalysis& sa) const -> bool { return !(*this == sa); }
}  // namespace snn::internal

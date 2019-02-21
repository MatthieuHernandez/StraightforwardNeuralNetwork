#include "StatisticAnalysis.h"


StatisticAnalysis::StatisticAnalysis(int numberOfCluster)
{
	clusters.resize(numberOfCluster);
	this->startTesting();
}

void StatisticAnalysis::startTesting()
{
	for (auto& c : clusters)
	{
		c.truePositive = 0;
		c.trueNegative = 0;
		c.falsePositive = 0;
		c.falseNegative = 0;
	}
	numberOfDataWellClassified = 0;
	numberOfDataMisclassified = 0;
}

void StatisticAnalysis::insertTestWithPrecision(const std::vector<float>& outputs,
                                                const std::vector<float>& desiredOutputs,
                                                float precision)
{
	bool classifiedWell = true;
	for (int i = 0; i < clusters.size(); i++)
	{
		if (outputs[i] > desiredOutputs[i] + precision)
		{
			clusters[i].falsePositive ++;
			classifiedWell = false;
		}
		else if (outputs[i] <= desiredOutputs[i] - precision)
		{
			clusters[i].falseNegative ++;
			classifiedWell = false;
		}
		else if (outputs[i] > desiredOutputs[i])
		{
			clusters[i].trueNegative ++;
		}
		else if (outputs[i] <= desiredOutputs[i])
		{
			clusters[i].trueNegative ++;
		}
	}
	if (classifiedWell)
		numberOfDataWellClassified++;
	else
		numberOfDataMisclassified++;
}

void StatisticAnalysis::insertTestSeparateByValue(const std::vector<float>& outputs,
                                                  const std::vector<float>& desiredOutputs,
                                                  float separator)
{
	bool classifiedWell = true;
	for (int i = 0; i < clusters.size(); i++)
	{
		if (outputs[i] > separator && desiredOutputs[i] > separator)
		{
			clusters[i].truePositive ++;
		}
		else if (outputs[i] <= separator && desiredOutputs[i] <= separator)
		{
			clusters[i].trueNegative ++;
		}
		else if (outputs[i] > separator && desiredOutputs[i] <= separator)
		{
			clusters[i].falsePositive ++;
			classifiedWell = false;
		}
		else if (outputs[i] <= separator && desiredOutputs[i] > separator)
		{
			clusters[i].falseNegative ++;
			classifiedWell = false;
		}
	}
	if (classifiedWell)
		numberOfDataWellClassified++;
	else
		numberOfDataMisclassified++;
}

void StatisticAnalysis::insertTestWithClassNumber(const std::vector<float>& outputs, int classNumber)
{
	float separator = 0.5f;
	float maxOutputValue = -1;
	int maxOutputIndex = -1;
	for (int i = 0; i < clusters.size(); i++)
	{
		if (maxOutputValue < outputs[i])
		{
			maxOutputValue = outputs[i];
			maxOutputIndex = i;
		}
		if (i == classNumber && outputs[i] > separator)
		{
			clusters[i].truePositive ++;
		}
		else if (i == classNumber && outputs[i] <= separator)
		{
			clusters[i].falseNegative ++;
		}
		else if (outputs[i] > separator)
		{
			clusters[i].falsePositive ++;
		}
		else if (outputs[i] <= separator)
		{
			clusters[i].trueNegative ++;
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

float StatisticAnalysis::getGlobalClusteringRate() const
{
	return numberOfDataWellClassified / (numberOfDataWellClassified + numberOfDataMisclassified);
}

float StatisticAnalysis::getWeightedClusteringRate() const
{
	float weightedClusteringRate = 0;
	for (const auto c : clusters)
	{
		const float numerator = c.truePositive;
		const float denominator = c.truePositive + c.falsePositive;

		if (numerator > 0)
			weightedClusteringRate += numerator / denominator;
	}
	return weightedClusteringRate / clusters.size();
}

float StatisticAnalysis::getF1Score() const
{
	float f1Score = 0;

	for (const auto c : clusters)
	{
		if (c.truePositive > 0)
		{
			const float precision = c.truePositive / (c.truePositive + c.falsePositive);
			const float recall = c.truePositive / (c.truePositive + c.falseNegative);
			f1Score += (precision * recall) / (precision + recall);
		}
	}
	return 2.0f * f1Score / clusters.size();
}

bool StatisticAnalysis::operator==(const StatisticAnalysis& sa) const
{
	return this->clusters == sa.clusters
		&& this->numberOfDataWellClassified == sa.numberOfDataWellClassified 
		&& this->numberOfDataMisclassified == sa.numberOfDataMisclassified;
}

bool StatisticAnalysis::operator!=(const StatisticAnalysis& sa) const
{
	return !this->operator==(sa);
}

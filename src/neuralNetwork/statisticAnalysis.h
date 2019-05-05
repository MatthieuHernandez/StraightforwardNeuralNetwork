#pragma once
#include <vector>
#include <boost/serialization/access.hpp>

struct binaryClassification
{
	float truePositive{};
	float trueNegative{};
	float falsePositive{};
	float falseNegative{};

	bool operator==(const binaryClassification&) const
	{
		return true;
	};

	template <typename Archive>
	void serialize(Archive& ar, unsigned)
	{
		ar & truePositive;
		ar & trueNegative;
		ar & falsePositive;
		ar & falseNegative;
	}
};

class StatisticAnalysis
{
private:

	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, unsigned version);

	std::vector<binaryClassification> clusters;
	float numberOfDataWellClassified{};
	float numberOfDataMisclassified{};

	float globalClusteringRate = 0;
	float weightedClusteringRate = 0;
	float f1Score = 0;

protected:

	void insertTestWithPrecision(const std::vector<float>& outputs, const std::vector<float>& desiredOutputs,
	                             float precision);
	void insertTestSeparateByValue(const std::vector<float>& outputs, const std::vector<float>& desiredOutputs,
	                               float separator);
	void insertTestWithClassNumber(const std::vector<float>& outputs, int classNumber);

	float computeGlobalClusteringRate();
	float computeWeightedClusteringRate();
	float computeF1Score();

	StatisticAnalysis() = default;
	StatisticAnalysis(int numberOfCluster);
	virtual ~StatisticAnalysis() = default;

	void startTesting();
	void stopTesting();

public:
	virtual float getGlobalClusteringRate() const;
	virtual float getWeightedClusteringRate() const;
	virtual float getF1Score() const;

	StatisticAnalysis& operator=(const StatisticAnalysis& sa) = default;
	bool operator==(const StatisticAnalysis& sa) const;
	bool operator!=(const StatisticAnalysis& sa) const;
};

template <class Archive>
void StatisticAnalysis::serialize(Archive& ar, unsigned)
{
	ar & this->clusters;
	ar & this->numberOfDataWellClassified;
	ar & this->numberOfDataMisclassified;
}

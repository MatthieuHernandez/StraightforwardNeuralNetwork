#pragma once
#include <vector>
#include <boost/serialization/access.hpp>

namespace snn::internal
{
    struct binaryClassification
    {
        float truePositive{};
        float trueNegative{};
        float falsePositive{};
        float falseNegative{};
        float totalError{};

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
        float numberOfDataWellClassified;
        float numberOfDataMisclassified;

        float globalClusteringRate = -1.0f;
        float weightedClusteringRate = -1.0f;
        float f1Score = -1.0f;
        float meanAbsoluteError = -1.0f;
        float rootMeanSquaredError = -1.0f;

        float globalClusteringRateMax = -1.0f;
        float weightedClusteringRateMax = -1.0f;
        float f1ScoreMax = -1.0f;
        float meanAbsoluteErrorMin = -1.0f;
        float rootMeanSquaredErrorMin = -1.0f;

        [[nodiscard]] float computeGlobalClusteringRate() const;
        [[nodiscard]] float computeWeightedClusteringRate() const;
        [[nodiscard]] float computeF1Score() const;
        [[nodiscard]] float computeMeanAbsoluteError() const;
        [[nodiscard]] float computeRootMeanSquaredError() const;

    protected:
        StatisticAnalysis() = default;
        StatisticAnalysis(const StatisticAnalysis&) = default;
        virtual ~StatisticAnalysis() = default;

        void initialize(int numberOfCluster);

        void setResultsAsNan();

        void evaluateOnceForRegression(const std::vector<float>& outputs, 
                                       const std::vector<float>& desiredOutputs,
                                       float precision);
        void evaluateOnceForMultipleClassification(const std::vector<float>& outputs,
                                                   const std::vector<float>& desiredOutputs,
                                                   float separator);
        void evaluateOnceForClassification(const std::vector<float>& outputs,
                                           int classNumber,
                                           float separator);

        void startTesting();
        void stopTesting();

        bool globalClusteringRateIsBetterThanMax = false;
        bool weightedClusteringRateIsBetterThanMax = false;
        bool f1ScoreIsBetterThanMax = false;
        bool meanAbsoluteErrorIsBetterThanMin = false;
        bool rootMeanSquaredErrorIsBetterThanMin = false;

    public:
        float getGlobalClusteringRate() const;
        float getWeightedClusteringRate() const;
        float getF1Score() const;
        float getMeanAbsoluteError() const;
        float getRootMeanSquaredError() const;

        float getGlobalClusteringRateMax() const;
        float getWeightedClusteringRateMax() const;
        float getF1ScoreMax() const;
        float getMeanAbsoluteErrorMin() const;
        float getRootMeanSquaredErrorMin() const;

        bool operator==(const StatisticAnalysis& sa) const;
        bool operator!=(const StatisticAnalysis& sa) const;
    };

    template <class Archive>
    void StatisticAnalysis::serialize(Archive& ar, unsigned)
    {
        ar & this->clusters;
        ar & this->numberOfDataWellClassified;
        ar & this->numberOfDataMisclassified;
        ar & this->globalClusteringRate;
        ar & this->weightedClusteringRate;
        ar & this->f1Score;
        ar & this->meanAbsoluteError;
        ar & this->rootMeanSquaredError;
        ar & this->globalClusteringRateMax;
        ar & this->weightedClusteringRateMax;
        ar & this->f1ScoreMax;
        ar & this->meanAbsoluteErrorMin;
        ar & this->rootMeanSquaredErrorMin;
        ar & this->globalClusteringRateIsBetterThanMax;
        ar & this->weightedClusteringRateIsBetterThanMax;
        ar & this->f1ScoreIsBetterThanMax;
        ar & this->meanAbsoluteErrorIsBetterThanMin;
        ar & this->rootMeanSquaredErrorIsBetterThanMin;
    }
}

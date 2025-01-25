#pragma once
#include <boost/serialization/access.hpp>
#include <vector>

namespace snn::internal
{
struct binaryClassification
{
        float truePositive{};
        float trueNegative{};
        float falsePositive{};
        float falseNegative{};
        float totalError{};

        auto operator==(const binaryClassification&) const -> bool { return true; };

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

        [[nodiscard]] auto computeGlobalClusteringRate() const -> float;
        [[nodiscard]] auto computeWeightedClusteringRate() const -> float;
        [[nodiscard]] auto computeF1Score() const -> float;
        [[nodiscard]] auto computeMeanAbsoluteError() const -> float;
        [[nodiscard]] auto computeRootMeanSquaredError() const -> float;

    protected:
        StatisticAnalysis() = default;
        StatisticAnalysis(const StatisticAnalysis&) = default;
        virtual ~StatisticAnalysis() = default;

        void initialize(int numberOfCluster);

        void setResultsAsNan();

        void evaluateOnceForRegression(const std::vector<float>& outputs, const std::vector<float>& desiredOutputs,
                                       float precision);
        void evaluateOnceForMultipleClassification(const std::vector<float>& outputs,
                                                   const std::vector<float>& desiredOutputs, float separator);
        void evaluateOnceForClassification(const std::vector<float>& outputs, int classNumber, float separator);

        void startTesting();
        void stopTesting();

        bool globalClusteringRateIsBetterThanMax = false;
        bool weightedClusteringRateIsBetterThanMax = false;
        bool f1ScoreIsBetterThanMax = false;
        bool meanAbsoluteErrorIsBetterThanMin = false;
        bool rootMeanSquaredErrorIsBetterThanMin = false;

    public:
        auto getGlobalClusteringRate() const -> float;
        auto getWeightedClusteringRate() const -> float;
        auto getF1Score() const -> float;
        auto getMeanAbsoluteError() const -> float;
        auto getRootMeanSquaredError() const -> float;

        auto getGlobalClusteringRateMax() const -> float;
        auto getWeightedClusteringRateMax() const -> float;
        auto getF1ScoreMax() const -> float;
        auto getMeanAbsoluteErrorMin() const -> float;
        auto getRootMeanSquaredErrorMin() const -> float;

        auto operator==(const StatisticAnalysis& sa) const -> bool;
        auto operator!=(const StatisticAnalysis& sa) const -> bool;
};

template <class Archive>
void StatisticAnalysis::serialize(Archive& ar, unsigned)
{
    ar& this->clusters;
    ar& this->numberOfDataWellClassified;
    ar& this->numberOfDataMisclassified;
    ar& this->globalClusteringRate;
    ar& this->weightedClusteringRate;
    ar& this->f1Score;
    ar& this->meanAbsoluteError;
    ar& this->rootMeanSquaredError;
    ar& this->globalClusteringRateMax;
    ar& this->weightedClusteringRateMax;
    ar& this->f1ScoreMax;
    ar& this->meanAbsoluteErrorMin;
    ar& this->rootMeanSquaredErrorMin;
    ar& this->globalClusteringRateIsBetterThanMax;
    ar& this->weightedClusteringRateIsBetterThanMax;
    ar& this->f1ScoreIsBetterThanMax;
    ar& this->meanAbsoluteErrorIsBetterThanMin;
    ar& this->rootMeanSquaredErrorIsBetterThanMin;
}
}  // namespace snn::internal

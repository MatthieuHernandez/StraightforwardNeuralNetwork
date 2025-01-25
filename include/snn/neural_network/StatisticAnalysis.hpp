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
        void serialize(Archive& archive, unsigned)
        {
            archive & truePositive;
            archive & trueNegative;
            archive & falsePositive;
            archive & falseNegative;
        }
};

class StatisticAnalysis
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

        std::vector<binaryClassification> clusters;
        float numberOfDataWellClassified;
        float numberOfDataMisclassified;

        float globalClusteringRate = -1.0F;
        float weightedClusteringRate = -1.0F;
        float f1Score = -1.0F;
        float meanAbsoluteError = -1.0F;
        float rootMeanSquaredError = -1.0F;

        float globalClusteringRateMax = -1.0F;
        float weightedClusteringRateMax = -1.0F;
        float f1ScoreMax = -1.0F;
        float meanAbsoluteErrorMin = -1.0F;
        float rootMeanSquaredErrorMin = -1.0F;

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
void StatisticAnalysis::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    archive& this->clusters;
    archive& this->numberOfDataWellClassified;
    archive& this->numberOfDataMisclassified;
    archive& this->globalClusteringRate;
    archive& this->weightedClusteringRate;
    archive& this->f1Score;
    archive& this->meanAbsoluteError;
    archive& this->rootMeanSquaredError;
    archive& this->globalClusteringRateMax;
    archive& this->weightedClusteringRateMax;
    archive& this->f1ScoreMax;
    archive& this->meanAbsoluteErrorMin;
    archive& this->rootMeanSquaredErrorMin;
    archive& this->globalClusteringRateIsBetterThanMax;
    archive& this->weightedClusteringRateIsBetterThanMax;
    archive& this->f1ScoreIsBetterThanMax;
    archive& this->meanAbsoluteErrorIsBetterThanMin;
    archive& this->rootMeanSquaredErrorIsBetterThanMin;
}
}  // namespace snn::internal

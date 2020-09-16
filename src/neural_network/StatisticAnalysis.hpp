#pragma once
#include <limits>
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
        float meanAbsoluteErrorMax = -1.0f;
        float rootMeanSquaredErrorMax = -1.0f;
        
        float computeGlobalClusteringRate();
        float computeWeightedClusteringRate();
        float computeF1Score();
        float computeMeanAbsoluteError();
        float computeRootMeanSquaredError();

    protected:
        StatisticAnalysis() = default;
        StatisticAnalysis(const StatisticAnalysis&) = default;
        virtual ~StatisticAnalysis() = default;

        void initialize(int numberOfCluster);

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

        bool globalClusteringRateIsBetterThanPreviously = false;
        bool weightedClusteringRateIsBetterThanPreviously = false;
        bool f1ScoreIsBetterThanPreviously = false;
        bool meanAbsoluteErrorIsBetterThanPreviously = false;
        bool rootMeanSquaredErrorIsBetterThanPreviously = false;

    public:
        float getGlobalClusteringRate() const;
        float getWeightedClusteringRate() const;
        float getF1Score() const;
        float getMeanAbsoluteError() const;
        float getRootMeanSquaredError() const;

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
}

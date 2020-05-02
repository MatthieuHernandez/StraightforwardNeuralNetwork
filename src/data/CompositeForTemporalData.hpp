#pragma once
#include "TemporalComposite.hpp"

namespace snn::internal
{
    class CompositeForTemporalData : public TemporalComposite
    {
    private:
        std::vector<int> indexesForShuffles;
    public:
        CompositeForTemporalData(snn::Set sets[2]);

        void shuffle() override;

        [[nodiscard]] bool isFirstTrainingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool isFirstTestingDataOfTemporalSequence(int index) const override;
        [[nodiscard]] bool needToLearnOnTrainingData(int index) const override;
        [[nodiscard]] bool needToEvaluateOnTestingData(int index) const override;

        [[nodiscard]] int isValid() override;
    };
}

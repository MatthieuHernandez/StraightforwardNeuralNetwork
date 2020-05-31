#pragma once
#include "ProblemComposite.hpp"

namespace snn::internal
{
    class CompositeForClassification : public ProblemComposite
    {
    public:
        CompositeForClassification(Set sets[2]);

        [[nodiscard]] int isValid() override;

        [[nodiscard]] const std::vector<float>& getTestingOutputs(const int) const override;
        [[nodiscard]] int getTrainingLabel(const int) override;
        [[nodiscard]] int getTestingLabel(const int) override;
    };
} 
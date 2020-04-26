#pragma once
#include "ProblemComposite.hpp"

namespace snn::internal
{
    class CompositeForRegression : public ProblemComposite
    {
    public:
        CompositeForRegression(Set sets[2]);

        [[nodiscard]] int isValid() override;

        [[nodiscard]] const std::vector<float>& getTestingOutputs(const int) override;
        [[nodiscard]] int getTrainingLabel(const int) override;
        [[nodiscard]] int getTestingLabel(const int) override;
    };
}

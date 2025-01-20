#pragma once
#include "ProblemComposite.hpp"

namespace snn::internal
{
class CompositeForRegression : public ProblemComposite
{
    public:
        CompositeForRegression(Set sets[2], int numberOfLabels);

        [[nodiscard]] auto isValid() const -> ErrorType override;

        [[nodiscard]] const std::vector<float>& getTestingOutputs(const int) const override;
        [[nodiscard]] int getTrainingLabel(const int) override;
        [[nodiscard]] int getTestingLabel(const int) override;
};
}  // namespace snn::internal

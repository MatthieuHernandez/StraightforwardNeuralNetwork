#pragma once
#include "../tools/Error.hpp"
#include "Set.hpp"

namespace snn::internal
{
class ProblemComposite
{
    private:
        std::vector<float> batchedLabels{};

    protected:
        const int numberOfLabels;
        Set* sets;

    public:
        ProblemComposite(Set sets[2], int numberOfLabels);
        virtual ~ProblemComposite() = default;

        [[nodiscard]] virtual auto isValid() const -> ErrorType;

        [[nodiscard]] virtual int getTrainingLabel(int index) = 0;
        [[nodiscard]] virtual int getTestingLabel(int index) = 0;

        [[nodiscard]] const std::vector<float>& getTrainingOutputs(int index, int batchSize);
        [[nodiscard]] virtual const std::vector<float>& getTestingOutputs(int index) const = 0;
};
}  // namespace snn::internal

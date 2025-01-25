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

        [[nodiscard]] virtual auto getTrainingLabel(int index) -> int = 0;
        [[nodiscard]] virtual auto getTestingLabel(int index) -> int = 0;

        [[nodiscard]] auto getTrainingOutputs(int index, int batchSize) -> const std::vector<float>&;
        [[nodiscard]] virtual auto getTestingOutputs(int index) const -> const std::vector<float>& = 0;
};
}  // namespace snn::internal

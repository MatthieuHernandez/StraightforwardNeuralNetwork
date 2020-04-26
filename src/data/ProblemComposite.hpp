#pragma once
#include "Set.hpp"

namespace snn::internal
{
    class ProblemComposite
    {
    protected:
        Set sets[2];

    public:
        ProblemComposite(Set sets[2]);

        [[nodiscard]] virtual int isValid();

        [[nodiscard]] virtual int getTrainingLabel(int index) = 0;
        [[nodiscard]] virtual int getTestingLabel(int index) = 0;

        [[nodiscard]] const std::vector<float>& getTrainingOutputs(int index);
        [[nodiscard]] virtual const std::vector<float>& getTestingOutputs(int index) = 0;
    };
}

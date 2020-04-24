#pragma once
#include "Data.hpp"

namespace snn::internal
{
    class ProblemComposite
    {
    protected:
        Set sets[2];

    public:
        ProblemComposite(Set set[2]);

        [[nodiscard]] virtual int isValid();

        [[nodiscard]] virtual const std::vector<float>& getTrainingData(int index);
        [[nodiscard]] virtual const std::vector<float>& getTestingData(int index);

        [[nodiscard]] virtual int getTrainingLabel(const int) { throw std::exception(); }
        [[nodiscard]] virtual int getTestingLabel(const int) { throw std::exception(); }

        [[nodiscard]] virtual const std::vector<float>& getTrainingOutputs(const int index);
        [[nodiscard]] virtual const std::vector<float>& getTestingOutputs(const int) = 0;
    };
}

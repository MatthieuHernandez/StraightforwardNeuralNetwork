#pragma once
#include <cstdint>
#include <vector>

#include "../tools/Tools.hpp"

namespace snn
{
enum class setType : uint8_t
{
    testing = 0,
    training = 1
};

namespace internal
{
struct Set
{
        explicit Set(setType t)
            : type(t)
        {
        }
        setType type;
        int index{0};
        int size{0};  // number of data inside set
        int numberOfTemporalSequence{};
        vector2D<float> inputs;
        vector2D<float> labels;
        std::vector<int> shuffledIndexes;
        std::vector<bool> areFirstDataOfTemporalSequence;
        std::vector<bool> needToTrainOnData;
        std::vector<bool> needToEvaluateOnData;
};

struct Data
{
        explicit Data()
            : training(setType::training),
              testing(snn::setType::testing)
        {
        }
        Set training;
        Set testing;
};
}  // namespace internal
}  // namespace snn

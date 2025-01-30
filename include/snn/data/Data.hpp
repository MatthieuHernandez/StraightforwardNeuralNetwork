#pragma once
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
        setType type;
        int index{0};
        size_t size{0};  // number of data inside set
        int numberOfTemporalSequence{};
        vector2D<float> inputs;
        vector2D<float> labels;
        std::vector<int> shuffledIndexes;
        std::vector<bool> areFirstDataOfTemporalSequence;
        std::vector<bool> needToTrainOnData;
        std::vector<bool> needToEvaluateOnData;
} __attribute__((packed, aligned(128)));  // NOLINT(*magic-numbers)

struct Data
{
        Set training;
        Set testing;
} __attribute__((packed, aligned(512)));  // NOLINT(*magic-numbers)
}  // namespace internal
}  // namespace snn

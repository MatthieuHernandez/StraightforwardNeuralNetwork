#pragma once
#include <vector>
#include "../tools/Tools.hpp"

namespace snn
{
    enum set
    {
        testing = 0,
        training = 1
    };
    
    struct Set
    {
        int index{0};
        int size{0}; // number of data inside set
        vector2D<float> inputs{};
        vector2D<float> labels{};
        std::vector<int> indexesToShuffle{};
        std::vector<bool> areFirstDataOfTemporalSequence{};
        std::vector<bool> needToLearnData{};
    };
}

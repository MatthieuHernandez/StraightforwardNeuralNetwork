#include "Tools.hpp"

#include <algorithm>

namespace snn::tools
{
int randomBetween(const int min, const int max)  // [min; max[
{
    std::uniform_real_distribution<> dist(min, max);
    return static_cast<int>(dist(rng));
}

float randomBetween(const float min, const float max)
{
    std::uniform_real_distribution<> dist(min, max);
    return static_cast<float>(dist(rng));
}

std::vector<float> randomVector(const float min, const float max, const size_t size)
{
    std::uniform_real_distribution<> dist(min, max);
    std::vector<float> vector(size);
    std::ranges::generate(vector, [&] { return static_cast<float>(dist(rng)); });
    return vector;
}
}  // namespace snn::tools
#include <chrono>
#include <cstdlib>
#include <iostream>
#include "Tools.hpp"

using namespace std;
using namespace snn;
using namespace internal;


int Tools::randomBetween(const int min, const int max) // [min; max[
{
    return rand() % (max - min) + min;
}

float Tools::randomBetween(const float min, const float max)
{
    return rand() / static_cast<float>(RAND_MAX) * (max - min) + min;
}

std::string Tools::toString(std::chrono::milliseconds duration)
{
    std::string str;

    if (duration.count() > 3600000)
        return std::to_string(duration.count() / 3600000) + "h " + std::to_string(duration.count() / 60000) +
            "min";

    if (duration.count() > 60000)
        return std::to_string(duration.count() / 60000) + "min " + std::to_string(duration.count() / 1000) + "s";

    if (duration.count() > 1000)
        return std::to_string(duration.count() / 1000) + "s";

    if (duration.count() > 0)
        return std::to_string(duration.count()) + "ms";

    return "";
}

#pragma once
#include <iostream>
#include <vector>

namespace snn
{
    template <typename T>
    using vector2D = std::vector<std::vector<T>>;

    template <typename T>
    using vector3D = std::vector<std::vector<std::vector<T>>>;

    enum logLevel
    {
        none = 0,
        minimal = 1,
        complete = 2
    };

    static constexpr logLevel verbose = none;
}

namespace snn::internal
{
    class Tools
    {
    public:
        static int randomBetween(const int min, const int max);

        static float randomBetween(const float min, const float max);

        template <typename T>
        static T getMinValue(std::vector<T> vector);

        template <typename T>
        static T getMaxValue(std::vector<T> vector);

    };

    template<logLevel T, typename... Targs>
    constexpr void log(Targs&&... messages)
    {
        if constexpr (T > none && T <= verbose)
            (std::cout << ... << messages) << std::endl;
    }
}
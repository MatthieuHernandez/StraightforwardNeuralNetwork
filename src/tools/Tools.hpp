#pragma once
#include <chrono>
#include <iostream>
#include <string>
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

    enum class debugLevel
    {
        none = 0,
        debug = 1
    };

    static constexpr logLevel logVerbose = minimal;
    static constexpr debugLevel debugVerbose = debugLevel::debug;
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

        static std::string toString(std::chrono::milliseconds duration);

    };

    template <logLevel T, typename... Targs>
    constexpr void log(Targs&&... messages)
    {
        if constexpr (T > none && T <= logVerbose)
            (std::cout << ... << messages) << std::endl;
    }

    template <typename... Targs>
    constexpr void debug(bool condition, Targs&&... messages)
    {
        if constexpr (debugVerbose == debugLevel::debug)
            if(condition)
                (std::cout << ... << messages) << std::endl;
    }

    template <typename T>
    std::vector<T> flatten(const std::vector<std::vector<T>>& vector2D)
    {
        std::vector<T> vector1D;
        size_t size = 0;
        for (const auto& v : vector2D)
            size += vector2D.size();
        vector1D.reserve(size);
        for (const auto& v : vector2D)
            vector1D.insert(vector1D.end(), v.begin(), v.end());
        return vector1D;
    }
}

#pragma once
#include <chrono>
#include <iostream>
#include <stdexcept>
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

    static constexpr logLevel verbose = none;
}

namespace snn::tools
{
    inline int randomBetween(const int min, const int max) // [min; max[
    {
        return rand() % (max - min) + min;
    }

    inline float randomBetween(const float min, const float max)
    {
        return rand() / static_cast<float>(RAND_MAX) * (max - min) + min;
    }

    inline std::string toString(std::chrono::milliseconds duration)
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

    template <typename T>
    static T getMinValue(std::vector<T> vector)
    {
        if (vector.size() > 1)
        {
            T minValue = vector[0];

            for (size_t i = 1; i < vector.size(); i++)
            {
                if (vector[i] < minValue)
                {
                    minValue = vector[i];
                }
            }
            return minValue;
        }
        throw std::runtime_error("Vector is empty");
    }

    template <typename T>
    static T getMaxValue(std::vector<T> vector)
    {
        if (vector.size() > 1)
        {
            T maxValue = vector[0];

            for (size_t i = 1; i < vector.size(); i++)
            {
                if (vector[i] > maxValue)
                {
                    maxValue = vector[i];
                }
            }
            return maxValue;
        }
        throw std::runtime_error("Vector is empty");
    }

    template <logLevel T, typename... Targs>
    constexpr void log(Targs&&... messages)
    {
        if constexpr (T > none && T <= verbose)
            (std::cout << ... << messages) << std::endl;
    }

    template <logLevel T, bool endLine, typename... Targs>
    constexpr void log(Targs&&... messages)
    {
        if constexpr (T > none && T <= verbose)
        {
            (std::cout << ... << messages);
            if constexpr (endLine)
                std::cout << std::endl;
        }
    }

    inline std::string toConstSizeString(int value, size_t length)
    {
        auto str = std::to_string(value);
        while (str.length() < length)
            str = " " + str;
        return str;
    }

    inline std::string toConstSizeString(float value, size_t length)
    {
        auto str = std::to_string(value);
        while (str.length() < length)
            str += "0";
        return str;
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

    constexpr int flatten(const int x, const int y, const int maxX)
    {
        return y * maxX + x;
    }

    constexpr int flatten(const int x, const int y, const int z, const int maxX, const int maxY)
    {
        return z * maxY * maxX + y * maxX + x;
    }

    constexpr int roughenX(const int index, const int maxX)
    {
        return index % maxX;
    }

    constexpr int roughenX(const int index, const int maxX, const int maxY)
    {
        return (index % (maxX * maxY)) % maxX;
    }

    constexpr int roughenY(const int index, const int maxX)
    {
        return index / maxX;
    }

    constexpr int roughenY(const int index, const int maxX, const int maxY)
    {
        return (index % (maxX * maxY)) / maxX;
    }

    constexpr int roughenZ(const int index, const int maxX, const int maxY)
    {
        return index % (maxX * maxY);
    }
}

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

namespace snn::internal
{
    class Tools
    {
    public:
        static int randomBetween(const int min, const int max);

        static float randomBetween(const float min, const float max);

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

        static std::string toString(std::chrono::milliseconds duration);
    };

    template <logLevel T, typename... Targs>
    constexpr void log(Targs&&... messages)
    {
        if constexpr (T > none && T <= verbose)
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
}

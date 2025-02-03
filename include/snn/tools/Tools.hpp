#pragma once
#include <chrono>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "Error.hpp"

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
}  // namespace snn

namespace snn::internal
{
enum coordinateIndex
{
    C = 0,
    X = 1,
    Y = 2
};
}

namespace snn::tools
{
static std::random_device rd;
static std::mt19937 rng(rd());

auto randomBetween(const int min, const int max) -> int;  // [min; max[
auto randomBetween(const float min, const float max) -> float;
auto randomVector(const float min, const float max, const size_t size) -> std::vector<float>;

auto toString(std::chrono::milliseconds duration) -> std::string;

auto toString(errorType err) -> std::string;

template <typename T>
static auto getMinValue(std::vector<T> vector) -> T
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
static auto getMaxValue(std::vector<T> vector) -> T
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
constexpr void log([[maybe_unused]] Targs&&... messages)
{
    if constexpr (T > none && T <= verbose)  // NOLINT(misc-redundant-expression)
    {
        (std::cout << ... << messages) << std::endl;
    }
}

template <logLevel T, bool endLine, typename... Targs>
constexpr void log([[maybe_unused]] Targs&&... messages)
{
    if constexpr (T > none && T <= verbose)  // NOLINT(misc-redundant-expression)
    {
        (std::cout << ... << messages);
        if constexpr (endLine)
        {
            std::cout << '\n';
        }
        else
        {
            std::cout << std::flush;
        }
    }
}

inline auto toConstSizeString(int value, size_t length) -> std::string
{
    auto str = std::to_string(value);
    while (str.length() < length) str = " " + str;
    return str;
}

template <int T>
auto toConstSizeString(float value, size_t length) -> std::string
{
    std::string str;
    if constexpr (T == 0)
    {
        str = std::format("{:.0f}", value);
    }
    else if constexpr (T == 2)
    {
        str = std::format("{:.2f}", value);
    }
    else if constexpr (T == 4)
    {
        str = std::format("{:.4f}", value);
    }
    else
    {
        throw std::exception();
    }

    while (str.length() < length)
    {
        str = " " + str;
    }
    return str;
}

template <typename T>
auto flatten(const vector2D<T>& vector2D) -> std::vector<T>
{
    std::vector<T> vector1D;
    size_t size = 0;
    for (const auto& v : vector2D)
    {
        size += vector2D.size();
    }
    vector1D.reserve(size);
    for (const auto& v : vector2D)
    {
        vector1D.insert(vector1D.end(), v.begin(), v.end());
    }
    return vector1D;
}

constexpr auto flatten(const int x, const int y, const int maxX) -> int { return (y * maxX) + x; }

constexpr auto flatten(const int x, const int y, const int z, const int maxX, const int maxY) -> int
{
    return (z * maxY * maxX) + (y * maxX) + x;
}

constexpr auto roughenX(const int index, const int maxX) -> int { return index % maxX; }

constexpr auto roughenX(const int index, const int maxX, const int maxY) -> int
{
    return (index % (maxX * maxY)) % maxX;
}

constexpr auto roughenY(const int index, const int maxX) -> int { return index / maxX; }

constexpr auto roughenY(const int index, const int maxX, const int maxY) -> int
{
    return (index % (maxX * maxY)) / maxX;
}

constexpr auto roughenZ(const int index, const int maxX, const int maxY) -> int { return index % (maxX * maxY); }
}  // namespace snn::tools

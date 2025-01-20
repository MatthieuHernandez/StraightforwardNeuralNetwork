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

int randomBetween(const int min, const int max);  // [min; max[
float randomBetween(const float min, const float max);
std::vector<float> randomVector(const float min, const float max, const size_t size);

auto toString(std::chrono::milliseconds duration) -> std::string;

auto toString(ErrorType err) -> std::string;

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
constexpr void log([[maybe_unused]] Targs&&... messages)
{
    if constexpr (T > none && T <= verbose) (std::cout << ... << messages) << std::endl;
}

template <logLevel T, bool endLine, typename... Targs>
constexpr void log([[maybe_unused]] Targs&&... messages)
{
    if constexpr (T > none && T <= verbose)
    {
        (std::cout << ... << messages);
        if constexpr (endLine)
            std::cout << std::endl;
        else
            std::cout << std::flush;
    }
}

inline std::string toConstSizeString(int value, size_t length)
{
    auto str = std::to_string(value);
    while (str.length() < length) str = " " + str;
    return str;
}

template <int T>
std::string toConstSizeString(float value, size_t length)
{
    std::string str;
    if constexpr (T == 0)
        str = std::format("{:.0f}", value);
    else if constexpr (T == 2)
        str = std::format("{:.2f}", value);
    else if constexpr (T == 4)
        str = std::format("{:.4f}", value);
    else
        throw std::exception();

    while (str.length() < length) str = " " + str;
    return str;
}

template <typename T>
std::vector<T> flatten(const vector2D<T>& vector2D)
{
    std::vector<T> vector1D;
    size_t size = 0;
    for (const auto& v : vector2D) size += vector2D.size();
    vector1D.reserve(size);
    for (const auto& v : vector2D) vector1D.insert(vector1D.end(), v.begin(), v.end());
    return vector1D;
}

constexpr int flatten(const int x, const int y, const int maxX) { return y * maxX + x; }

constexpr int flatten(const int x, const int y, const int z, const int maxX, const int maxY)
{
    return z * maxY * maxX + y * maxX + x;
}

constexpr int roughenX(const int index, const int maxX) { return index % maxX; }

constexpr int roughenX(const int index, const int maxX, const int maxY) { return (index % (maxX * maxY)) % maxX; }

constexpr int roughenY(const int index, const int maxX) { return index / maxX; }

constexpr int roughenY(const int index, const int maxX, const int maxY) { return (index % (maxX * maxY)) / maxX; }

constexpr int roughenZ(const int index, const int maxX, const int maxY) { return index % (maxX * maxY); }
}  // namespace snn::tools

#pragma once
#include <concepts>
#include <type_traits>

template <class I>
concept NeuronInput =
requires(const I inputs, int index)
{
    { inputs[index] } -> std::same_as<const float&>;
    { inputs.size() } -> std::same_as<size_t>;
};
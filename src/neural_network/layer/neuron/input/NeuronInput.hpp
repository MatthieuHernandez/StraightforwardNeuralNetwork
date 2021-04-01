#include <concepts>
#include <type_traits>

template <class I>
concept NeuronInput =
requires(I inputs)
{
    { inputs[0] } -> std::same_as<float&>;
    { inputs.size() } -> std::same_as<size_t>;
};
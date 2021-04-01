#include <concepts>
#include <type_traits>

template <typename T>
concept FloatRef = std::is_same<float&, T>::value;

template <typename T>
concept Size = std::is_same<size_t, T>::value;

template <class I>
concept NeuronInput =
requires(I inputs)
{
    { inputs[0] } -> FloatRef;
    { inputs.size() } -> Size;
};
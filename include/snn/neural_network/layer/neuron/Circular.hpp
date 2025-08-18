#pragma once
#include <vector>

namespace snn::internal
{
template <typename T>
class Circular final
{
    private:
        friend class Circular<std::vector<float>>;
        std::vector<T> queue;
        size_t indexPush = 0;
        size_t indexGet = 0;
        float divider = 1.0F;

    public:
        Circular() = default;  // use restricted to Boost library only
        Circular(const Circular<T>&) = default;
        Circular(Circular<T>&&) = default;
        auto operator=(const Circular<T>&) -> Circular<T>& = default;
        auto operator=(Circular<T>&&) -> Circular<T>& = default;
        ~Circular() = default;

        void initialize(size_t queueSize, size_t dataSize = 1);  // Should be call after the ctor.
        void reset();                                            // Do the same as initialize.

        [[nodiscard]] auto getBack() -> const T*;
        [[nodiscard]] auto getSum() const -> T;
        [[nodiscard]] auto getAverage() const -> T;
        [[nodiscard]] auto MultiplyAndAccumulate(const Circular<float>& multiplier) const -> T;
        void pushBack(const T& data);

        auto operator<=>(const Circular<T>& other) const = default;
};

template <>
void Circular<float>::initialize(size_t size, size_t dataSize);

template <>
void Circular<std::vector<float>>::initialize(size_t size, size_t dataSize);

template <>
void Circular<float>::reset();

template <>
void Circular<std::vector<float>>::reset();

template <>
auto Circular<float>::getSum() const -> float;

template <>
auto Circular<std::vector<float>>::getSum() const -> std::vector<float>;

template <>
auto Circular<float>::getAverage() const -> float;

template <>
auto Circular<std::vector<float>>::getAverage() const -> std::vector<float>;

template <>
auto Circular<std::vector<float>>::MultiplyAndAccumulate(const Circular<float>& multiplier) const -> std::vector<float>;

}  // namespace snn::internal
#include "Circular.tpp"  // IWYU pragma: keep

extern template class snn::internal::Circular<float>;
extern template class snn::internal::Circular<std::vector<float>>;

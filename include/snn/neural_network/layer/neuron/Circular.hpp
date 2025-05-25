#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <cstdint>
#include <vector>

namespace snn::internal
{
template <typename T>
class Circular final
{
    private:
        friend class StochasticGradientDescent;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

        std::vector<T> queue;
        size_t indexPush = 0;
        size_t indexGet = 0;

    public:
        Circular() = default;  // use restricted to Boost library only
        Circular(const Circular<T>&) = default;
        Circular(Circular<T>&&) = default;
        auto operator=(const Circular<T>&) -> Circular<T>& = default;
        auto operator=(Circular<T>&&) -> Circular<T>& = default;
        ~Circular() = default;

        void initialize(size_t queueSize, size_t dataSize = 1);  // Should be call after the ctor.

        [[nodiscard]] auto getBack() -> const T*;
        void pushBack(const T& data);

        auto operator<=>(const Circular<T>& other) const = default;
};

template <typename T>
template <class Archive>
void Circular<T>::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    archive & queue;
    archive & indexGet;
    archive & indexPush;
}

template <>
void Circular<float>::initialize(size_t size, size_t dataSize);

template <>
void Circular<std::vector<float>>::initialize(size_t size, size_t dataSize);

}  // namespace snn::internal
#include "Circular.tpp"  // IWYU pragma: keep

extern template class snn::internal::Circular<float>;
extern template class snn::internal::Circular<std::vector<float>>;

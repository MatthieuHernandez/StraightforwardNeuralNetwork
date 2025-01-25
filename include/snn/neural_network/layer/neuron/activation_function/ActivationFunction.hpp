#pragma once
#include <boost/serialization/access.hpp>
#include <memory>
#include <string>
#include <vector>

namespace snn
{
enum class activation
{
    sigmoid = 0,
    iSigmoid,
    tanh,
    ReLU,
    GELU,
    gaussian,
    identity
};
}
namespace snn::internal
{
class ActivationFunction
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        static std::vector<std::shared_ptr<ActivationFunction>> activationFunctions;

    public:
        const float min;
        const float max;

        ActivationFunction(float min, float max);
        virtual ~ActivationFunction() = default;
        static auto initialize() -> std::vector<std::shared_ptr<ActivationFunction>>;
        static auto get(activation type) -> std::shared_ptr<ActivationFunction>;

        [[nodiscard]] virtual auto function(const float) const -> float = 0;
        [[nodiscard]] virtual auto derivative(const float) const -> float = 0;

        [[nodiscard]] virtual auto getType() const -> activation = 0;

        [[nodiscard]] virtual auto getName() const -> std::string = 0;

        virtual auto operator==(const ActivationFunction& activationFunction) const -> bool;
        virtual auto operator!=(const ActivationFunction& activationFunction) const -> bool;
};
}  // namespace snn::internal

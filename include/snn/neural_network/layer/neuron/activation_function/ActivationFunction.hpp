#pragma once
#include <memory>
#include <vector>
#include <boost/serialization/access.hpp>

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
        static std::vector<std::shared_ptr<ActivationFunction>> initialize();
        static std::shared_ptr<ActivationFunction> get(activation type);

        [[nodiscard]] virtual float function(const float) const = 0;
        [[nodiscard]] virtual float derivative(const float) const = 0;

        [[nodiscard]] virtual activation getType() const = 0;

        [[nodiscard]] virtual std::string getName() const = 0;

        virtual bool operator==(const ActivationFunction& activationFunction) const;
        virtual bool operator!=(const ActivationFunction& activationFunction) const;
    };
}

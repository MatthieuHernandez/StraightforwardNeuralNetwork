#pragma once
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

    public:
        static std::vector<ActivationFunction*> activationFunctions;

        const float min;
        const float max;

        ActivationFunction(float min, float max);
        virtual ~ActivationFunction() = default;
        static void initialize();
        static ActivationFunction* get(activation type);

        virtual float function(const float) const = 0;
        virtual float derivative(const float) const = 0;

        virtual activation getType() const = 0;

        virtual bool operator==(const ActivationFunction& activationFunction) const;
        virtual bool operator!=(const ActivationFunction& activationFunction) const;
    };
}

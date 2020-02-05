#pragma once
#include <vector>
#include <boost/serialization/access.hpp>

namespace snn
{
	enum activationFunction
	{
		sigmoid = 0,
		iSigmoid,
		tanh,
		ReLU,
		gaussian
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

		ActivationFunction() = default;
		virtual ~ActivationFunction() = default;
		static void initialize();
		static ActivationFunction* get(activationFunction type);

		virtual float function(const float) const = 0;
		virtual float derivative(const float) const = 0;

		virtual activationFunction getType() const = 0;

		virtual bool operator==(const ActivationFunction& activationFunction) const;
		virtual bool operator!=(const ActivationFunction& activationFunction) const;
	};
}

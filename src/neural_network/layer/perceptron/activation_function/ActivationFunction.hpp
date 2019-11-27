#pragma once
#pragma warning(push, 0)
#include <boost/serialization/access.hpp>
#pragma warning(pop)

namespace snn
{
	enum activationFunctionType
	{
		sigmoid = 0,
		iSigmoid,
		tanH,
		reLU,
		gaussian
	};
}
namespace snn::internal
{
	class ActivationFunction
	{
	private :

		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, unsigned version);

	public :
		//static std::vector<ActivationFunction*> listOfActivationFunction;

		ActivationFunction() = default;
		ActivationFunction(const ActivationFunction& activationFunction) = default;
		virtual ~ActivationFunction() = default;
		static void initialize();
		static ActivationFunction* create(activationFunctionType type);

		virtual float function(const float) const = 0;
		virtual float derivative(const float) const = 0;

		virtual activationFunctionType getType() const = 0;

		virtual bool operator==(const ActivationFunction& activationFunction) const;
		virtual bool operator!=(const ActivationFunction& activationFunction) const;
	};
}
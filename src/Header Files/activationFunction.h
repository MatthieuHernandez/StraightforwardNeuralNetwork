#pragma once
#include <boost/serialization/access.hpp>

enum activationFunctionType
{
	sigmoid = 0,
	iSigmoid,
	tanH,
	reLU,
	gaussian
};

class ActivationFunction
{
private :

	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& ar, const unsigned int version);


public :

	//static std::vector<ActivationFunction*> listOfActivationFunction;

	ActivationFunction() = default;
	ActivationFunction(const ActivationFunction& activationFunction);
	virtual ~ActivationFunction() = default;
	static void initialize();
	static ActivationFunction* create(activationFunctionType type);

	virtual float function(const float) const = 0;
	virtual float derivative(const float) const = 0;

	virtual activationFunctionType getType() const = 0;

	virtual bool operator==(const ActivationFunction& activationFunction) const;
	virtual bool operator!=(const ActivationFunction& activationFunction) const;
};

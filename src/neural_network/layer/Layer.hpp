#pragma once
#pragma warning(push, 0)
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#pragma warning(pop)
#include "LayerOption.hpp"
#include "perceptron/perceptron.hpp"

namespace snn::internal
{
	enum LayerType
	{
		allToAll = 0
	};

	class Layer
	{
	private :
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, unsigned version);

	protected:
		int numberOfInputs = 0;
		int numberOfNeurons = 0;

		LayerOption* option;

		std::vector<float> errors;
		std::vector<Perceptron> neurons;

	public:
		Layer(const int numberOfInputs,
              const int numberOfNeurons,
              LayerOption* option);
		Layer() = default;
		virtual ~Layer() = default;

		virtual std::vector<float> output(const std::vector<float>& inputs) = 0;
		virtual std::vector<float> backOutput(std::vector<float>& inputsError) = 0;
		virtual void train(std::vector<float>& inputsError) = 0;

		virtual int isValid() const;

		virtual LayerType getType() const = 0;

		virtual bool operator==(const Layer& layer) const;
		virtual Layer& operator=(const Layer& layer) = 0;
		virtual bool operator!=(const Layer& layer) const;
	};

	template <class Archive>
	void Layer::serialize(Archive& ar, unsigned version)
	{
		ar & this->option;
		ar & this->numberOfInputs;
		ar & this->numberOfNeurons;
		ar & this->errors;
		ar & this->neurons;
	}
}

#pragma once
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include "LayerType.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn::internal
{
	class Layer
	{
	private :
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, unsigned version);

	protected:
		std::vector<float> errors;

	public:
        Layer(layerType type,
              int numberOfInputs,
              int numberOfNeurons);

        Layer() = default;
        virtual ~Layer() = default;

        static const layerType type;

        /*const*/ int numberOfInputs;
        /*const*/ int numberOfNeurons;

		std::vector<Perceptron> neurons;

		virtual std::vector<float> output(const std::vector<float>& inputs) = 0;
		virtual std::vector<float> backOutput(std::vector<float>& inputsError) = 0;
		virtual void train(std::vector<float>& inputsError) = 0;

        [[nodiscard]] virtual int isValid() const;
		virtual bool operator==(const Layer& layer) const;
		virtual Layer& operator=(const Layer& layer) = 0;
		virtual bool operator!=(const Layer& layer) const;
	};

	template <class Archive>
	void Layer::serialize(Archive& ar, unsigned version)
	{
		ar & this->numberOfInputs;
		ar & this->numberOfNeurons;
		ar & this->errors;
		ar & this->neurons;
	}
}

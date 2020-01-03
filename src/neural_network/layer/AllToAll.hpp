#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include "Layer.hpp"
#include "perceptron/Perceptron.hpp"

namespace snn::internal
{
	class AllToAll : public Layer
	{
	private :
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, unsigned version);


	public :
		AllToAll() = default;
		~AllToAll() = default;

		AllToAll(int numberOfInputs,
                 int numberOfNeurons,
                 activationFunction activation,
                 float* learningRate,
                 float* momentum);

		std::vector<float> output(const std::vector<float>& inputs) override;
		std::vector<float> backOutput(std::vector<float>& inputsError) override;
		void train(std::vector<float>& inputsError) override;

		int isValid() const override;

		Layer& operator=(const Layer& layer) override;
		bool operator==(const AllToAll& layer) const;
		bool operator!=(const AllToAll& layer) const;
	};

	template <class Archive>
	void AllToAll::serialize(Archive& ar, const unsigned version)
	{
		boost::serialization::void_cast_register<AllToAll, Layer>();
		ar & boost::serialization::base_object<Layer>(*this);
	}
}

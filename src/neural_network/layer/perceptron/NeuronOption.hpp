#pragma once
#pragma warning(push, 0)
#include <boost/serialization/access.hpp>
#pragma warning(pop)

namespace snn::internal
{
	class NeuronOption
	{
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:

		float learningRate = 0.05f;
		float momentum = 0.0f;

		NeuronOption() = default;
		~NeuronOption() = default;

		bool operator==(const NeuronOption& option) const;
		NeuronOption& operator=(const NeuronOption& option) = default;
	};

	template <class Archive>
	void NeuronOption::serialize(Archive& ar, const unsigned version)
	{
		ar & this->learningRate;
		ar & this->momentum;
	}
}

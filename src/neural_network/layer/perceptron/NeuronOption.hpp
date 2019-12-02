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

		LayerOption() = default;
		~LayerOption() = default;

		bool operator==(const LayerOption& option) const;
		LayerOption& operator=(const LayerOption& option) = default;
	};

	template <class Archive>
	void LayerOption::serialize(Archive& ar, const unsigned version)
	{
		ar & this->learningRate;
		ar & this->momentum;
	}
}

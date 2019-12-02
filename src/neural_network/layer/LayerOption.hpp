#pragma once
#pragma warning(push, 0)
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#pragma warning(pop)
#include "perceptron/NeuronOption.hpp"

namespace snn::internal
{
	class LayerOption : public NeuronOption
	{
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:

		LayerOption() = default;
		~LayerOption() = default;

		bool operator==(const LayerOption& option) const;
		LayerOption& operator=(const LayerOption& option);
	};

	template <class Archive>
	void LayerOption::serialize(Archive& ar, const unsigned version)
	{
		boost::serialization::void_cast_register<LayerOption, NeuronOption>();
		ar & boost::serialization::base_object<NeuronOption>(*this);
	}
}

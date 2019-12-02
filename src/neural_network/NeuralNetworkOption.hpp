#pragma once
#pragma warning(push, 0)
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#pragma warning(pop)
#include "layer/LayerOption.hpp"

namespace snn::internal
{
	class NeuralNetworkOption : public LayerOption
	{
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:

		NeuralNetworkOption() = default;
		~NeuralNetworkOption() = default;

		bool operator==(const NeuralNetworkOption& option) const;
		NeuralNetworkOption& operator=(const NeuralNetworkOption& option) = default;
	};

	template <class Archive>
	void NeuralNetworkOption::serialize(Archive& ar, const unsigned version)
	{
		boost::serialization::void_cast_register<NeuralNetworkOption, LayerOption>();
		ar & boost::serialization::base_object<LayerOption>(*this);
	}
}

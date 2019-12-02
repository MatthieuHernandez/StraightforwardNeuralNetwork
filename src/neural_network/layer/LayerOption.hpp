#pragma once
#pragma warning(push, 0)
#include <boost/serialization/access.hpp>
#pragma warning(pop)

namespace snn::internal
{
	class LayerOption : NeuronOption
	{
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:

		LayerOption() = default;
		~LayerOption() = default;

		bool operator==(const StraightforwardOption& option) const;
		StraightforwardOption& operator=(const StraightforwardOption& option) = default;
	};

	template <class Archive>
	void StraightforwardOption::serialize(Archive& ar, const unsigned version)
	{
		boost::serialization::void_cast_register<LayerOption, NeuronOption>();
		ar & boost::serialization::base_object<NeuronOption>(*this);
	}
}

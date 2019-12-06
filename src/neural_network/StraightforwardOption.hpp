#pragma once
#include <string>
#pragma warning(push, 0)
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#pragma warning(pop)
#include "NeuralNetworkOption.hpp"

namespace snn
{
	class StraightforwardOption : public internal::NeuralNetworkOption
	{
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:

		bool autoSaveWhenBetter = false;
		std::string saveFilePath = "save.snn";

		StraightforwardOption() = default;
		~StraightforwardOption() = default;

		bool operator==(const StraightforwardOption& option) const;
		StraightforwardOption& operator=(const StraightforwardOption& option);
	};

	template <class Archive>
	void StraightforwardOption::serialize(Archive& ar, const unsigned version)
	{
		boost::serialization::void_cast_register<StraightforwardOption, NeuralNetworkOption>();
		ar & boost::serialization::base_object<NeuralNetworkOption>(*this);
		ar & this->autoSaveWhenBetter;
		ar & this->saveFilePath;
	}
}

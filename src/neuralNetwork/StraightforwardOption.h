#pragma once
#include <string>
#include <boost/serialization/access.hpp>

namespace snn
{
	class StraightforwardOption
	{
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive& ar, const unsigned int version);

	public:

		bool autoSaveWhenBetter = false;
		std::string saveFilePath = "save.snn";
		float learningRate = 0.05f;
		float momentum = 0.0f;

		StraightforwardOption() = default;
		~StraightforwardOption() = default;

		bool operator==(const StraightforwardOption& option) const;
		StraightforwardOption& operator=(const StraightforwardOption& option) = default;
	};


	template <class Archive>
	void StraightforwardOption::serialize(Archive& ar, const unsigned version)
	{
		ar & this->autoSaveWhenBetter;
	}
}

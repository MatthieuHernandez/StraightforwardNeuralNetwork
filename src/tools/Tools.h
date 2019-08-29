#pragma once
#include <vector>

namespace snn::internal
{
	class Tools
	{
	public:
		static int randomBetween(const int min, const int max);

		static float randomBetween(const float min, const float max);

		template <typename T>
		static T getMinValue(std::vector<T> vector);

		template <typename T>
		static T getMaxValue(std::vector<T> vector);
	};
}

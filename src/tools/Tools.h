#pragma once
#include <vector>

namespace snn::tools
{
	int randomBetween(const int min, const int max);

	float randomBetween(const float min, const float max);

	template <typename T>
	T getMinValue(std::vector<T> vector);

	template <typename T>
	T getMaxValue(std::vector<T> vector);
}

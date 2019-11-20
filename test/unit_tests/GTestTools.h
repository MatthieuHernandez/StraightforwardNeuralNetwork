#pragma once
#pragma warning(push, 0) 
#include "gtest/gtest.h"
#pragma warning(pop)

template<typename T>
void EXPECT_ABOUT_EQ(T min, T value, T max, std::string valueName = "value")
{
	ASSERT_TRUE(value >= min) << valueName + " = " + std::to_string(value) + "\t\t >= " + std::to_string(min);
	ASSERT_TRUE(value <= max) << valueName + " = " + std::to_string(value) + "\t\t <= " + std::to_string(max);
}


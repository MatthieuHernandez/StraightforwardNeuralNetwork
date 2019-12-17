#pragma once
#include <gtest/gtest.h>

template<typename T>
void EXPECT_ABOUT_EQ(T min, T value, T max, std::string valueName = "value");

extern void PRINT_LOG(std::string str);

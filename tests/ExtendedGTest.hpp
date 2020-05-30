#pragma once
#include <gtest/gtest.h>

template<typename T>
void ASSERT_BETWEEN(T min, T value, T max, std::string valueName = "value");

void ASSERT_SUCCESS();

void ASSERT_FAIL(std::string message = "");

extern void PRINT_LOG(std::string message);

extern void PRINT_RESULT(std::string message);

extern void ASSERT_ACCURACY(float actual, float expected);

extern void ASSERT_MAE(float actual, float expected);

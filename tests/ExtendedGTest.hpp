#pragma once
#include <gtest/gtest.h>

template<typename T>
void ASSERT_BETWEEN(T min, T value, T max, std::string valueName = "value");

void ASSERT_SUCCESS();

void ASSERT_FAIL(std::string message = "");

void ASSERT_VECTOR_EQ(const std::vector<float> values, const std::vector<float> expected_values);

void PRINT_LOG(std::string message);

void PRINT_NUMBER_OF_PARAMETERS(int value);

void PRINT_NUMBER_OF_NEURONS(int value);

void PRINT_RESULT(std::string message);

void ASSERT_ACCURACY(float actual, float expected);

void ASSERT_RECALL(float actual, float expected);

void ASSERT_MAE(float actual, float expected);

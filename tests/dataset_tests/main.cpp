#include <gtest/gtest.h>
#include "../GTestTools.hpp"

int main(int ac, char* av[])
{
	testing::InitGoogleTest(&ac, av);
	const auto tests = RUN_ALL_TESTS();
	::testing::GTEST_FLAG(filter) = "Dataset_";
	return tests;
}

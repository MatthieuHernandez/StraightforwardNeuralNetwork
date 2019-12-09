#include "GTestTools.hpp"
#pragma warning(push, 0) 
#include <gtest/gtest.h>
#pragma warning(pop)

int main(int ac, char* av[])
{
	testing::InitGoogleTest(&ac, av);
	const auto tests = RUN_ALL_TESTS();
	system("PAUSE");
	return tests;
}

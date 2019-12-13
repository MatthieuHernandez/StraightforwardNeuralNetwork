#pragma once
#include "../../GTestTools.h"
#include "Cifar10.hpp"

using namespace std;

class Cifar10Test : public testing::Test
{
protected:

	Cifar10Test()
	{
		data = new Cifar10();
	}

public:

	Data* data;
};

TEST_F(Cifar10Test, loadData)
{
    ASSERT_NE(Data, nullptr);
}
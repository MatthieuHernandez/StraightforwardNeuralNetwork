#pragma once
#include "../../GTestTools.h"
#include "Mnist.hpp"

using namespace std;

class MnistTest : public testing::Test
{
protected:

	MnistTest()
	{
		data = new Mnist();
	}

public:

	Data* data;
};

TEST_F(MnistTest, loadData)
{
    ASSERT_NE(Data, nullptr);
}
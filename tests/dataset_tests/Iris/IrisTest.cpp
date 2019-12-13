#pragma once
#include "../../GTestTools.h"
#include "Iris.hpp"

using namespace std;

class IrisTest : public testing::Test
{
protected:

	IrisTest()
	{
		data = new Iris();
	}

public:

	Data* data;
};

TEST_F(IrisTest, loadData)
{
    ASSERT_NE(Data, nullptr);
}
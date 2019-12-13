#pragma once
#include "../../GTestTools.h"
#include "Wine.hpp"

using namespace std;

class WineTest : public testing::Test
{
protected:

	WineTest()
	{
		data = new Wine();
	}

public:

	Data* data;
};

TEST_F(WineTest, loadData)
{
    ASSERT_NE(Data, nullptr);
}
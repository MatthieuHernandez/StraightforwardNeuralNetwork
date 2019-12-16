#include "../../GTestTools.hpp"
#include "Wine.hpp"

using namespace std;

class WineTest : public testing::Test
{
protected :
    WineTest()
    {
        dataset = new Wine();
    }

public :
    Dataset* dataset;
};

TEST_F(WineTest, loadData)
{
    ASSERT_FALSE(dataset->data);
}
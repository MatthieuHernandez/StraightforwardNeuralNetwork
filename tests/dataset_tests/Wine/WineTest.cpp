#include "../../GTestTools.hpp"
#include "data/Data.hpp"
#include "Wine.hpp"

using namespace std;
using namespace snn;

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
    ASSERT_TRUE(dataset->data);
}
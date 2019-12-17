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
        Wine dataset;
        data = move(dataset.data);
    }

public :
    unique_ptr<Data> data;
};

TEST_F(WineTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 13);
    ASSERT_EQ(data->numberOfLabel, 3);
    ASSERT_EQ(data->sets[training].inputs.size(), 178);
    ASSERT_EQ(data->sets[training].labels.size(), 178);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 178);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 178);
}
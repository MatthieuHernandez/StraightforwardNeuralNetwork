#include "../../GTestTools.hpp"
#include "data/Data.hpp"
#include "Iris.hpp"

using namespace std;
using namespace snn;

class IrisTest : public testing::Test
{
protected:
    IrisTest()
    {
        Iris dataset;
        data = move(dataset.data);
    }

public:
    unique_ptr<Data> data;
};

TEST_F(IrisTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 4);
    ASSERT_EQ(data->numberOfLabel, 3);
    ASSERT_EQ(data->sets[training].inputs.size(), 150);
    ASSERT_EQ(data->sets[training].labels.size(), 150);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 150);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 150);
}
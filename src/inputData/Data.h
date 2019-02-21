#ifndef DATA_H
#define DATA_H
#include <vector>

namespace data
{
	enum set
	{
		testing = 0,
		training = 1
	};

	enum problemType
	{
		classification = 0,
		regression = 1
	};

	class Data
	{
	protected:
		std::vector<int> indexes;

		void clearData();

	public:

		void shuffle();
		void unshuffle();

		int sizeOfData{}; // size of one data, equal to size of neural network inputs
		int numberOfLabel{}; // the number of class, equal to size of neural network outputs

		problemType problem;

		struct Set
		{
			int index{0};
			int size{0}; // number of data inside set
			std::vector<std::vector<float>> data{};
			std::vector<std::vector<float>> labels{};
		} sets[2];

		virtual ~Data() = default;
		virtual void loadData() = 0;

		virtual std::vector<float>& getTrainingData(const int index);
		virtual std::vector<float>& getTestingData(const int index);

		virtual int getTrainingLabel(const int) { throw std::exception(); }
		virtual int getTestingLabel(const int) { throw std::exception(); }

		virtual std::vector<float>& getTrainingOutputs(const int index);
		virtual std::vector<float>& getTestingOutputs(const int) { throw std::exception(); }

		std::vector<float>& getData(set set, const int index);
		std::vector<float>& getOutputs(set set, const int index);
		int getLabel(set set, const int index);
	};
}
#endif // DATA_H

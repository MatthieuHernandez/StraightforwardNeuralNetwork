#pragma once
#include <vector>

namespace snn
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
		problemType problem;
		int sizeOfData{}; // size of one data, equal to size of neural network inputs
		int numberOfLabel{}; // the number of class, equal to size of neural network outputs

		struct Set
		{
			int index{0};
			int size{0}; // number of data inside set
			std::vector<std::vector<float>> inputs{};
			std::vector<std::vector<float>> labels{};
		} sets[2];

		Data(std::vector<std::vector<float>>& trainingInputs,
		     std::vector<std::vector<float>>& trainingLabels,
		     std::vector<std::vector<float>>& testingInputs,
		     std::vector<std::vector<float>>& testingLabels);

		virtual ~Data() = default;

		void shuffle();
		void unshuffle();

		virtual std::vector<float>& getTrainingData(const int index);
		virtual std::vector<float>& getTestingData(const int index);

		virtual int getTrainingLabel(const int) { throw std::exception(); }
		virtual int getTestingLabel(const int) { throw std::exception(); }

		virtual std::vector<float>& getTrainingOutputs(const int index);
		virtual std::vector<float>& getTestingOutputs(const int) = 0;

		std::vector<float>& getData(set set, const int index);
		std::vector<float>& getOutputs(set set, const int index);
		int getLabel(set set, const int index);
	};
}

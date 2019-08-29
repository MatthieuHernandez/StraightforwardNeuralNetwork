#pragma once
#include <vector>

namespace snn
{
	enum set
	{
		testing = 0,
		training = 1
	};

	class Data
	{
	private:
		void initialize(std::vector<std::vector<float>>& trainingInputs,
		     std::vector<std::vector<float>>& trainingLabels,
		     std::vector<std::vector<float>>& testingInputs,
		     std::vector<std::vector<float>>& testingLabels,
		     float value);

	protected:
		std::vector<int> indexes;
		float value;
		void clearData();

		Data(std::vector<std::vector<float>>& trainingInputs,
		     std::vector<std::vector<float>>& trainingLabels,
		     std::vector<std::vector<float>>& testingInputs,
		     std::vector<std::vector<float>>& testingLabels,
		     float value);

		Data(std::vector<std::vector<float>>& inputs,
		     std::vector<std::vector<float>>& labels,
		     float value);

	public:
		int sizeOfData{}; // size of one data, equal to size of neural network inputs
		int numberOfLabel{}; // the number of class, equal to size of neural network outputs

		struct Set
		{
			int index{0};
			int size{0}; // number of data inside set
			std::vector<std::vector<float>> inputs{};
			std::vector<std::vector<float>> labels{};
		} sets[2];

		virtual ~Data() = default;

		void normalization(float min, float max);

		void shuffle();
		void unshuffle();

		[[nodiscard]] float getValue() const {return value;}

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

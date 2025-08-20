#include <concepts>

template <typename T>
concept WithoutBatchSize = requires(T t) {
    { t.resetLearningVariables() } -> std::same_as<void>;
};

template <typename T>
concept WithBatchSize = requires(T t, int batchSize) {
    { t.resetLearningVariables(batchSize) } -> std::same_as<void>;
};

template <typename T>
concept LearningObject = WithoutBatchSize<T> || WithBatchSize<T>;
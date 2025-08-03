#include <concepts>

template <typename T>
concept LearningObject = requires(T t) {
    { t.resetLearningVariables() } -> std::same_as<void>;
};
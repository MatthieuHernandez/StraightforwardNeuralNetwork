---
layout: default
title: Data
nav_order: 4
permalink: /data
has_children: true
---

Data is the class to store your data. You can declare it like this.

```cpp
vector<vector<float>> inputs;
vector<vector<float>> label;
Data data(problem::classification, inputData, expectedOutputs, nature::nonTemporal);
```

```
enum class problem
    {
        classification,
        multipleClassification,
        regression
    };
```

There are 3 types of problem `classification`, `multipleClassification` and `regression`.

 * **Classification** is used when you want to classify you input

{:toc}


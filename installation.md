---
layout: default
title: Installation
nav_order: 2
---

## Installation (with *CMake* 3.17.1)

### Linux, UNIX - GCC 10.1
* To compile open a command prompt in the `build` folder then  the library run `cmake -G"Unix Makefiles" ./..  && make` from `StraightforwardNeuralNetwork/build`

* To run unit tests exectute `./tests/unit_tests/UnitTests.out` from `StraightforwardNeuralNetwork/build`

* To run dataset tests run `./ImportDatasets.sh` from `StraightforwardNeuralNetwork\tests\dataset_tests` and exectute `./tests/dataset_tests/DatasetTests.out` from `StraightforwardNeuralNetwork/build`

### Windows - MSVC++ 14.2
 * You can generate a Visual Studio project with CMake by running the following command from the `build` folder:
 `cmake -G"Visual Studio 16 2019" ./..`
 

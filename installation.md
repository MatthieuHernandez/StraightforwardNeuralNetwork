---
layout: default
title: Installation
nav_order: 2
---
# Instalation (with *CMake* 3.17.1) &#127981;

## Linux, UNIX - GCC 10.1
* To compile open a command prompt and run `cmake -G"Unix Makefiles" ./..  && make` from the `build` folder.

* To run the unit tests execute `./tests/unit_tests/UnitTests` from `build` folder.

* To run the dataset tests run `./tests/dataset_tests/ImportDatasets.sh` and execute `./tests/dataset_tests/DatasetTests` from `build` folder.

## Windows - MSVC++ 14.2
* You can generate a Visual Studio project by running `cmake -G"Visual Studio 16 2019" ./..` from `build` folder.

* To run the unit tests open `./tests/unit_tests/UnitTests.vcxproj` in Visual Studio.

* To run the dataset tests run `./tests/dataset_tests/ImportDatasets.sh` and open `./tests/dataset_tests/DatasetTests.vcxproj` in Visual Studio.
 
<br/>

 [After installation]({{site.baseurl}}/quick_start.html){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
 

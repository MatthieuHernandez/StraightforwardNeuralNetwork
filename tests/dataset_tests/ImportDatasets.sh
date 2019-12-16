#!/bin/bash
git clone --no-tags --depth 1 "https://github.com/MatthieuHernandez/Datasets-for-Machine-Learning.git" tmp
cp -a ./tmp/CIFAR-10 ./
cp -a ./tmp/Iris ./
cp -a ./tmp/MNIST ./
cp -a ./tmp/Wine ./
rm -rf ./tmp
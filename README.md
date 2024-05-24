# Soil Trace Element Prediction Script

This repository contains Python scripts that support the publication titled "Predicted Losses of Sulfur and Selenium in European Soils Using Machine Learning: A Call for Prudent Model Interrogation and Selection". 

## Introduction

These scripts were used to perform the machine learning, statistical analysis, and figure creation for this work.
This repository is submitted for transparency and so that others may use it as a resource.

## Description

The scripts build prediction using: 
- Multi-layer Perceptron (MLP) Regressor
- Support Vector Regression using the linear support vector regression (Lin-SVR)

Input variable (see describption in the paper) necessary to run the model are: 
- Clay content, soil PH
- Soil organic carbon (SOC) content
- Average daily precipitation (Precip)
- Average daily evapotranspiration (ET)
- Aridity index (AI)
- Average chemical index of alteration (CIA)
- Total S or Se atmospheric deposition.

The scripts in this repository organized into 4 steps of the workflow.
1. Transforms the predictor variables to make distributions more normal.
2. Z-scores (standardizes) the predictor variables.
3. Tunes, evaluates, and executes the machine learning models to generate predictions. One script for each model-element pair (e.g., MLP-S)
4. Generates maps and charts of the model predictions and statistics.

## Installation

To use the scripts in this repo, follow these steps:

1. Clone this repository to your local development system
2. Install the packages and dependences
The scripts require the following packages, although future versions may work also.
numpy           1.26.4
openpyxl        3.1.2
pandas          2.2.2
scipy           1.13.0
matplotlib      3.8.4
scikit-learn    1.4.2
Cartopy         0.22.0 (the following are dependencies of Cartopy)
geos            3.8.0
proj            9.3.1
pyproj          3.6.1
pyshp           2.3.1
shapely         2.0.1

Cartopy is the package use for making maps. It is recommended to use conda (instead of pip)
to install Cartopy. 

3. Set up and execute with your local Python environment

## Usage

Each script is executed individually in a step-wise sequential process.
Any user-defined inputs are defined as constants that can be altered.
The input to the first script is the master table (Master_table.xlsx).

## Licence

Copyright © 2024 EcoChem Lab.

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

A copy of the license is available in the repository's [LICENSE](LICENSE) file.

## Citing

coming soon!!!

# DMD-Neural-Operator

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red)](https://pytorch.org/)

Neural operator architecture that combines Dynamic Mode Decomposition (DMD) with deep learning for solving partial differential equations (PDEs).

<img width="800px" src="https://github.com/NekkittAY/DMD-Neural-Operator/blob/main/doc/NN_diagram.png"/>

<img width="800px" src="https://github.com/NekkittAY/DMD-Neural-Operator/blob/main/doc/formula_1.png"/>
<img width="800px" src="https://github.com/NekkittAY/DMD-Neural-Operator/blob/main/doc/formula_2.png"/>

## A neural operator using dynamic mode decomposition analysis to approximate the partial differential equations

### Abstract

Solving partial differential equations (PDEs) for various initial and boundary conditions requires significant computational resources. We propose a neural operator $G_\theta: \mathcal{A} \to \mathcal{U}$, mapping functional spaces, which combines dynamic mode decomposition (DMD) and deep learning for efficient modeling of spatiotemporal processes. The method automatically extracts key modes and system dynamics and uses them to construct predictions, reducing computational costs compared to traditional methods (FEM, FDM, FVM). The approach is demonstrated on the heat equation, Laplace's equation, and Burgers' equation, where it achieves high solution reconstruction accuracy.

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Algorithm](#algorithm)

## Overview

DMD-Neural-Operator is a hybrid approach that:
1. Uses DMD for dimensionality reduction and feature extraction from PDE solutions
2. Combines DMD modes and dynamics with neural networks for operator learning
3. Provides an efficient way to approximate PDE solutions in parameterized settings

## Technology Stack

- **Core**: Python 3.8+
- **Deep Learning**: PyTorch 2.6+
- **DMD**: PyDMD 2025.4+
- **Numerical Computing**: NumPy, SciPy
- **Visualization**: Matplotlib
- **Development**: tqdm, torchviz

## Features

- Dimensionality reduction using DMD analysis
- Neural operator architecture for function space mapping
- Efficient processing of spatiotemporal data
- Customizable network architecture with multiple branches

## Algorithm

<img width="800px" src="https://github.com/NekkittAY/DMD-Neural-Operator/blob/main/doc/Algorithm.png"/>

# C-WiSPr: A Chebyshev Polynomial-based Wind Speed Profile Characterization Framework

**C**hebyshev polynomials-based **Wi**nd **S**peed **Pr**ofile characterization

[![Journal](https://img.shields.io/badge/Journal-Wind%20Energy%20(Wiley)-blue)](https://onlinelibrary.wiley.com/journal/10991824)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the complete implementation of the C-WiSPr framework presented in the publication:

> **"A Chebyshev Polynomial-based Wind Speed Profile Characterization Framework: Applications in Mesoscale Model Evaluation"**  
> *Journal of Wind Energy, Wiley Publications*

C-WiSPr provides a novel approach to characterize wind speed profiles using Chebyshev polynomial approximations, enabling efficient representation and analysis of vertical wind profiles for wind energy applications and mesoscale model evaluation.

## 🚀 Key Features

- **Efficient Profile Representation**: Reduces complex vertical wind profiles to a small set of Chebyshev coefficients
- **Gap-Filling Capabilities**: Reconstructs complete wind profiles from sparse measurements
- **Multi-Dataset Support**: Handles both observational (lidar profiler) and model (NOW23) data
- **Comprehensive Analysis Tools**: Complete workflow from coefficient computation to wind energy analysis
- **Publication-Ready Visualizations**: Advanced plotting functions for scientific publications

## 📁 Repository Structure

```
C-WiSPr/
├── libraries.py                           # Core Chebyshev polynomial functions
├── plotters.py                           # Visualization and plotting utilities
├── Compute_Chebyshev_for_NOW23.py        # NOW23 dataset coefficient computation
├── Compute_Chebyshev_for_profilers.py    # Profiler data coefficient computation
├── Wind_Energy_analysis.ipynb            # Complete analysis and visualizations
└── README.md                            # This file
```

## 🔧 Core Components

### 1. **libraries.py** - Mathematical Foundation
The mathematical core of C-WiSPr containing:

- **Chebyshev Polynomial Generation**: `Chebyshev_Basu()` - Computes Chebyshev polynomials of the first and second kind
- **Coefficient Computation**: `Chebyshev_Coeff()` - Derives coefficients from wind profiles via inverse transformation
- **Profile Reconstruction**: `WindProfile()` - Reconstructs full wind profiles from coefficients
- **Height Normalization**: `normalize()` - Maps physical heights to normalized domain [-1, 1]
- **Data Processing Pipeline**: Advanced functions for handling multi-dimensional wind datasets
- **Loss Functions**: Custom loss functions for machine learning applications

**Key Parameters:**
- `poly_order = 4`: Default polynomial order (configurable)
- `CPtype = 1`: Chebyshev polynomial type (1st or 2nd kind)
- `ref_H`: Reference height levels covering profiler and NOW23 ranges

### 2. **plotters.py** - Visualization Suite
Scientific plotting utilities including:

- **Statistical Plots**: Hexbin plots with performance metrics (MAE, RMSE, R², MAPE)
- **Profile Comparisons**: Multi-panel wind profile visualizations
- **Color Management**: Custom colormaps (Turbo) for publication-quality figures
- **Error Analysis**: Earth Mover's Distance (EMD) and other statistical measures

### 3. **Computation Scripts**

#### **Compute_Chebyshev_for_NOW23.py**
Processes NOW23 dataset wind profiles:
- Reads vertical wind profiles from NetCDF files
- Computes Chebyshev coefficients for height range 10-500m
- Supports parallel processing for multiple stations
- Configurable polynomial order (default: 6th order)

#### **Compute_Chebyshev_for_profilers.py**
Handles lidar profiler observations:
- Processes NYSM profiler network data
- Applies quality control filters for data completeness
- Ensures adequate vertical coverage (100-200m, 225-375m, 400-500m segments)
- Generates coefficients for 15 profiler stations across New York State

### 4. **Wind_Energy_analysis.ipynb** - Complete Analysis
Comprehensive Jupyter notebook containing:
- Data loading and preprocessing workflows
- Coefficient computation demonstrations
- Wind profile reconstruction examples
- Statistical analysis and model evaluation
- All figures and analyses presented in the manuscript
- Wind energy resource assessment applications

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy xarray pandas matplotlib seaborn scipy scikit-learn torch
```

### Quick Start

1. **Compute Chebyshev Coefficients**:
```python
from libraries import Chebyshev_Coeff, WindProfile, normalize
import numpy as np

# Example: 4th-order approximation
heights = np.array([10, 50, 100, 200, 300, 400, 500])  # meters
wind_speeds = np.array([5.2, 6.8, 7.5, 8.1, 8.4, 8.6, 8.7])  # m/s

# Compute coefficients
coeffs = Chebyshev_Coeff(heights, wind_speeds, poly_order=4, CPtype=1)

# Reconstruct profile at any height
new_heights = np.linspace(10, 500, 100)
reconstructed_profile = WindProfile(new_heights, coeffs)
```

2. **Batch Processing**:
```bash
# Process NOW23 data (station index as argument)
python Compute_Chebyshev_for_NOW23.py 0

# Process all profiler stations
python Compute_Chebyshev_for_profilers.py
```

3. **Analysis and Visualization**:
Open and run `Wind_Energy_analysis.ipynb` for complete analysis workflows.

## 📊 Key Applications

### Wind Energy Resource Assessment
- **Hub Height Extrapolation**: Accurate wind speed prediction at turbine hub heights
- **Shear Characterization**: Quantitative analysis of wind shear profiles
- **Resource Mapping**: Efficient representation for large-scale wind resource databases

### Mesoscale Model Evaluation
- **Profile Comparison**: Systematic comparison between model and observational profiles
- **Bias Identification**: Detection of systematic model biases at different heights
- **Performance Metrics**: Standardized evaluation using coefficient-based statistics

### Data Gap-Filling
- **Missing Data Reconstruction**: Fill gaps in incomplete vertical profiles
- **Quality Control**: Smooth noisy measurements while preserving physical characteristics
- **Temporal Interpolation**: Reconstruct profiles for missing time periods

## 🔬 Scientific Background

### Chebyshev Polynomial Advantages
1. **Optimal Approximation**: Near-optimal polynomial approximation properties
2. **Orthogonality**: Mathematically orthogonal basis functions reduce overfitting
3. **Efficient Computation**: Fast algorithms for coefficient computation and profile reconstruction
4. **Physical Interpretability**: Coefficients relate to profile characteristics (mean, shear, curvature)

### Framework Validation
The C-WiSPr framework has been validated against:
- **15 NYSM Profiler Stations**: Comprehensive lidar measurements across New York State  
- **NOW23 Dataset**: High-resolution wind profile observations
- **Multiple Wind Datasets**: Wangara, Høvsøre, Leipzig field campaigns

## 📈 Performance Metrics

The framework demonstrates excellent performance with typical metrics:
- **R² > 0.95**: High correlation between original and reconstructed profiles
- **RMSE < 0.5 m/s**: Low reconstruction errors
- **MAE < 0.3 m/s**: Minimal absolute errors
- **Compression Ratio > 10:1**: Efficient data representation

## 🤝 Contributing

We welcome contributions to enhance the C-WiSPr framework:

1. **Bug Reports**: Submit issues for any bugs or unexpected behavior
2. **Feature Requests**: Suggest new functionalities or improvements
3. **Code Contributions**: Submit pull requests with enhancements
4. **Documentation**: Improve documentation and examples

## 📄 Citation

If you use C-WiSPr in your research, please cite:

```bibtex
@article{cwispr2024,
  title={A Chebyshev Polynomial-based Wind Speed Profile Characterization Framework: Applications in Mesoscale Model Evaluation},
  author={[Harish Baki, Sukanta Basu]},
  journal={Wind Energy},
  year={2025},
  publisher={Wiley},
  note={Minor revision}
}
```

## 📞 Contact

For questions, collaborations, or support:
- **Primary Contact**: [Harish Baki, Postdoctoral Associate, University at Albany, hbaki@albany.edu]
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for general questions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Developed for advancing wind energy research through innovative mathematical frameworks** 🌬️⚡
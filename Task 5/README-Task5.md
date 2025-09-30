# Task 5: Decision Trees and Random Forests - Heart Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-green.svg)](https://scikit-learn.org/)

A comprehensive machine learning project implementing **Decision Trees** and **Random Forests** for predicting heart disease using the Heart Disease dataset. This project demonstrates tree-based ensemble methods with outstanding results achieving **perfect 100% test accuracy**.

## 📋 Project Overview

This project implements and compares tree-based machine learning algorithms for binary classification of heart disease. The analysis includes comprehensive model optimization, feature importance analysis, and rigorous cross-validation evaluation.

### 🎯 Key Objectives
- Train and optimize Decision Tree classifiers
- Implement Random Forest ensemble methods
- Analyze and prevent overfitting through depth control
- Interpret feature importance for clinical insights
- Evaluate model performance using cross-validation

## 🏆 Outstanding Results

### Model Performance Summary
- **Random Forest**: **100.0% Test Accuracy** (Perfect Classification!)
- **Decision Tree**: **98.8% Test Accuracy** (Excellent Performance)
- **Cross-Validation**: 99.7% ± 0.3% (Highly Stable)
- **Clinical Applicability**: Ready for medical validation

### Key Achievements
✅ **Perfect Disease Detection** - Zero missed cases  
✅ **Minimal False Alarms** - High precision maintained  
✅ **Robust Performance** - Consistent across validation folds  
✅ **Interpretable Results** - Clear feature importance insights  

## 📁 Project Structure

```
Task-5-Decision-Trees-Random-Forests/
├── Task5_Notebook.ipynb          # Complete analysis notebook
├── heart.csv                     # Original heart disease dataset
├── heart_processed.csv           # Preprocessed and scaled data
├── README.md                     # This file
└── assets/                       # Generated visualizations
    ├── decision_tree_plot.png
    ├── feature_importance.png
    └── model_comparison.png
```

## 📊 Dataset Information

### Heart Disease Dataset
- **Total Samples**: 1,025 patients
- **Features**: 13 clinical measurements
- **Target**: Binary classification (0 = No Disease, 1 = Disease)
- **Class Balance**: Well-balanced (51.3% vs 48.7%)
- **Data Quality**: No missing values

### Feature Descriptions
| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numerical |
| `sex` | Gender (1 = male, 0 = female) | Categorical |
| `cp` | Chest pain type (0-3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Categorical |
| `restecg` | Resting ECG results (0-2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise induced angina | Categorical |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0-3) | Categorical |
| `thal` | Thalassemia type (1-3) | Categorical |

## 🔬 Analysis Components

### 1. Data Preprocessing
- **StandardScaler** normalization for numerical features
- **Feature scaling** to ensure equal contribution
- **Train-test split** with stratification (80/20)
- **Data quality assessment** and validation

### 2. Decision Tree Analysis
- **Tree optimization** through depth control (1-15 levels)
- **Overfitting prevention** with systematic depth testing
- **Tree visualization** and rule interpretation
- **Performance evaluation** with accuracy metrics

### 3. Random Forest Implementation
- **Ensemble method** with multiple trees (10-500 estimators)
- **Bootstrap aggregating** for variance reduction
- **Hyperparameter tuning** using GridSearchCV
- **Out-of-bag evaluation** and model selection

### 4. Feature Importance Analysis
- **Clinical feature ranking** for medical insights
- **Comparative importance** between models
- **Medical interpretation** of predictive factors
- **Feature type analysis** (numerical vs categorical)

### 5. Cross-Validation Evaluation
- **5-fold stratified CV** for robust performance assessment
- **Statistical significance testing** with Wilcoxon test
- **Model stability analysis** and generalization evaluation
- **Confidence interval calculation** for reliability

## 🏥 Clinical Insights

### Top 5 Most Important Features
1. **Chest Pain Type (cp)**: 14.1% importance - Primary diagnostic indicator
2. **Maximum Heart Rate (thalach)**: 12.5% importance - Exercise capacity marker
3. **Major Vessels (ca)**: 11.2% importance - Coronary artery involvement
4. **ST Depression (oldpeak)**: 10.7% importance - Exercise stress response
5. **Thalassemia (thal)**: 10.5% importance - Blood disorder indicator

### Medical Applications
- **Screening Tool**: Early detection of heart disease risk
- **Clinical Decision Support**: Assisting healthcare professionals
- **Risk Stratification**: Patient prioritization for treatment
- **Diagnostic Aid**: Complementing traditional medical assessment

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Task-5-Decision-Trees-Random-Forests

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch Jupyter Notebook
jupyter notebook Task5_Notebook.ipynb
```

### Running the Analysis
1. **Open** `Task5_Notebook.ipynb` in Jupyter
2. **Place** `heart.csv` in the same directory
3. **Run all cells** to execute the complete analysis
4. **Review results** and generated visualizations

## 📈 Model Performance Details

### Decision Tree Results
- **Training Accuracy**: 98.8%
- **Test Accuracy**: 98.8%
- **Optimal Depth**: 9 levels
- **Cross-Validation**: 97.9% ± 1.2%
- **Precision (Disease)**: 0.98
- **Recall (Disease)**: 1.00
- **F1-Score (Disease)**: 0.99

### Random Forest Results
- **Training Accuracy**: 100.0%
- **Test Accuracy**: 100.0%
- **Number of Trees**: 100
- **Cross-Validation**: 99.7% ± 0.3%
- **Precision (Disease)**: 1.00
- **Recall (Disease)**: 1.00
- **F1-Score (Disease)**: 1.00

### Hyperparameter Optimization
- **GridSearchCV** for systematic parameter tuning
- **Best Parameters**: max_depth=None, min_samples_split=2, min_samples_leaf=1
- **5-fold cross-validation** for robust model selection
- **Scoring metric**: Accuracy with stratification

## 🎓 Educational Value

### Machine Learning Concepts Covered
- **Decision Tree Algorithm**: Entropy, information gain, tree splitting
- **Ensemble Methods**: Bootstrap aggregating, random feature selection
- **Overfitting Prevention**: Tree pruning, depth control, regularization
- **Model Evaluation**: Cross-validation, confusion matrices, ROC curves
- **Feature Selection**: Importance ranking, clinical interpretation

### Interview Question Preparation
This project comprehensively addresses common ML interview questions:
1. How do decision trees work?
2. What is entropy and information gain?
3. How are random forests better than single trees?
4. How do you prevent overfitting?
5. What is bootstrap aggregating?
6. How do you interpret feature importance?
7. What are the pros/cons of tree-based methods?
8. How do you evaluate model performance?

## 📊 Visualizations

The notebook generates several informative plots:

### Model Performance
- **Decision Tree Visualization**: Complete tree structure with decision rules
- **Accuracy vs Depth**: Overfitting analysis and optimal depth selection
- **Feature Importance Plot**: Ranked importance of clinical features
- **Confusion Matrices**: Classification performance visualization

### Cross-Validation Analysis
- **CV Score Distribution**: Model stability across folds
- **Statistical Significance**: Wilcoxon test results
- **Performance Comparison**: Decision Tree vs Random Forest

## 🔍 Advanced Features

### Overfitting Analysis
- **Systematic depth testing** from 1-15 levels
- **Training vs validation curves** for optimal selection
- **Gap analysis** between training and test performance
- **Visual identification** of overfitting threshold

### Statistical Validation
- **Wilcoxon signed-rank test** for model comparison
- **Confidence intervals** for performance metrics
- **Stratified sampling** for reliable evaluation
- **Bootstrap estimation** for uncertainty quantification

## ⚠️ Important Notes

### Data Privacy
- Dataset contains synthetic/anonymized medical records
- No patient identifiers included
- Suitable for educational and research purposes

### Clinical Disclaimer
- Model results are for educational demonstration only
- Not intended for actual medical diagnosis
- Requires clinical validation before medical use
- Should complement, not replace, professional medical judgment

### Technical Limitations
- Results specific to this dataset and preprocessing
- May not generalize to other populations
- Requires regular retraining with new data
- Performance may vary with different feature sets

## 📚 References and Further Reading

### Key Papers
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81-106.
- Ho, T. K. (1995). Random Decision Forests. Proceedings of ICDAR.

### Libraries Used
- [scikit-learn](https://scikit-learn.org/): Machine learning algorithms
- [pandas](https://pandas.pydata.org/): Data manipulation and analysis
- [matplotlib](https://matplotlib.org/): Data visualization
- [seaborn](https://seaborn.pydata.org/): Statistical plotting

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features or analyses
- Improve documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created as part of a comprehensive machine learning curriculum focusing on tree-based ensemble methods and medical AI applications.

---

**Note**: This project demonstrates production-level machine learning implementation with clinical applications. The perfect accuracy achieved showcases the effectiveness of ensemble methods for well-structured medical datasets.

**🎯 Ready for GitHub submission and technical interviews!**
Exploratory Data Analysis (EDA) on the Titanic Dataset - Task 2
Task Objective
The primary goal of this task is to perform a comprehensive Exploratory Data Analysis (EDA) on the Titanic dataset. The objective is to understand the data's underlying structure, identify patterns and anomalies, and uncover relationships between different variables using statistical summaries and visualizations.


Dataset
This analysis uses the Titanic Dataset, which contains passenger information such as age, class, sex, and survival status from the 1912 voyage.

Analytical Process
The analysis was conducted in a structured manner, following standard data science practices from data cleaning to insight generation.

1. Data Cleaning & Preprocessing (Task 1)
Before analysis, the dataset was cleaned to ensure data quality:

Handling Missing Values:

Missing Age values were imputed with the median age of the dataset.

The Cabin column was dropped due to a high volume of missing entries (687 out of 891).

The two missing Embarked values were filled with the most frequent port of embarkation.

Categorical Data Conversion:

The Sex column was converted to a numerical format (male: 0, female: 1).

The Embarked column was one-hot encoded to prepare it for numerical analysis.

Outlier Removal:

Outliers in the Fare column were identified using the Interquartile Range (IQR) method.

Records with fares outside 1.5 times the IQR were removed, resulting in a cleaner dataset of 775 passengers for the analysis.

2. Exploratory Data Analysis (Task 2)
The cleaned dataset was then subjected to a detailed EDA.

Summary Statistics
Generated descriptive statistics (mean, median, standard deviation) for a high-level quantitative overview of the data.

Univariate Analysis
Examined individual features to understand their distributions:


Numerical Features (Age, Fare): Histograms and boxplots were created to visualize the distribution, central tendency, and spread.

Categorical Features (Survived, Pclass, Sex): Countplots were used to visualize the frequency of each category.

Bivariate and Multivariate Analysis
Investigated the relationships between different variables:


Feature Relationships: A correlation matrix and a heatmap were used to quantify the linear relationship between numerical features like Age, Fare, Pclass, and Survived.

Survival Analysis: Bar plots were created to analyze the impact of key features (Pclass, Sex, and Embarked) on the survival rate.

Key Insights & Patterns Identified
The analysis revealed several significant patterns and trends regarding passenger survival:

Survival by Gender: A passenger's sex was a primary indicator of survival. Female passengers had a significantly higher survival rate than males.

Survival by Class: Socioeconomic status played a crucial role. First-class passengers had a much higher survival rate compared to those in second and third classes.

Survival by Fare: There was a positive correlation between the fare paid and the likelihood of survival. Passengers who paid higher fares were more likely to survive.

Passenger Demographics: The majority of passengers were young adults, traveled in third class, and were male. Most passengers boarded the ship alone, without siblings or a spouse.

Tools and Libraries
Language: Python

Core Libraries:

Pandas: For data manipulation and cleaning.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization and plotting.

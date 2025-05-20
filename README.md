# 🌦️ Rainfall Prediction in Australia Using Machine Learning
📘 Overview
This project aims to predict whether it will rain the next day in various locations across Australia using historical weather data. The dataset, sourced from the Australian Bureau of Meteorology, contains daily weather observations from numerous Australian weather stations. The target variable is RainTomorrow, indicating whether it rained the following day.
RDocumentation
+1
GitHub
+1

📂 Dataset Description
The dataset comprises over 140,000 daily observations with the following features:
RDocumentation

Date: Date of observation

Location: Name of the weather station

MinTemp: Minimum temperature (°C)

MaxTemp: Maximum temperature (°C)

Rainfall: Amount of rainfall (mm)

Evaporation: Evaporation (mm)

Sunshine: Sunshine (hours)

WindGustDir: Direction of strongest wind gust

WindGustSpeed: Speed of strongest wind gust (km/h)

WindDir9am: Wind direction at 9am

WindDir3pm: Wind direction at 3pm

WindSpeed9am: Wind speed at 9am (km/h)

WindSpeed3pm: Wind speed at 3pm (km/h)

Humidity9am: Humidity at 9am (%)

Humidity3pm: Humidity at 3pm (%)

Pressure9am: Atmospheric pressure at 9am (hPa)

Pressure3pm: Atmospheric pressure at 3pm (hPa)

Cloud9am: Cloud cover at 9am (oktas)

Cloud3pm: Cloud cover at 3pm (oktas)

Temp9am: Temperature at 9am (°C)

Temp3pm: Temperature at 3pm (°C)

RainToday: Whether it rained today (Yes/No)

RainTomorrow: Target variable — whether it will rain tomorrow (Yes/No)
arXiv
+3
RDocumentation
+3
GitHub
+3
GitHub
+1
RDocumentation
+1

🔄 Data Preprocessing
1. Handling Missing Values
Numerical Features: Imputed missing values using median values.

Categorical Features: Imputed missing values using the most frequent category.

2. Feature Engineering
Date Feature: Extracted month from the Date column to capture seasonal patterns.

RainToday: Converted 'Yes'/'No' to binary values (1/0).
GitHub

3. Encoding Categorical Variables
Applied Label Encoding to transform categorical variables into numerical format.

📊 Exploratory Data Analysis (EDA)
Distribution Analysis: Plotted histograms for numerical features to understand distributions.

Correlation Matrix: Generated heatmaps to identify correlations between features.

Rainfall Patterns: Analyzed rainfall patterns across different locations and months.
GitHub

🧠 Model Building
1. Data Splitting
Split the dataset into training and testing sets using an 80/20 ratio.

2. Model Selection
Implemented the following classification models:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

3. Model Evaluation
Evaluated models using metrics such as accuracy, precision, recall, and F1-score.

Plotted ROC curves to compare model performances.
GitHub

🔍 Feature Importance
Utilized feature importance scores from the Random Forest and XGBoost models to identify the most influential features in predicting rainfall.

📈 Results
Best Model: XGBoost Classifier achieved the highest accuracy.

Key Features: Humidity at 3pm, Rainfall, and Pressure at 9am were among the top predictors.
GitHub
RDocumentation
+1
GitHub
+1

🛠️ Requirements
Ensure the following libraries are installed:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
🚀 How to Run
Clone the repository or download the notebook.

Ensure the dataset (weatherAUS.csv) is in the working directory.

Run the notebook sequentially to reproduce the results.

👤 Author
SAMIKSHYA PRUSETH

📄 License
This project is licensed under the MIT License.

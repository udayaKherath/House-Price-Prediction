# House Price Prediction ML Project Documentation

## Introduction
This document provides an overview of the House Price Prediction Machine Learning (ML) project. The goal of this project is to predict house prices based on various features such as square footage, number of bedrooms, location, etc. 

### Dataset https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data

## Preprocessing

### Data Cleaning
- **Handling Missing Values**: We addressed missing values in the dataset by imputing them with the mean of the respective feature. This approach helped us retain valuable data without introducing bias.
- **Handling Outliers**: Outliers were detected using the interquartile range (IQR) method. We then applied winsorization to cap extreme values, ensuring that they did not unduly influence our model's predictions.

### Feature Engineering
- **Creating New Features**: We created several new features to augment the dataset. For instance, we introduced a "total_rooms" feature by summing the number of bedrooms and bathrooms, which provided additional context for predicting house prices.
- **Encoding Categorical Variables**: Categorical variables were encoded using one-hot encoding to transform them into numerical form. This allowed our models to effectively interpret and utilize these variables during training.

### Normalization/Scaling
- **Scaling Numerical Features**: Numerical features were scaled using standardization (z-score normalization). This ensured that features with different scales did not disproportionately impact the performance of our models.

### Splitting Data
- **Training-Testing Split**: We divided the dataset into training and testing sets, with 80% of the data allocated for training and 20% for testing. This allowed us to assess the generalization performance of our models on unseen data.

### Feature Selection
- **Selecting Relevant Features**: To identify the most relevant features for predicting house prices, we employed recursive feature elimination (RFE). This helped us streamline our dataset by retaining only the features with the greatest predictive power.

These preprocessing steps were essential in preparing our dataset for training machine learning models. By addressing missing values, engineering informative features, and ensuring data uniformity through scaling, we optimized our dataset for effective model training and prediction.

- **Model Selection**: Overview of the different ML models experimented with and rationale for choosing the final model.
- **Model Evaluation**: Evaluation metrics used to assess the performance of the ML models.
- **Deployment**: Discussion on how the final model is deployed for real-world use. 

## Model Selection
Several ML models were experimented with, including:
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Neural Networks

## Model Evaluation
Evaluation metrics used to assess the performance of the model include:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R2)

After thorough experimentation and evaluation, we found that the Linear regression model consistently outperformed other models in terms of predictive accuracy. The neural network demonstrated superior performance in capturing complex patterns within the dataset, making it the most suitable choice for our house price prediction task.

![Figure 1](https://raw.githubusercontent.com/udayaKherath/House-Price-Prediction/master/img1.png)

## Deployment
The final model is deployed using Flask API. Users can input house features and receive a predicted price.

## Conclusion
This project demonstrates the process of developing a house price prediction ML model from data preprocessing to model deployment. Further improvements could include incorporating additional features, fine-tuning hyperparameters


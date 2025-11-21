# CancerRecurrencePredictor-ML-Showcase
A Java-based machine learning pipeline simulation for predicting cancer recurrence risk, featuring K-Fold Cross-Validation and performance metric analysis. Developed for PhD application showcase.
# Project Overview

This project is a self-contained, end-to-end simulation of a machine learning (ML) pipeline designed to predict the risk of cancer recurrence in patients. It demonstrates strong foundations in object-oriented programming (OOP), data handling, rigorous model evaluation, and computational thinking—all critical skills for biomedical informatics and computational biology research.

The primary goal is to showcase the methodology used to build, train, and validate a predictive model aimed at improving cancer patient outcomes through early risk stratification.

## 🛠️ Technology Stack

Language: Java (JDK 8+)

Libraries: Standard Java Utilities (No external ML libraries required)

Methodology: Simulated Weighted Linear Model (Conceptually similar to Logistic Regression)

Validation: 5-Fold Cross-Validation

## 💡 Methodology and ML Pipeline

The project simulates four key stages of a real-world machine learning application:

### 1. Data Simulation (generateSimulatedData / PatientRecord)

The PatientRecord class defines the input features (clinical variables) and the ground truth outcome (recurrence status).

Features Used:

- tumorSize (e.g., 1–50 mm)

- grade (Aggressiveness: 1, 2, or 3)

- lymphNodeStatus (Presence of lymph node metastasis: 0 or 1)

A dataset of 150 records is synthesized where recurrence status is correlated with higher-risk features (larger tumors, higher grade), providing meaningful data for the model to "learn" from.

### 2. Model Training and Prediction (MLModelSimulator)

The MLModelSimulator acts as the core classifier.

Training (train method): The model calculates feature weights by assessing how much each feature contributes to the recurrence outcome in the training data. This process simulates finding the optimal parameters of a linear model.

Prediction (predict method): For any new patient, the model calculates a Risk Score by summing the weighted features:


$$\text{Risk Score} = (F_1 \times W_1) + (F_2 \times W_2) + (F_3 \times W_3) - \text{Bias}$$


If the Risk Score is positive (above the internal threshold), recurrence (1) is predicted.

### 3. Rigorous Validation: K-Fold Cross-Validation

To ensure the model is robust and not biased by a single data split, the project implements 5-Fold Cross-Validation (CV):

The dataset is divided into 5 equal folds.

The model is trained 5 separate times. In each iteration, 4 folds are used for training and the remaining 1 fold is used for testing.

This ensures every data point is tested exactly once, and the final reported metrics are the average of all 5 runs, guaranteeing a reliable measure of performance.

## 🚀 How to Run the Project

This is a single-file Java application.

Save the Code: Save the entire content of the code block as CancerRecurrencePredictor.java.

Compile: Open your terminal or command prompt, navigate to the directory where you saved the file, and compile it:

**javac CancerRecurrencePredictor.java**


Execute: Run the compiled class file:

**java CancerRecurrencePredictor**


The console output will display the cross-validation process fold-by-fold, including the Confusion Matrix visualization for each, and the final averaged performance metrics.

Developed as a showcase project for a PhD application to NUS Yong Loo Lin School of Medicine.

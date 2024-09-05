# Automated Chest X-Ray Classification Using Deep Learning

## Overview

This project aims to develop a highly accurate and interpretable deep learning-based tool for the automated classification of chest X-ray images into three categories: Normal, Pneumonia, and COVID-19. The objective is to enhance diagnostic support for radiologists by providing a reliable and interpretable model.

## Objectives

1. **Create a High-Quality Dataset**: Develop a balanced dataset of chest X-ray images, including diverse cases of Normal, Pneumonia, and COVID-19, to ensure a comprehensive foundation for training reliable models.
   
2. **Achieve High Classification Accuracy**: Develop a deep learning model with at least 90% classification accuracy on the test set to differentiate effectively between Normal, Pneumonia, and COVID-19 cases.
   
3. **Ensure Balanced Performance**: Ensure the model achieves a minimum F1-score of 0.9 across all classes, indicating balanced performance between precision and recall for each category.
   
4. **Demonstrate Model Interpretability**: Utilize Grad-CAM visualisations to highlight clinically relevant areas in X-ray images, ensuring that model predictions are based on appropriate image regions.

## Project Workflow

1. **Data Collection and Preprocessing**: 
   - Curate a diverse dataset of chest X-ray images from open-source repositories.
   - Apply preprocessing steps such as resizing, normalization, and augmentation to improve data quality.

2. **Model Development**:
   - Implement various deep learning architectures, including EfficientNet B1 and DenseNet121, for classification tasks.
   - Fine-tune models using transfer learning, adjust hyperparameters, and apply regularization techniques like early stopping to optimize performance.

3. **Performance Evaluation**:
   - Evaluate models using metrics such as accuracy, F1-score, precision, and recall to ensure high diagnostic performance.
   - Visualise model performance through confusion matrices and other graphical representations.

4. **Model Interpretability**:
   - Use Grad-CAM visualisations to interpret model predictions and validate that decisions are based on relevant medical features in the X-ray images.

5. **Outcome and Objective Validation**:
   - Validate the model's performance and interpretability against the objectives, confirming its suitability for clinical application.

## Results

- The project successfully created a balanced and diverse dataset of chest X-ray images, forming a robust foundation for model training.
- The EfficientNet B1 and DenseNet121 models achieved classification accuracies exceeding 90%, meeting the project's performance benchmarks.
- The models also met the objective of achieving an F1-score of at least 0.9 across all classes, demonstrating balanced performance.
- Grad-CAM visualisations effectively highlighted clinically relevant areas in the X-ray images, confirming the model's interpretability and enhancing trust in its clinical applicability.

## Conclusion

The developed deep learning-based tool for automated chest X-ray classification meets the project's objectives of accuracy, balanced performance, and interpretability. This tool provides a valuable diagnostic support system for radiologists, capable of reliably distinguishing between Normal, Pneumonia, and COVID-19 cases. Future work will focus on further refining the models and exploring additional techniques to enhance performance and usability.

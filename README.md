# Admissions Data Prediction with Neural Network

This project demonstrates how to use a neural network model to predict admissions data using TensorFlow and Keras. The model is built and evaluated using various machine learning techniques including data preprocessing, feature scaling, model design, and performance evaluation.

## Code Overview

### Libraries Used:
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Array operations and numerical computations.
- **Matplotlib**: Plotting graphs and charts.
- **TensorFlow/Keras**: Building and training a neural network model.
- **Scikit-learn**: Data splitting, scaling, and model evaluation.

### Steps:
1. **Data Loading**:
   - The dataset is loaded from a CSV file named `admissions_data.csv`.

2. **Data Preprocessing**:
   - The dataset is split into features (`features`) and labels (`labels`).
   - The features include columns 1 to 7, and the label is in the last column.

3. **Train-Test Split**:
   - The data is split into training and testing sets using `train_test_split` with a 75-25% ratio.

4. **Feature Scaling**:
   - The features are standardized using `StandardScaler` to ensure they are on the same scale for model training.

5. **Model Design**:
   - A neural network model is designed using Keras with:
     - **Input Layer**: The number of input features.
     - **Hidden Layer 1**: 16 neurons with ReLU activation function.
     - **Dropout Layer 1**: To prevent overfitting.
     - **Hidden Layer 2**: 8 neurons with ReLU activation function.
     - **Dropout Layer 2**: Another dropout layer.
     - **Output Layer**: 1 neuron (for regression output).
   - The model is compiled using Adam optimizer with a learning rate of 0.005 and Mean Squared Error (MSE) as the loss function.

6. **Model Training**:
   - The model is trained using the training data for 100 epochs with a batch size of 8. 
   - Early stopping is applied to prevent overfitting by monitoring the validation loss.

7. **Model Evaluation**:
   - After training, the model is evaluated on the test set using Mean Absolute Error (MAE) and R-squared score (`r2_score`).
   - The MAE value and R-squared score are printed to assess model performance.

8. **Plotting Training History**:
   - Training and validation MAE, loss, and validation loss are plotted across epochs to visualize the model's performance during training.

### Code Highlights:
- **Early Stopping**: Stops the training early if the validation loss does not improve, saving time and preventing overfitting.
- **Dropout Layers**: Used to prevent overfitting by randomly setting some neurons' outputs to zero during training.
- **Model Evaluation**: MAE is calculated for performance, and R-squared score is used to measure how well the model's predictions fit the actual data.
- **Visualization**: Graphs of MAE and loss metrics over epochs provide insights into how the model is learning.

## Requirements
- `tensorflow`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `numpy`

#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

class Perceptron:
    #Inititalize the perceptron class
    
    def __init__(self, learning_rate = 0.1, max_epochs = 1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
    
    #Create a method to fit the perceptron, with X as the data and y as the labels(outcome I think lol)
    
    def fit(self, X, y):
    # Add a bias term (x0 = 1) to the input data, accounting for threshold data
    #Create the augmented feature matrix
        X = np.c_[np.ones(X.shape[0]), X]
    #Create weight vector(initialize at 0)
        self.weights = np.zeros(X.shape[1])
        
        for epoch in range(self.max_epochs):
            #Initialize the # of misclassified data points at 0 (counter)
            misclassified = 0
            for i in range(X.shape[0]):
            #If dot product is ≤ 0, then it is misclassified.
                if y[i] * np.dot(X[i], self.weights) <= 0:
                    #Updating the params
                    self.weights += self.learning_rate * y[i] * X[i]
                    #And count it as a misclassification
                    misclassified += 1
            if misclassified == 0:
                break
            #Aka don't do anything it was correctly classified
            
            #Now I'll create a method for making predictions
    
    def predict(self, X):
        # Add a bias term (x0 = 1) to the input data
        X = np.c_[np.ones(X.shape[0]), X]
        #Now make predictions sign function 
        #applied to the dot product of the test data and the weight vector
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(np.dot(X[i], self.weights))
            predictions.append(prediction)
        return predictions


#Sample data vectors
X_train = np.array([[-1, -1], [1, 1], [1, -1], [-1, 1], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
y_train = np.array([1, 1, -1, -1, 1, 1, -1, -1])

#Class A: 1
#Class B: -1

#Creating a Perceptron instance
perceptron = Perceptron(learning_rate=0.1, max_epochs=1000)

# Train the Perceptron
perceptron.fit(X_train, y_train)

# Define some test data
X_test = np.array([[0, 0], [2, 2], [0, 2], [2, 0]])

# Make predictions
predictions = perceptron.predict(X_test)



# Display the predictions
for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Data point {X_test[i]} belongs to Class A")
    else:
        print(f"Data point {X_test[i]} belongs to Class B")


import numpy as np
import matplotlib.pyplot as plt 
from utils import *
from sklearn.metrics import confusion_matrix

# Data is imported from utils 

# # Create a training set and a test set of size 500 and 100 respectively
threshold = 100
X_train,Y_train = create_train_data(X,Y,start=0,end=500,threshold=threshold)
X_test,Y_test = create_test_data(X,Y,start=500,end=600,threshold=threshold)

# Number of examples
training_examples = X_train.shape[0]
# Features
features = X_train.shape[1]
# Classes 
k = 10

# Weights and Bias
W = np.random.randn(features,k)
b = np.zeros((1,k))

# learning rate and iterations 
lr = 0.1
iterations = 10000
W,b,costs = model(X_train,Y_train,W,b,lr,iterations,[],training_examples)

Y_pred = predictions(X_test,W,b)

accuracy = calc_accuracy(Y_pred,Y_test)

print(f"Learning Rate {lr}, Final Test Accuracy {accuracy}")

xlabel = 'Iterations'
ylabel = 'Cost'
xticks = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
title = f"Task 1: Cost vs iterations plot for learning rate {lr}"
name = 'Task1_0.1.jpg'
plot_training_data(range(iterations+1),costs,xlabel,ylabel,xticks,title,name)

cmat = confusion_matrix(Y_pred,Y_test)
numbers = [0,1,2,3,4,5,6,7,8,9]
accuracy_per_number = []
for num in digits:
    accuracy_per_number.append(np.round(cmat[num,num]/np.sum(cmat[:,num])*100,1))     

xlabel = 'Digits'
ylabel = 'Accuracy(%)'
plot_title = 'Task1 - Accuracy Plot - learning rate - ' + str(lr)
plot_name = 'task1_acc_plot_'+str(lr)+'.jpg'

plot_percentage_accuracy_per_digit(numbers,accuracy_per_number,xlabel,ylabel,plot_title,plot_name)



threshold = 100
X_train,Y_train = create_train_data(X,Y,start=0,end=500,threshold=threshold)
X_test,Y_test = create_test_data(X,Y,start=500,end=600,threshold=threshold)

# Weights and Bias
W = np.random.randn(features,k)
b = np.zeros((1,k))

lr = 0.01
iterations = 10000
W,b,costs = model(X_train,Y_train,W,b,lr,iterations,[],training_examples)

Y_pred = predictions(X_test,W,b)

accuracy = calc_accuracy(Y_pred,Y_test)

print(f"Learning Rate {lr}, Final Test Accuracy {accuracy}")

xlabel = 'Iterations'
ylabel = 'Cost'
xticks = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
title = f"Task 1: Cost vs iterations plot for learning rate {lr}"
name = 'Task1_0.01.jpg'
plot_training_data(range(iterations+1),costs,xlabel,ylabel,xticks,title,name)

cmat = confusion_matrix(Y_pred,Y_test)
numbers = [0,1,2,3,4,5,6,7,8,9]
accuracy_per_number = []
for num in digits:
    accuracy_per_number.append(np.round(cmat[num,num]/np.sum(cmat[:,num])*100,1))     

xlabel = 'Digits'
ylabel = 'Accuracy(%)'
plot_title = 'Task1 - Accuracy Plot - learning rate - ' + str(lr)
plot_name = 'task1_acc_plot_'+str(lr)+'.jpg'

plot_percentage_accuracy_per_digit(numbers,accuracy_per_number,xlabel,ylabel,plot_title,plot_name)

threshold = 100
X_train,Y_train = create_train_data(X,Y,start=0,end=500,threshold=threshold)
X_test,Y_test = create_test_data(X,Y,start=500,end=600,threshold=threshold)

# Weights and Bias
W = np.random.randn(features,k)
b = np.zeros((1,k))

lr = 0.001
iterations = 10000
W,b,costs = model(X_train,Y_train,W,b,lr,iterations,[],training_examples)

Y_pred = predictions(X_test,W,b)

accuracy = calc_accuracy(Y_pred,Y_test)

print(f"Learning Rate {lr}, Final Test Accuracy {accuracy}")

xlabel = 'Iterations'
ylabel = 'Cost'
xticks = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
title = f"Task 1: Cost vs iterations plot for learning rate {lr}"
name = 'Task1_0.001.jpg'
plot_training_data(range(iterations+1),costs,xlabel,ylabel,xticks,title,name)

cmat = confusion_matrix(Y_pred,Y_test)
numbers = [0,1,2,3,4,5,6,7,8,9]
accuracy_per_number = []
for num in digits:
    accuracy_per_number.append(np.round(cmat[num,num]/np.sum(cmat[:,num])*100,1))     

xlabel = 'Digits'
ylabel = 'Accuracy(%)'
plot_title = 'Task1 - Accuracy Plot - learning rate - ' + str(lr)
plot_name = 'task1_acc_plot_'+str(lr)+'.jpg'

plot_percentage_accuracy_per_digit(numbers,accuracy_per_number,xlabel,ylabel,plot_title,plot_name)

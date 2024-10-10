import math as m
import numpy as np
import matplotlib.pyplot as plt

inp, y_real = [[-235, 7, 24, 14, 1, 9, 6.235], 
               [-240, 5.45, 16, 12.24, 0.25, 8.234, -12.364], 
               [-159.25, 10, 26.035, 17.2, 3, 13, 9.13606]], \
[-1, -1, 1] # -1 == class 'A'     1 == class 'B'

weights = [0.7, 0.4, -1.53, 1.612, -0.88124, 1.351, 0.266]
bias = 0.2
learning_rate = 0.02


# the goal of an SVM is to find the hyperplane that would maximize the distance between the hyperplane and all of classes' points
## hard and soft margin


# soft margin 
N = len(inp)
lr = 0.03
e = 0.003
C = 0.0512

def hinge_loss(y_groundTruth, y_pred) -> float:
    return max(0, 1-y_groundTruth*y_pred)


def dhinge_dw(y_groundTruth, y_pred, x_input: list): 
    yreal_by_input = [-y_groundTruth* x_input[j] for j in range(len(x_input))]
    return yreal_by_input if 1-(y_groundTruth*y_pred)>0 else [0]*len(x_input)
        
def dhinge_db(y_groundTruth): return y_groundTruth



epochs = 100
epochs_lst = []
loss = []
for epoch in range(epochs):
    epochs_lst.append(epoch)

    # pred = w*x-b
    y_pred = [sum(inp[i][k]*weights[k] for k in range(len(weights))) for i in range(len(inp)) ] 
    y_pred = [y_pred[i]+bias for i in range(len(y_pred))]   

    hinge_loss_value = [1/N * sum(hinge_loss(y_real[i], y_pred[i]) for i in range(N))]
    loss.append(hinge_loss_value[0])


    # sum of weight vectors (dhinge_dw) for all samples
    w_gradient = [dhinge_dw(y_real[i],y_pred[i], inp[i]) for i in range(len(inp))]
#    w_gradient = [sum(w_gradient[i][j]) for i in range(len(w_gradient)) j in range(len(w_gradient[0]))]
    w_gradient = [sum(w_gradient[i][j] for i in range(len(w_gradient))) for j in range(len(w_gradient[0]))]
    #print(f"w_gradient: {w_gradient}") 
    b_gradient = dhinge_db(bias)
    #print(f"b_gradient: {b_gradient}") 


    weights = [weights[q]-lr*w_gradient[q] for q in range(len(weights))] 
    bias = bias-lr*b_gradient
    
    magnitute = 0.5*(m.sqrt(sum(weights[i]**2 for i in range(len(weights))))**2)
    print(f"Epoch {epoch+1}/{epochs}   loss: {hinge_loss_value},   magnitute: {magnitute }\n")

    
print(f"epochs_lst: {epochs_lst}\n\n loss: {loss}")
plt.plot(epochs_lst, loss); plt.show()


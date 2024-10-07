
## without any libraries
# forward -> calculate the loss -> backward by the derivatives

import math 


def relu(x: float): return max([0, x])
def derivative_relu(x: float): return 1 if x>=0 else 0

def leaky_relu(x: float, neg_slope=0.3): return x if x>=0 else neg_slope*x
def derivative_leaky_relu(x: float, neg_slope: 0.3): 1 if x>=0 else neg_slope

def sigmoid(x: float): return 1/ (1 + math.exp(-x))

def tanh(x: float): return (math.exp(x)-math.exp(-x) / math.exp(x)+math.exp(-x))


def MSE_costF(y_real, y_predicted):
    return (y_real-y_predicted)**2

def derivative_MSE_costF(y_real, y_predicted):
    return 2*(y_real-y_predicted)


act_functions = [relu, leaky_relu, sigmoid, tanh]

inp, y_real = [[-235, 7, 24, 14, 1, 9, 6.235], 
               [-240, 5.45, 16, 12.24, 0.25, 8.234, -12.364], 
               [-159.25, 10, 26.035, 17.2, 3, 13, 9.13606]], \
[5.5346, 27.235, 15.788]


first_weights = [ [0.124, 0.564, 0.8, -1.74], 
           [-0.124, 2, 1.2342, 0.131],
           [1.64, -0.9134, 0.53, 0.8],
           [0.7, 0.4, -1.53, 1.612],
           [-0.2332, 0.134, 0.778, 0.37],
           [0.3213, -0.82, 1.42, 0.23],
           [1.15, 0.664, -0.35, -0.12] ] 
biases = [0.2, 0.3, -0.65, 0.12]  
last_weights = [[2.12], [-0.12], [0.7823], [1.5613]]  
learning_rate = 0.02

print(len(inp), len(inp[0]), len(first_weights[0])) ## 3, 7, 4




epochs = 25
for epoch in range(epochs):
    print(f"Epoch {epoch}... result: ")
    
    ## x*first_weights # 3x7 * 7x4 = 3x4
    hidden_layer = []
    for row_x in range(len(inp)): 
        row = []
        for col_w in range(len(first_weights[0])):
            
            dot = 0
            for col_x in range(len(inp[0])):
                dot += inp[row_x][col_x]*first_weights[col_x][col_w]
            row.append(dot)
        hidden_layer.append(row)
            
    print(len(hidden_layer))
    print(hidden_layer)



    ## hidden_layer_neurons + bias
    for sample in range(len(hidden_layer)):
        for b_indx, b in enumerate(biases):
            hidden_layer[sample][b_indx] += b
    print(hidden_layer)



    ## applying relu 
    for sample in range(len(hidden_layer)):
        for neuron in range(len(hidden_layer[0])): 
            hidden_layer[sample][neuron] = relu(hidden_layer[sample][neuron])
    print(f"hidden_layer after applying an activation function to W * x + b: {hidden_layer}\n")



    ## multiplying hidden_layer * last_weights ## output should be: 3x1 or 1x3
    result_matrix_y_pred = []
    for hidden_layer_row in range(len(hidden_layer)):
        row_y_pred = []
        for last_weights_col in range(len(last_weights[0])): 

            dot_product = 0
            for hidden_layer_col in range(len(hidden_layer[0])):
                dot_product += hidden_layer[hidden_layer_row][hidden_layer_col] * last_weights[hidden_layer_col][last_weights_col]
            row_y_pred.append(dot_product)

        result_matrix_y_pred.append(row_y_pred)

    print(f"\n hidden layer neurons (3x4) * last_weights (4x1): {result_matrix_y_pred}\n")


    ## applying leaky relu
    for sample in range(len(result_matrix_y_pred)):
        for neuron in range(len(result_matrix_y_pred[0])): 
            result_matrix_y_pred[sample][neuron] = leaky_relu(result_matrix_y_pred[sample][neuron])
    print(f"result_matrix_y_pred after applying leaky_relu(x) to it: {result_matrix_y_pred}\n")



    ### calculating the loss (MSE)
    #loss_error = [(y_real_i-y_pred_i[0])**2 for y_real_i, y_pred_i in zip(y_real, result_matrix_y_pred) ]
    loss_error = MSE_costF(...)
    print(f"loss (average for 3 samples): {sum(loss_error)/len(loss_error)} \n")





    ### using derivaives, going backward
    dE_dloss = derivative_MSE_costF(result_matrix_y_pred)
    dloss_hidd2 = derivative_leaky_relu(result_matrix_y_pred)  ## derivative of leaky_relu w.r.t its input
    dhidd2_weights2 = ... #*derivative of 'multiplying hidden_layer * last_weights' w.r.t. its input
    ## d(wx+b)/dw + d(wx+b)/db
    ### x + 1
    #### hidden_layer_preRelu + 1  (or hidden_layer(post relu) + 1   ??)
    ##### answer: use post activ.function (that is, hidden_layer(post relu) (or just hidden_layer))
    ###### so we'll have:
    dhidd2_weights2 = hidden_layer + 1


    dhidd1postactf_dhidd1preactf = derivative_relu(hidden_layer) ## derivative of relu w.r.t its input
    dhidd1preactf_dweights1 = ... #*derivative of 'multiplying input*first_weights' w.r.t. its input
    ## d(wx+b)/dw + d(wx+b)/db
    ### x + 1
    #### inp + 1
    ##### so we'll have:
    dhidd1preactf_dweights1 = inp + 1


    dE_dweights1 = dE_dloss * dloss_hidd2 * dhidd2_weights2 * dhidd1postactf_dhidd1preactf * dhidd1preactf_dweights1
    dE_dweights2 = dE_dloss * dloss_hidd2 * dhidd2_weights2
    dE_biases = dE_dloss * dloss_hidd2

    ## updating the weights using an optimization algorithm like Adam or Stochastic Gradient Descent
    first_weights = first_weights - learning_rate * dE_dweights1
    biases = biases - learning_rate * dE_biases
    last_weights = first_weights - learning_rate * dE_dweights2



    








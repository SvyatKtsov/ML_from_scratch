
## without any libraries
# forward -> calculate the loss -> backward by the derivatives

from audioop import bias
import math 


def relu(x: float): return max([0, x])
def derivative_relu(x: float): return 1 if x>=0 else 0

def leaky_relu(x: float, neg_slope=0.3): return x if x>=0 else neg_slope*x
def derivative_leaky_relu(x: float, neg_slope= 0.3): return 1 if x>=0 else neg_slope

def sigmoid(x: float): return 1/ (1 + math.exp(-x))

def tanh(x: float): return ((math.exp(x)-math.exp(-x)) / (math.exp(x)+math.exp(-x)))


def MSE_costF(y_real, y_predicted):
    return (y_real-y_predicted)**2

def derivative_MSE_costF(y_real, y_predicted):
    return 2*(y_predicted-y_real)


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
    print(f"Epoch {epoch+1}... result: ")
    
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
    loss_error = [MSE_costF(y_real=y_real_i, y_predicted=y_pred_i[0]) for y_real_i, y_pred_i in 
                  zip(y_real, result_matrix_y_pred)]

    print(f"--------Epoch {epoch+1} loss (average for 3 samples): {sum(loss_error)/len(loss_error)}--------\n")





    ### using derivaives, going backward

    #dE_dloss = derivative_MSE_costF(result_matrix_y_pred)
    dE_dloss = [derivative_MSE_costF(y_real=y_real_i, y_predicted=y_pred_i[0]) for y_real_i, y_pred_i in 
                  zip(y_real, result_matrix_y_pred)]
    dE_dloss = [dE_dloss]
    assert len(dE_dloss)>0, 'len(lst) must be > 0'
    print(dE_dloss)
                  


    #dloss_dhidd2 = derivative_leaky_relu(result_matrix_y_pred)  ## derivative of leaky_relu w.r.t its input
    dloss_dhidd2 = [[0 for _ in range(len(result_matrix_y_pred[0]))] for _ in range(len(result_matrix_y_pred))]
    for sample in range(len(result_matrix_y_pred)): # hidden_layer
        for neuron in range(len(result_matrix_y_pred[0])): 
            dloss_dhidd2[sample][neuron] = derivative_leaky_relu(result_matrix_y_pred[sample][neuron])
    print(f"dloss_dhidd2: {dloss_dhidd2}")

    #dhidd2_weights2 = ... #*derivative of 'multiplying hidden_layer * last_weights' w.r.t. its input
    ## d(wx+b)/dw + d(wx+b)/db
    ### x + 1
    #### hidden_layer_preRelu + 1  (or hidden_layer(post relu) + 1   ??)
    ##### answer: use post activ.function (that is, hidden_layer(post relu) (or just hidden_layer))
    ###### so we'll have:
    #dhidd2_weights2 = hidden_layer + 1
    dhidd2_weights2 = [[0 for _ in range(len(hidden_layer[0]))] for _ in range(len(hidden_layer))]
    for sample in range(len(hidden_layer)): 
        for neuron in range(len(hidden_layer[0])): 
            dhidd2_weights2[sample][neuron] = hidden_layer[sample][neuron]+1
    #print(dhidd2_weights2)

    #dhidd1postactf_dhidd1= derivative_relu(hidden_layer) ## derivative of relu w.r.t its input
    dhidd1postactf_dhidd1 = [[0 for _ in range(len(hidden_layer[0]))] for _ in range(len(hidden_layer))]
    for sample in range(len(hidden_layer)): # hidden_layer
        for neuron in range(len(hidden_layer[0])): 
            dhidd1postactf_dhidd1[sample][neuron] = derivative_relu(hidden_layer[sample][neuron])


    dhidd1_dweights1 = ... #*derivative of 'multiplying input*first_weights' w.r.t. its input
    ## d(wx+b)/dw + d(wx+b)/db
    ### x + 1
    #### inp + 1
    ##### so we'll have:
    #dhidd1postactf_dhidd1actf_dweights1 = inp + 1
    dhidd1postactf_dhidd1actf_dweights1 = [[0 for _ in range(len(inp[0]))] for _ in range(len(inp))]
    for sample in range(len(inp)): # inp
        for neuron in range(len(inp[0])): 
            dhidd1postactf_dhidd1actf_dweights1[sample][neuron] = inp[sample][neuron]+1
    print(f"dhidd1postactf_dhidd1actf_dweights1: {dhidd1postactf_dhidd1actf_dweights1}")

    dhidd1_dbiases = 1 
    # d(wx+b)/d(b)==1 (because b is a constant (...number...))





    print(f"dE_dloss: {dE_dloss}")
    print(f"dloss_dhidd2: {dloss_dhidd2}")
    #dE_dweights2 = dE_dloss * dloss_dhidd2 * dhidd2_weights2
    by_dloss_dhidd2 = [] # by_dloss_dhidd2
    for i in range(len(dE_dloss)):
        row = []
        for j in range(len(dloss_dhidd2[0])):

            dot=0
            for k in range(len(dE_dloss[0])): # range(len(dE_dloss[0]))
                dot += dE_dloss[i][k]*dloss_dhidd2[k][j]
            row.append(dot)
        by_dloss_dhidd2.append(row)
    print(f"by_dloss_dhidd2:{by_dloss_dhidd2}")

    dE_dweights2 = []
    for i in range(len(dE_dloss)):
        row = []
        for j in range(len(dloss_dhidd2[0])):

            dot=0
            for k in range(len(dE_dloss[0])):
                dot += dE_dloss[i][k]*dloss_dhidd2[k][j]
            row.append(dot)
        dE_dweights2.append(row)



    #dE_dweights1 = dE_dloss * dloss_dhidd2 * dhidd2_weights2 * dhidd1postactf_dhidd1 * dhidd1_dweights1
    print(f"\ndE_dweights2:{dE_dweights2}")
    print(f"dhidd1postactf_dhidd1:{dhidd1postactf_dhidd1}")

    result_next = []
    for i in range(len(dE_dweights2)): 
     row = []
     for j in range(len(dhidd1postactf_dhidd1[0])):

        dot=0
        for k in range(len(dE_dloss[0])):
            dot += dE_dweights2[i][k]*dhidd1postactf_dhidd1[k][j]
        row.append(dot)
     result_next.append(row)       

    dE_dweights1 = []
    for i in range(len(result_next)):
        row = []
        for j in range(len(dhidd1_dweights1[0])):

            dot=0
            for k in range(len(result_next[0])):
                dot += result_next[i][k]*dhidd1_dweights1[k][j]
            row.append(dot)
        dE_dweights1.append(row)




    #dE_biases = dE_dloss * dloss_dhidd2 * dhidd2_weights2 * dhidd1postactf_dhidd1 * dhidd1_dbiases
    #dE_biases = []
    by_dhidd1 = []
    for i in range(len(dE_dweights2)):
        row = []
        for j in range(len(dhidd1postactf_dhidd1[0])):

            dot=0
            for k in range(len(dE_dweights2[0])):
                dot += dE_dweights2[i][k]*dhidd1postactf_dhidd1[k][j]
            row.append(dot)
        by_dhidd1.append(row)

    dE_biases = []
    for i in range(len(by_dhidd1)):
        row = []
        for j in range(len(dhidd1_dbiases[0])):

            dot=0
            for k in range(len(by_dhidd1[0])):
                dot += by_dhidd1[i][k]*dhidd1_dbiases[k][j]
            row.append(dot)
        dE_biases.append(row)



    ## updating the weights using an optimization algorithm like Adam or Stochastic Gradient Descent

    #first_weights = first_weights - learning_rate * dE_dweights1
    #first_weights = []
    for i in range(len(dE_dweights1)):
        for j in range(len(dE_dweights1[0])):
            first_weights[i][j] = first_weights[i][j] - learning_rate * dE_dweights1[i][j]
    

    #biases = biases - learning_rate * dE_biases
    #biases = []
    for i in range(len(dE_biases)):    
        biases[i] = biases[i] - learning_rate * dE_biases[i]


    #last_weights = last_weights - learning_rate * dE_dweights2
    #last_weights = []
    for i in range(len(dE_dweights2)):
        for j in range(len(dE_dweights2[0])):
            last_weights[i][j] = last_weights[i][j] - learning_rate * dE_dweights2[i][j]

    print('=============================================')
    




print("\n\n\n program exetuted \n")



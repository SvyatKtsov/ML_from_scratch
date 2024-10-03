
## without any libraries
# forward -> calculate the loss -> backward by the derivatives


def relu(x: float): 
    return max([0, x])

def leaky_relu(): pass
def sigmoid(): pass
def tanh(): pass
def cross_entropy(): pass

act_functions = [relu, leaky_relu, sigmoid, tanh, cross_entropy]

inp, y_real = [[-235, 7, 24, 14, 1, 9, 6.235], 
               [-240, 5.45, 16, 12.24, 0.25, 8.234, -12.364], 
               [-159.25, 10, 26.035, 17.2, 3, 13, 9.13606]], \
[5.5346, 27.235, 15.788]


## one hidden layer and for each 
weights = [ [0.124, 0.564, 0.8, -1.74], 
           [-0.124, 2, 1.2342, 0.131],
           [1.64, -0.9134, 0.53, 0.8],
           [0.7, 0.4, -1.53, 1.612],
           [-0.2332, 0.134, 0.778, 0.37],
           [0.3213, -0.82, 1.42, 0.23],
           [1.15, 0.664, -0.35, -0.12] ] # shape = inp_neurons_num * h_neurons_num => 7x3
biases = [0.2, 0.3, -0.65, 0.12] # 1 for each 'h_neurons'

print(len(inp), len(inp[0]), len(weights[0])) ## should be 3, 7, 4


## x*weights # 3x7 * 7x4 = 3x4
hidden_layer = []
for row_x in range(len(inp)): 
    row = []
    for col_w in range(len(weights[0])):
           
        dot = 0
        for col_x in range(len(inp[0])):
            dot += inp[row_x][col_x]*weights[col_x][col_w]
        row.append(dot)
    hidden_layer.append(row)
        
print(len(hidden_layer))
print(hidden_layer)


## hidden_layer_neurons + bias
### '...bias terms are additional constants attached to neurons and added to the weighted input before the activation function is applied'
for sample in range(len(hidden_layer)):
    for b_indx, b in enumerate(biases):
        hidden_layer[sample][b_indx] += b
print(hidden_layer)



## applying an activation function
for sample in range(len(hidden_layer)):
    for neuron in range(len(hidden_layer[0])): 
        hidden_layer[sample][neuron] = relu(hidden_layer[sample][neuron])
print(f"hidden_layer after applying an activation function to W * x + b: {hidden_layer}\n")


## multiplying hidden_layer * last_weights 
last_weights = ...
# hidden_layer






### calculating the loss (y_real - y_pred)**2



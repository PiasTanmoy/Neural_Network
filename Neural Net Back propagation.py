import random
import numpy as np
import pandas


'''Standard functions'''
def sigmoid(x):
    return (1.0 /  1.0 + np.exp(-x) )

def sigmoid_prime(x):
    return ( sigmoid(x) * (1-sigmoid(x)))


'''
This is a neural network class which uses 
back propagation
to train it's weights and biases
'''

class NeuralNetwork(object):
    
    '''
    constructor
    the neural net structure is
    fed in to the network
    '''
    def __init__(self, neural_net):
        '''Instance variables'''
        self.num_layers = len(neural_net)
        self.sizes = neural_net
        
        '''
        randomly generated biases
        range between 0 to 1
        (y, 1) is the dimension 
        actually 1D array 
        size = number of neurons at that stage
        '''
        self.biases = []
        for y in neural_net[1:]:
            temp = np.random.uniform(0, 1, (y, 1))
            self.biases.append(temp)
            
        
        '''
        weight matrix
        3 5
        row = 5
        col = 3
        '''
        self.weights = []
        for x, y in zip(neural_net[:-1], neural_net[1:]):
            temp = np.random.uniform(0, 1, (y, x))
            self.weights.append(temp)
  


            
    def train_neural_net(self, train_set, learning_rate):
        
        #print(train_set[0])
        
        for x, y in train_set:
            
            backprop_b, backprop_w = self.back_propagation(x, y)
            #print(delta_nabla_w)
            #print("KKKKKKKKKKKK")
            #print(self.weights)
            #self.weights = [(w - learning_rate*nw) for w, nw in zip(self.weights, backprop_w) ]
            w_temp = []
            for w, nw in zip(self.weights, backprop_w):
                temp = (w - learning_rate*nw)
                w_temp.append(temp)
            self.weights = w_temp
            
            
            
            #self.biases = [(b - learning_rate*nb) for b, nb in zip(self.biases, backprop_b)]
            b_temp = []
            for b, nb in zip(self.biases, backprop_b):
                temp = (b - learning_rate*nb)
                b_temp.append(temp)
            self.biases = b_temp
    
   
    
    
    def back_propagation(self, x, y):
        
        nabla_b = []
        for b in self.biases:
            temp = np.zeros(b.shape)
            nabla_b.append(temp)
                   
            
        #nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_w = []
        for w in self.weights:
            temp = np.zeros(w.shape)
            nabla_w.append(temp)
        #print("nabla_w")
        #print(nabla_w)
        
        a = x
        a_list = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            a_list.append(a)
        
        temp_sig = sigmoid_prime(zs[-1])
        temp_cost = self.cost_derivative(a_list[-1], y)
        delta = temp_cost*temp_sig
        #delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        #print(activations[-2])
        nabla_w[-1] = np.dot(delta, np.transpose(a_list[-2]))
        
        
        for l in range(2, self.num_layers):
            
            z = zs[-l]
            #print(z)
            sp = sigmoid_prime(z)
            #print(sp)
            wt = np.dot(np.transpose(self.weights[-l+1]), delta)
            #print(wt)
            delta = wt*sp
            #print(delta)
            nabla_b[-l] = delta
            #print(nabla_b[-l])
            aT = np.transpose(a_list[-l-1])
            #print(aT)
            nabla_w[-l] = np.dot(delta, aT)
            #print(nabla_w[-l])
            
        return (nabla_b, nabla_w)
    
   
    
    def feedforward(self, activation):
        for bias, wt in zip(self.biases, self.weights):
            wa = np.dot(wt, activation)
            activation = sigmoid(wa + bias)
        return activation
    
    
    def backward_propagate_error(network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
                
                
    def update_weights(network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']
    
  
    
    def evaluate2(self, test_data):
        
        test_results = []
        for (x, y) in test_data:
            temp = self.feedforward(x)
            category = np.argmax(temp) + 1
            arr = [category, y]
            #print(arr)
            test_results.append(arr)
            
        count = 0
        
        for y_bar, y in test_results:
            if(y_bar == y):
                count+=1
        return count
            
         
        
    def cost_derivative(self, y_bar, y):
        return (y_bar - y)




net = NeuralNetwork([3, 2, 5, 10, 2])
train_set = np.array([ [ [ [0.3], [0.8], [0.5] ], [ [0], [1] ] ] ])
train_set2 = np.array([ [ [ [0.3], [0.8], [0.5] ],  1  ], [ [ [0.3], [0.8], [0.5] ],  1  ],[ [ [0.3], [0.8], [0.5] ],  1  ] ])
net.train_neural_net(train_set, 0.5)
net.evaluate2(train_set2)



data = np.loadtxt("patternData.txt")
final_dataset = []
for x in data:
    y = x[-1]
    #print(y)
    
    a = x[:-1]
    a = np.vstack(a)
    #print(a)
    
    z = np.array([0]*3)
    z[int(y)-1] = 1
    z = np.vstack(z)
    #print(z)
    
    final = np.array([a, z])
    
    #print(final)
    #print(final.shape)
    
    final_dataset.append(final)
    
    
    
    
#print(final_dataset)  



data_test = np.loadtxt("patternData.txt")
test_dataset = []
for x in data_test:
    y = x[-1]
    #print(y)
    
    a = x[:-1]
    a = np.vstack(a)
    #print(a)
    
    z = np.array([0]*3)
    z[int(y)-1] = 1
    z = np.vstack(z)
    #print(z)
    
    final = np.array([a, y])
    
    #print(final)
    #print(final.shape)
    
    test_dataset.append(final)
#print(test_dataset) 



net = NeuralNetwork([7,4, 5, 6, 12, 6, 3])

final_data = np.array(final_dataset)

net.train_neural_net(final_dataset, 0.2)
net.evaluate2(test_dataset)        
                    
       
    

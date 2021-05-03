import numpy as np
import matplotlib.pyplot as plt

class Model:
   def __init__(self):
        print("self")
        self.cost = Cost()
        self.act = Activation()
        self.model=[]

   def initialize_parameters_deep(self,layer_dims):
        np.random.seed(3)
        self.model=layer_dims
        parameters = {}
        L = len(layer_dims)            # number of layers in the network
    
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        return parameters
    
   def linear_forward(self,A, W, b):
    
        Z = np.dot(W, A) + b  
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache
   def linear_activation_forward(self,A_prev, W, b, activation):

        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.act.sigmoid(Z)
        
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.act.relu(Z)
        
        elif activation == "linear":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.act.linear(Z)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
    
        return A, cache

   def linear_backward(self,dZ, linear_cache):
       A_prev, W, b = linear_cache 
       m = A_prev.shape[1]

       dW = (1/m)*np.dot(dZ,A_prev.T)
       db = (1/m)*np.sum(dZ, axis = 1,keepdims= True)

       dA_prev = np.dot(W.T,dZ)

       return dA_prev,dW,db

   def linear_activation_backward(self,dA, cache, activation):
       linear_cache, activation_cache = cache
       if activation == "relu":
           dZ = self.act.relu_backward(dA, activation_cache)
            
       elif activation == "sigmoid":
          dZ = self.act.sigmoid_backward(dA, activation_cache)
          
           
       dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
       return dA_prev, dW, db

   def update_parameters(self,parameters, grads, learning_rate):
      L = len(parameters) // 2 # number of layers in the neural network
      for l in range(L):
         parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
         parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
      return parameters

   

   
class Cost:
   def __init__(self,loss="Cross"):
        print("Cost")
        self.loss=loss
        
   def compute_cost(self,AL, Y):
       m = Y.shape[1]
       cost=0
       if self.loss=="Cross":
           cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
       cost = np.squeeze(cost)     
       assert(cost.shape == ())
       return cost

   def derive_cost(self,AL, Y):
       m = Y.shape[1]
       dAL=0
       if self.loss=="Cross":
           dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
       return dAL
   
    
class Activation:
   def __init__(self,act="relu"):
        print("Activation")
        self.act=act

   def relu(self,input1):
       ANS = np.zeros(input1.shape)
       for i in range(0,ANS.shape[0]):
           for j in range(0,ANS.shape[1]):
               ANS[i,j] = max(0, input1[i,j])
        #return ANS , input1 
       cache = input1
       return ANS , cache 

   def sigmoid(self,Z) :
       A = 1/(1 + np.exp(-1 * Z))
       cache = Z
       return A,cache
   def linear(self,Z) :
       return Z,Z
   
   def relu_backward(self,dA, Z) :

       grad_relu = np.zeros(Z.shape)
       for i in range(Z.shape[0]):
           for j in range(Z.shape[1]):
                if Z[i,j] >= 0 :
                    grad_relu[i,j] = 1
                else:
                    grad_relu[i,j] = 0
       return  dA * grad_relu 

   def sigmoid_backward(self,dA, Z):
       gard_sigmoid = np.zeros(Z.shape)
       out,cache = self.sigmoid(Z) 
       grad_sigmoid = out*(1-out)
       return dA * grad_sigmoid


class Classifier(Model):
    def __init__(self):
        print("Classifier")
        self.cost = Cost()
        self.act = Activation()
        self.para=[]
        self.Loss=[]
        Model.__init__(self)
    def L_model_forward(self,X, parameters):

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = Model.linear_activation_forward(self,A_prev, 
                                                 parameters['W' + str(l)], 
                                                 parameters['b' + str(l)], 
                                                 activation='relu')
            caches.append(cache)
            
        AL, cache = Model.linear_activation_forward(self,A, 
                                              parameters['W' + str(L)], 
                                              parameters['b' + str(L)], 
                                              activation='sigmoid')
        caches.append(cache)
                
        assert(AL.shape == (1, X.shape[1]))
        
        return AL,caches
    def L_model_backward(self,AL, Y, caches):
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        
        dAL = self.cost.derive_cost(AL,Y)
        
        
        
        current_cache = caches[-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = Model.linear_activation_backward(self,dAL,current_cache,"sigmoid")
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = Model.linear_activation_backward(self,dAL,current_cache,"relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        
        return grads

    def Train(self,X, Y, layers_dims, learning_rate=0.075, num_iterations=500, print_cost=False):
       np.random.seed(1)
       costs = []
       param=[]
       parameters = Model.initialize_parameters_deep(self,layers_dims)
       #print(parameters,X)
       for i in range(0, num_iterations):
           AL, caches = self.L_model_forward(X, parameters)
           cost = self.cost.compute_cost(AL, Y)
           
           grads = self.L_model_backward(AL, Y, caches)
           parameters = self.update_parameters(parameters, grads, learning_rate)
           self.Loss.append(cost)
           self.para.append(parameters)
           param.append(parameters)
           if print_cost and i % 100 == 0:
               print ("Cost after iteration %i: %f" % (i, cost))
           if print_cost and i % 100 == 0:
               costs.append(cost)
            
    # plot the cost
       plt.plot(np.squeeze(self.Loss))
       plt.ylabel('cost')
       plt.xlabel('iterations (per tens)')
       plt.title("Learning rate =" + str(learning_rate))
       plt.show()

       return parameters

def predict(X, parameters,model):

    AL,caches = model.L_model_forward(X, parameters)
    print(AL)
    predictions = AL > 0.5

    return predictions
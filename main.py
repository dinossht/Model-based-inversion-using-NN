import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
# based on https://gist.github.com/maunashjani/922e5a2a60130367dc58f1ee1fd6da36
# and https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

in_len = 100
X = np.random.randn(1,in_len) 

# NOTE: input pulse could be a variable as well in training with gprMax


# NOTE: Needs to be normalized to work, WHY?
# NOTE: Does not work with negative values?
delays=np.random.uniform(low=0.1, high=0.9, size=(1,100))#[0.1, 0.67, 0.2, 0.5, 0.3]
scales=np.random.uniform(low=0.1, high=0.9, size=(1,100))#[0.1, 0.0, 0.85,0.9, 0.4]
y=np.array([delays, scales])  
y=y.reshape((1,2*delays.shape[1]))
out_size = y.shape[1]

def feedforward_modelbased(delay):
    temp=np.zeros((1,in_len))
    temp[0,0] = 1
    out = np.zeros((1,in_len))

    LEN = int(delay.shape[1]/2)

    for i in range(LEN):
        scale = delay[0][LEN + i]
        delay_scaled =  scale * np.roll(temp, int((in_len*delay).round()[0][i])) 
        idx = delay_scaled != 0
        out[idx] = delay_scaled[idx]

    return out

print(f"y:{y}")
plt.plot(feedforward_modelbased(y)[0],'b')
plt.show()

# Define useful functions    

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t

def tanh_derivative(x):
    return 1-tanh(x)**2

def LeakyReLU(x):
    return np.where(x > 0, x, x * 0.01) 

def LeakyReLU_derivative(x):
    dx = np.ones_like(x)
    dx[x < 0] = 1.0
    return dx 


# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],10) # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(10,out_size)
        self.y = y
        self.output = np.zeros(y.shape)
        self.lr = 1e-0
        
    def feedforward(self, X):
        self.input = X
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += self.lr * d_weights1
        self.weights2 += self.lr * d_weights2

    def train(self, X, y):
        self.output = self.feedforward(X)
        self.backprop()
        

NN = NeuralNetwork(X,y)
loss_arr = []
loop_N = 50
eps = 1e-5
for i in range(loop_N): # trains the NN 1,000 times
    #if i % 100 ==0: 
    if loop_N <= 100:
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward(X)))
    loss = str(np.mean(np.square(y - NN.feedforward(X))))
    loss_arr.append(float(loss))
    print ("Loss: \n" + loss) # mean sum squared loss
    print ("\n")
    
    X = feedforward_modelbased(NN.feedforward(X))
    NN.train(X, y)

    if loop_N <= 1000:
        plt.clf()
        plt.subplot(211)
        plt.xlim([0, loop_N])
        plt.plot(loss_arr)

        plt.subplot(212)
        plt.plot(feedforward_modelbased(y)[0],'b')
        plt.plot(X[0],'r--')
        plt.ylim([0,1.0])
        plt.legend(['label','pred'])
        plt.title(f"Iteration: {i}")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    if np.mean(np.square(X[0] - feedforward_modelbased(y)[0])) <= eps:
        print(f"Early stopping after itr: {i}")
        break


plt.plot(feedforward_modelbased(y)[0],'b')
plt.plot(X[0],'r--')
plt.legend(['label','pred'])
plt.show()

plt.plot(loss_arr)
plt.show()


"""
t = np.linspace(-1, 1, 2 * 100, endpoint=False)
i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)
plt.plot(t, i, t, q, t, e, '--')
plt.show()
"""
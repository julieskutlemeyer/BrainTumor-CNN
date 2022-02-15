# from numpy import exp, array, random, dot

# class NeuralNetwork():
#     def __init__(self):
#         random.seed(1)

#         self.weights=2*random.random((3,1)) - 1 #3 rader, 1 kolonne med randomme tall fra -1,1. 

#     def __sigmoid(self, x): #gjør om inputen til sigmoid (inputen er alle inputen ganget med vekten, også de plusset sammen), altså et tall mellom 0,1
#         return 1/(1+exp(-x))

#     def __sigmoid_derivative(self, x): #deriverer sigmoiden. vil da bare ha verdier når sigmoiden er nærme 0
#         return x*(1-x)

#     def think(self, inputs): 
#         return self.__sigmoid(dot(inputs, self.weights)) #blir en array med genererte outputs

#     def train(self, inputs, outputs, training_iterations):
#         for iteration in range(training_iterations):
#             output_star = self.think(inputs) #x=dot mellom vekten og alle inputene. s(x)
#             error = outputs - output_star #liste med error

#             adjust = dot(inputs, error*self.__sigmoid_derivative(output_star)) #verdier nærme 0.
#             self.weights += adjust

# neural_network = NeuralNetwork()
# inputs=array([[0,0,1] [1,1,1] [1,0,1] [0,1,1] ])
# outputs=array([[0, 1, 1, 0]])

# neural_network.train(inputs, outputs, 1000)

# # denne blir da såppas god at den skal klare å kjenne igjen lignende inputs og få de til å få lignende outputs


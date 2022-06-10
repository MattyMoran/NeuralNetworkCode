import numpy as np


class MLP:
    " Multi-layer perceptron " 
    def __init__(self, sizes, beta=20, momentum=0.7):

        """
        sizes is a list of length four. The first element is the number of features 
                in each samples. In the MNIST dataset, this is 784 (28*28). The second 
                and the third  elements are the number of neurons in the first 
                and the second hidden layers, respectively. The fourth element is the 
                number of neurons in the output layer which is determined by the number 
                of classes. For example, if the sizes list is [784, 5, 7, 10], this means 
                the first hidden layer has 5 neurons and the second layer has 7 neurons. 
        
        beta is a scalar used in the sigmoid function
        momentum is a scalar used for the gradient descent with momentum 
        """
        self.beta = beta
        self.momentum = momentum

        self.nin = sizes[0] # number of features in each sample
        self.nhidden1 = sizes[1] # number of neurons in the first hidden layer 
        self.nhidden2 = sizes[2] # number of neurons in the second hidden layer 
        self.nout = sizes[3] # number of classes / the number of neurons in the output layer
        
        self.hidden1 = np.zeros((9000, 8))
        self.hidden2 = np.zeros((9000, 8))

        # Initialise the network of two hidden layers 
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden1)-0.5)*2/np.sqrt(self.nin) # hidden layer 1 
        self.weights2 = (np.random.rand(self.nhidden1+1,self.nhidden2)-0.5)*2/np.sqrt(self.nhidden1) # hidden layer 2
        self.weights3 = (np.random.rand(self.nhidden2+1,self.nout)-0.5)*2/np.sqrt(self.nhidden2) # output layer


    def train(self, inputs, targets, eta, niterations):
        """
        inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.

        targets is a numpy array of shape (num_train, D) containing the training labels
                    consisting of num_train samples each of dimension D.

        eta is the learning rate for optimization 
        niterations is the number of iterations for updating the weights 

        """
        ndata = np.shape(inputs)[0] # number of data samples 
        # adding the bias
        inputs = np.concatenate((inputs,-np.ones((ndata,1))),axis=1)

        # numpy array to store the update weights 
        updatew1 = np.zeros((np.shape(self.weights1))) 
        updatew2 = np.zeros((np.shape(self.weights2)))
        updatew3 = np.zeros((np.shape(self.weights3)))


        for n in range(niterations):

            #############################################################################
            # TODO: implement the training phase of one iteration which consists of two phases:
            # the forward phase and the backward phase. you will implement the forward phase in 
            # the self.forwardPass method and return the outputs to self.outputs. Then compute 
            # the error (hints: similar to what we did in the lab). Next is to implement the 
            # backward phase where you will compute the derivative of the layers and update 
            # their weights. 
            #############################################################################
            
            # forward phase 
            self.outputs = self.forwardPass(inputs)
            #print(self.outputs.shape)
            #print(self.outputs.sum())

            # Error using the sum-of-squares error function
            error = 0.5*np.sum((self.outputs-targets)**2)

            if (np.mod(n,100)==0):
                print("Iteration: ",n, " Error: ",error)

            # backward phase 
            # Compute the derivative of the output layer. NOTE: you will need to compute the derivative of 
            # the softmax function. Hints: equation 4.55 in the book. 
            deltao = ((self.outputs - targets)*self.outputs*(1.0 - self.outputs))/(self.outputs.shape[0])
            #print(deltao.shape)
            
            #print(self.weights3.shape)
            #print(self.weights2.shape)
            #print(self.weights1.shape)
            
            #print(self.hidden2.shape)
            #print(self.hidden1.shape)
            #print("Deltah2")
            #print(deltao.shape)
            #print(self.weights3.shape)
            
            #print("Deltah1")
            #print(deltao.shape)
            #print(self.weights2.shape)
            
            # compute the derivative of the second hidden layer 
            #deltah2 = None 
            deltah2 = self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao,np.transpose(self.weights3)))
            #detlah2 = deltah2[:,:-1]
            #print(deltah2.shape)
            
            # compute the derivative of the first hidden layer 
            #deltah1 = None
            deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2[:,:-1],np.transpose(self.weights2)))
            #detlah1 = deltah1[:,:-1]
            #print(deltah1.shape)
            #print(deltah1.shape)

            # update the weights of the three layers: self.weights1, self.weights2 and self.weights3
            # here you can update the weights as we did in the week 4 lab (using gradient descent) 
            # but you can also add the momentum 
            
            updatew1 = (eta*(np.dot(np.transpose(inputs),deltah1[:,:-1])))+(self.momentum*updatew1)
            updatew2 = eta*(np.dot(np.transpose(self.hidden1),deltah2[:,:-1]))+(self.momentum*updatew2)
            #print(updatew2.shape)
            updatew3 = eta*(np.dot(np.transpose(self.hidden2),deltao))+(self.momentum*updatew3)
            
            #print(inputs.shape)
            #print(deltah1.shape)
            #print(updatew1.shape)
            #print(self.hidden1.shape)
            #print(deltah2[:,:-1].shape)
            #print(updatew2.shape)
            #print(self.hidden2.shape)
            #print(deltao.shape)
            #print(updatew3.shape)
            
            #print(updatew1.shape)
            #print(updatew2.shape)
            #print(updatew3.shape)
            
            self.weights1 -= updatew1
            self.weights2 -= updatew2
            self.weights3 -= updatew3
            #############################################################################
            # END of YOUR CODE 
            #############################################################################




    def forwardPass(self, inputs):
        """
            inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.  
        """
        #############################################################################
        # TODO: Implement the forward phase of the model. It has two hidden layers 
        # and the output layer. The activation function of the two hidden layers is 
        # sigmoid function. The output layer activation function is the softmax function
        # because we are working with multi-class classification. 
        #############################################################################
        
        # layer 1 
        # compute the forward pass on the first hidden layer with the sigmoid function
           
        #print("Inputs = ", inputs.shape)
        self.hidden1 = np.dot(inputs,self.weights1) 
        self.hidden1 = 1.0/(1.0+np.exp(-self.beta*self.hidden1)) 
        self.hidden1 = np.concatenate((self.hidden1,-np.ones((np.shape(inputs)[0],1))),axis=1) 
        #self.hidden1 = None 
        #print("Hidden1 = ", self.hidden1.shape)

        # layer 2
        # compute the forward pass on the second hidden layer with the sigmoid function
        
        self.hidden2 = np.dot(self.hidden1,self.weights2) 
        #print("Hidden2 pre = ", hidden2.shape)
        self.hidden2 = 1.0/(1.0+np.exp(-self.beta*self.hidden2)) 
        self.hidden2 = np.concatenate((self.hidden2,-np.ones((np.shape(inputs)[0],1))),axis=1) 
        #self.hidden2 = None
        #print("Hidden2 post = ", self.hidden2.shape)


        # output layer 
        # compute the forward pass on the output layer with softmax function
        output = np.dot(self.hidden2,self.weights3) 
        safetyVar = []
        for y in range(output.shape[0]):
                #output[y,:]
                
                safetyVar = output[y] - np.max(output[y])
                output[y] = np.exp(safetyVar)/np.sum(np.exp(safetyVar))                            
        #safetyVar = output - np.max(output)
        #outputs = np.exp(safetyVar)/np.sum(np.exp(safetyVar))
        #outputs = None 
        #print(np.sum(output[0]))

        #############################################################################
        # END of YOUR CODE 
        #############################################################################
        return output


    def evaluate(self, X, y):
        """ 
            this method is to evaluate our model on unseen samples 
            it computes the confusion matrix and the accuracy 
    
            X is a numpy array of shape (num_train, D) containing the testing images
                    consisting of num_train samples each of dimension D. 
            y is  a numpy array of shape (num_train, D) containing the testing labels
                    consisting of num_train samples each of dimension D.
        """

        inputs = np.concatenate((X,-np.ones((np.shape(X)[0],1))),axis=1)
        outputs = self.forwardPass(inputs)
        nclasses = np.shape(y)[1]

        # 1-of-N encoding
        outputs = np.argmax(outputs,1)
        targets = np.argmax(y,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print("The confusion matrix is:")
        print(cm)
        print("The accuracy is ",np.trace(cm)/np.sum(cm)*100)
 

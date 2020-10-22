import numpy as np
import ipdb


class nn_model():
    def __init__(self, args, train_shape, init_weights = False, alpha_1=None, alpha_2=None, beta_1=None, beta_2=None):
        """
        @brief:= initialize the model
        @in:= args: the cmd args
              train_shape : the shape of the training data excluding label
              init_weights : whether initial weights are given or not
              alpha_i, beta_i : initial weights
        """

        self.epochs = args.num_epochs
        self.eta = args.step_size
        self.in_dim = train_shape[1]
        self.hidden_dim = args.hidden_dim
        self.out_dim = 10 #10 classes
        self.seed = args.seed
        np.random.seed(self.seed)
        self.eps = args.eps

        hidden_dim = args.hidden_dim

        #ipdb.set_trace()

        if init_weights:
            if self.hidden_dim < alpha_1.shape[0] :
                self.alpha = np.hstack( ( (beta_1.reshape(beta_1.shape[0],1))[:hidden_dim], alpha_1[:hidden_dim]) )
                self.beta = np.hstack( (beta_2.reshape(beta_2.shape[0],1), alpha_2[:,:hidden_dim]) )
            else:
                self.alpha = np.vstack( (np.hstack( (beta_1.reshape(beta_1.shape[0],1), alpha_1) ), np.ones( (self.hidden_dim-256,train_shape[1]+1) )) )
                self.beta = np.hstack( (beta_2.reshape(beta_2.shape[0],1), alpha_2, np.ones((self.out_dim, self.hidden_dim-256))) )
        else:
            self.alpha = np.random.random((args.hidden_dim, train_shape[1]+1))
            self.beta = np.random.random((self.out_dim, args.hidden_dim+1))
            #self.alpha = np.zeros((args.hidden_dim, train_shape[1]+1))
            #self.beta = np.zeros((self.out_dim, args.hidden_dim+1))


    def linear_forward(self, train_data, batch=False):
        """
        @brief:= forward pass through linear layer
        @in:= train_data : the input features
              batch : whether to evaluate in batch or not
        
        """

        if not batch:
            return np.matmul(self.alpha, (np.hstack( ([1], train_data))).T)

        return np.matmul(self.alpha, (np.hstack( (np.ones((train_data.shape[0],1)), train_data))).T)

    def linear_backward(self, hidden_data, softmax_back, sigmoid_back):
        """
        @brief:= backward pass for linear
        """

        temp = (softmax_back.reshape(self.hidden_dim))*sigmoid_back

        dalphs_ = np.matmul(temp.reshape(self.hidden_dim,1), hidden_data.reshape(1,self.in_dim))        

        return np.hstack( (temp.reshape(self.hidden_dim,1), dalphs_) )

    def sigmoid_forward(self, hidden_data):
        """
        @brief:= forward pass through sigmoid
        """

        return 1/(1+np.exp(-1*hidden_data))

    def sigmoid_backward(self, hidden_out):
        """
        @brief:= backward pass through sigmoid
        """

        return hidden_out*(1-hidden_out)
        

    def softmax_xeloss_forward(self, hidden_out, labels, batch=False):
        """
        @brief:= forward pass through softmax and cross_entropy
        @in:= hidden_out : the output of the hidden layer
              labels : the true label of the example
              batch: to pass as batch or not
        @out:= loss : the total loss of the sample
               last_out : the output of softmax
        """

        if not batch:
            in_softmax = np.matmul(self.beta, (np.hstack( (np.ones((1)), hidden_out)) ))
            exps = np.exp(in_softmax - np.max(in_softmax))
            out_softmax = (exps) /  (np.sum(exps))
            
            return (-1 * np.log(out_softmax[int(labels)]+self.eps), out_softmax)


        in_softmax = np.matmul(self.beta, (np.vstack( (np.ones((1,hidden_out.shape[1])), hidden_out)) ))
        exps = np.exp(in_softmax - np.max(in_softmax))
        out_softmax = (exps) /  (np.sum(exps, axis=0))

        loss = 0

        for i in range(labels.size):
            loss += np.log(out_softmax[int(labels[i]),i] + self.eps)

        return (loss / (-1 * labels.size), out_softmax)

    def softmax_xeloss_backward(self, out_softmax, hidden_out, label):
        """
        @brief:= backward pass through softmax and cross_entropy
        @in:= hidden_out : the output of the hidden layer
              out_softmax : the output of the softmax layer
        @out:= dbeta : the derivatives wrt beta
               d_hidden : the derivates wrt hidden layer
        """
       
        back_grad = out_softmax

        back_grad[int(label)] -= 1

        temp = np.matmul(back_grad.reshape(self.out_dim,1), hidden_out.reshape(1,self.hidden_dim))

        dbeta = np.hstack((back_grad.reshape(self.out_dim,1), temp))

        return (dbeta, np.matmul((self.beta.T)[1:,:], back_grad.reshape(self.out_dim,1)))


    def train(self, train_data, train_labels, test_data, test_labels):
        print("-----------------------------------------")
        print("Start Train")
        print("-----------------------------------------")

        #ipdb.set_trace()

        train_losses = []
        test_losses = []
        test_accs = []

        
        for epoch in range(self.epochs):
            
            
            print("Epoch : {}/{}".format(epoch+1, self.epochs))

            loss_train_batch=0

            for (i,train_point) in enumerate(train_data):
                #ipdb.set_trace()

                hidden_data = self.linear_forward(train_data[i])
                hidden_out = self.sigmoid_forward(hidden_data)
                loss, out_softmax = self.softmax_xeloss_forward(hidden_out, train_labels[i])
               
                dbeta, softmax_back = self.softmax_xeloss_backward(out_softmax, hidden_out, train_labels[i])
                sigmoid_back = self.sigmoid_backward(hidden_out)
                dalpha = self.linear_backward(train_data[i], softmax_back, sigmoid_back)

                
                self.alpha -= (self.eta * dalpha)
                self.beta -= (self.eta * dbeta)
                loss_train_batch += loss

            print("Avg Train Loss for this epoch : {}".format(loss_train_batch/train_labels.size))

            test_loss, test_acc = self.test(test_data, test_labels)

            train_losses.append(loss_train_batch/train_labels.size)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            print("Avg Test Loss for this epoch : {}".format(test_loss))
            print("Test Acc for this epoch : {}".format(test_acc))
            print("-----------------------------------------")


        print("-----------------------------------------")
        print("End Train")
        print("-----------------------------------------")

        return train_losses, test_losses, test_accs

    def test(self, test_data, test_labels):
        """
        @breif:= return the loss of the given test samples
        @in:= test_data : the test data features
              test_labels : the true labels of test data
        @out:= loss_batch : the avg loss of the entire batch
               acc : the acc on the entire batch
        """
        
        hidden_data_batch = self.linear_forward(test_data, batch=True)
        hidden_out_batch = self.sigmoid_forward(hidden_data_batch)
        loss_batch, out_softmax = self.softmax_xeloss_forward(hidden_out_batch, test_labels, batch=True)

        correct = np.argmax(out_softmax, axis=0) == test_labels

        return loss_batch, (sum(correct)/test_labels.size)

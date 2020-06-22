import numpy as np
import matplotlib.pyplot as plt


def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)
    return p / np.sum(p, axis=ax, keepdims=True)


def cos(x):
    return np.cos(x)


def softplus(x):
    return np.logaddexp(0.0, x)


def tanh(x):
    return np.tanh(x)


def derivative_tanh(x):
    tan = np.square(tanh(x))
    return 1-tan


def derivative_softplus(x):
    np.seterr(over='ignore', under='ignore')
    return 1 / (1 + np.exp(-x))


# using taylor series for one more stable implement and difference between value of derivative_softplus and value of
# taylor series is 1e^-7
def derivative_softplus_approximation(x):
    return (1 / 2) + (x / 4) - (x**3 / 48) + (x**5 / 480)-(17*x**7/80640)+(31*x**9/1451520)


class NeuralNetwork:

    def __init__(self, x_train, y_train, hidden_neurons, learning_rate, lamda, activation_function, batch_size, epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_dim = x_train.shape[1]
        self.output_dim = y_train.shape[1]
        self.a = None
        self.s = None
        self.z = None
        # initialize weights
        self.w1 = np.random.randn(self.hidden_neurons, self.input_dim)  # W1->weights input layer-hidden layer
        self.w2 = np.random.randn(self.output_dim, self.hidden_neurons + 1)  # W2->weights hidden layer-output layer

    def xavier_weights(self):
        #use seed for the examples of model
        #np.random.seed(0)
        # W1->weights input layer-hidden layer
        self.w1 = np.random.randn(self.hidden_neurons, self.input_dim) * np.sqrt(6/(self.hidden_neurons+self.input_dim))
        # W2->weights hidden layer-output layer
        hd = self.hidden_neurons+1
        self.w2 = np.random.randn(self.output_dim, self.hidden_neurons + 1) * np.sqrt(6/(self.output_dim+hd))

    def cost_function(self, y_train):
        max_error = np.max(self.s, axis=1)
        # Compute the cost function to check convergence
        # Using the logsumexp trick for numerical stability - lec8.pdf slide 43
        ew = np.sum(y_train * self.s) - np.sum(max_error) - \
             np.sum(np.log(np.sum(np.exp(self.s - np.array([max_error, ] * self.s.shape[1]).transpose()), 1))) - \
             (0.5 * self.lamda) * (np.sum(np.square(self.w1))+np.sum(np.square(self.w2)))
        return ew

    def gradient_of_cost_w2(self, y_train):
        y = softmax(self.s)
        # calculate gradient
        gradEw = ((y_train - y).transpose()).dot(self.z) - self.lamda * self.w2
        return gradEw


    def gradient_of_cost_w1(self, x_train, y_train):
        y = softmax(self.s)
        # calculate gradient
        d2 = (y_train - y).dot(self.w2[:, 1:])
        if self.activation_function == 'softplus':
            derivative = derivative_softplus(self.a)
        elif self.activation_function == 'tanh':
            derivative = derivative_tanh(self.a)
        else:
            derivative = -(np.sin(self.a))
        temp = d2*derivative
        gradEw2 = np.dot(temp.transpose(), x_train) - self.lamda * self.w1
        return gradEw2

    def feedforward(self, x_train, y_train):
        self.a = x_train.dot(self.w1.transpose())
        if self.activation_function == 'softplus':
            self.z = softplus(self.a)
        elif self.activation_function == 'tanh':
            self.z = tanh(self.a)
        else:
            self.z = cos(self.a)
        self.z = np.hstack((np.ones((self.z.shape[0], 1)), self.z))
        self.s = self.z.dot(self.w2.transpose())
        ew = self.cost_function(y_train)
        return ew

    def back_propagation(self, x_train, y_train):
        gradEw2 = self.gradient_of_cost_w2(y_train)
        gradEw1 = self.gradient_of_cost_w1(x_train, y_train)
        new_w2 = self.w2 + self.learning_rate*gradEw2
        new_w1 = self.w1 + self.learning_rate*gradEw1
        return new_w2, new_w1

    def set_weights(self, w1, w2):
        self.w1 = w1
        self.w2 = w2

    def gradient_check_w2(self, epsilon, x_sample, y_sample):
        gradEw2 = self.gradient_of_cost_w2(y_sample)
        print("gradEw2 shape: ", gradEw2.shape)
        numericalGrad = np.zeros(gradEw2.shape)
        # Compute all numerical gradient estimates and store them in
        # the matrix numericalGrad
        old_w2 = np.copy(self.w2)
        for k in range(numericalGrad.shape[0]):
            for d in range(numericalGrad.shape[1]):
                # add epsilon to the w[k,d]
                self.w2 = np.copy(old_w2)
                self.w2[k, d] += epsilon
                e_plus = self.feedforward(x_sample, y_sample)

                # subtract epsilon to the w[k,d]
                self.w2 = np.copy(old_w2)
                self.w2[k, d] -= epsilon
                e_minus = self.feedforward(x_sample, y_sample)
                # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
                numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)
        return gradEw2, numericalGrad

    def gradient_check_w1(self, epsilon, x_sample, y_sample):
        gradEw1 = self.gradient_of_cost_w1(x_sample, y_sample)
        print("gradEw1 shape: ", gradEw1.shape)
        numericalGrad = np.zeros(gradEw1.shape)
        # Compute all numerical gradient estimates and store them in
        # the matrix numericalGrad
        old_w1 = np.copy(self.w1)
        for k in range(numericalGrad.shape[0]):
            for d in range(numericalGrad.shape[1]):
                # add epsilon to the w[k,d]
                self.w1 = np.copy(old_w1)
                self.w1[k, d] += epsilon
                e_plus = self.feedforward(x_sample, y_sample)

                # subtract epsilon to the w[k,d]
                self.w1 = np.copy(old_w1)
                self.w1[k, d] -= epsilon
                e_minus = self.feedforward(x_sample, y_sample)
                # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
                numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)
        return gradEw1, numericalGrad

    def gradient_check(self):
        epsilon = 1e-6
        _list = np.random.randint(self.x_train.shape[0], size=5)
        x_sample = np.array(self.x_train[_list, :])
        y_sample = np.array(self.y_train[_list, :])
        self.feedforward(x_sample, y_sample)
        print('---Start of gradient check---')
        old_w1 = self.w1
        old_w2 = self.w2
        gradEw1, numericalGrad1 = self.gradient_check_w1(epsilon, x_sample, y_sample)
        self.w1 = old_w1
        gradEw2, numericalGrad2 = self.gradient_check_w2(epsilon, x_sample, y_sample)
        # print and calculate the relative difference and difference
        print("The difference estimate for gradient of W1 is : ", np.max(np.abs(gradEw1 - numericalGrad1)))
        diff = np.linalg.norm(gradEw1 - numericalGrad1)/np.linalg.norm(gradEw1 + numericalGrad1)
        print("The relative difference estimate for gradient of W1 is : ", diff)
        print("The difference estimate for gradient of W2 is : ", np.max(np.abs(gradEw2 - numericalGrad2)))
        diff2 = np.linalg.norm(gradEw2 - numericalGrad2)/np.linalg.norm(gradEw2 + numericalGrad2)
        print("The relative difference estimate for gradient of W2 is : ", diff2)
        self.set_weights(old_w1, old_w2)

    def train(self):
        second_dimension = (self.x_train.shape[0] + self.batch_size // 2) // self.batch_size
        list_ew = np.zeros((self.epochs, second_dimension))
        for epoch in range(1, self.epochs+1):
            count = 0
            if epoch > 1:
                # Randomize data point
                permutation = np.random.permutation(self.x_train.shape[0])
                x_train = self.x_train[permutation]
                y_train = self.y_train[permutation]
            else:
                x_train, y_train = self.x_train, self.y_train

            for i in range(0, x_train.shape[0], self.batch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                if x_train.shape[0]-(i + 2*self.batch_size) < self.batch_size/2:
                    x_train_batch = x_train[i:]
                    y_train_batch = y_train[i:]
                else:
                    x_train_batch = x_train[i:i + self.batch_size]
                    y_train_batch = y_train[i:i + self.batch_size]

                ew = self.feedforward(x_train_batch, y_train_batch)
                list_ew[epoch-1, count] += ew
                count += 1
                new_w2, new_w1 = self.back_propagation(x_train_batch, y_train_batch)
                self.set_weights(new_w1, new_w2)
            # Show the current cost function on screen
            if epoch % 10 == 0 or epoch == 1 or epoch == self.epochs:
                cost = 0
                for ind in list_ew[epoch-1]:
                    cost = cost + ind
                cost = cost / len(list_ew[epoch-1])
                print('Epoch : %d, Cost function :%f' % (epoch, cost))
        self.plot_train(list_ew)

    def predict(self, x_test):
        a = x_test.dot(self.w1.transpose())
        if self.activation_function == 'softplus':
            z = softplus(a)
        elif self.activation_function == 'tanh':
            z = tanh(a)
        else:
            z = cos(a)
        z = np.hstack((np.ones((z.shape[0], 1)), z))
        s = z.dot(self.w2.transpose())
        y_test = softmax(s)
        # Hard classification decisions
        prediction = np.argmax(y_test, 1)
        return prediction

    def accuracy(self, pred, y_test):
        x = np.mean(pred == np.argmax(y_test, 1))
        x = x*100
        return x

    def plot_train(self, ew):
        cost = np.zeros((self.epochs, 1))
        for i in range(0, self.epochs):
            c = 0
            for elem in ew[i]:
                c += elem
            c = c / len(ew[i])
            cost[i] = c
        plt.plot(np.squeeze(cost))
        plt.ylabel('Cost')
        plt.xlabel('Epochs (per ten)')
        plt.title('Learning rate = '+str(self.learning_rate))
        plt.show()

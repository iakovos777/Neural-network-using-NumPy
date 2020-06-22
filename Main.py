import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Neural_network import NeuralNetwork
import time


def normalize(x_train, x_test):
    train = x_train.astype(float) / 255
    test = x_test.astype(float) / 255
    return train, test


# load dataset of mnist
def load_dataset_mnist():
    df = None

    y_train = []

    for i in range(10):
        tmp = pd.read_csv('data/mnist/train%d.txt' % i, header=None, sep=" ")
        # build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_train.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    train_data = df.values
    y_train = np.array(y_train)

    # load test files
    df = None

    y_test = []

    for i in range(10):
        tmp = pd.read_csv('data/mnist/test%d.txt' % i, header=None, sep=" ")
        # build labels - one hot vector

        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_test.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    test_data = df.values
    y_test = np.array(y_test)

    return train_data, test_data, y_train, y_test


def visualize_mnist(X_train):
    # plot 100 random images from the training set
    n = 100
    sqrt_n = int(n ** 0.5)
    samples = np.random.randint(X_train.shape[0], size=n)

    plt.figure(figsize=(11, 11))

    cnt = 0
    for i in samples:
        cnt += 1
        plt.subplot(sqrt_n, sqrt_n, cnt)
        plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')

    plt.show()


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def one_hot_encode(data):
    one_hot = np.zeros((data.shape[0], 10))
    one_hot[np.arange(data.shape[0]), data] = 1
    return one_hot


def load_dataset_cifar():
    features = None
    labels = None
    for i in range(5):
        fileName = 'data/cifar-10/data_batch_' + str(i + 1)
        data = unpickle(fileName)
        if i == 0:
            features = data[b'data']
            labels = np.array(data[b'labels'])
        else:
            features = np.append(features, data[b'data'], axis=0)
            labels = np.append(labels, data[b'labels'], axis=0)
    file_test = 'data/cifar-10/test_batch'
    test_data = unpickle(file_test)
    test_features = test_data[b'data']
    test_labels = np.array(test_data[b'labels'])
    features, test_features = normalize(features, test_features)
    #one-hot encoding here
    y_train = one_hot_encode(labels)
    y_test = one_hot_encode(test_labels)
    return features, test_features, y_train, y_test


def visualize_cifar(x_train, y_train,classes):
    # classes->store the name of classes
    ind = [None]*10
    # take one example from every class
    for li in range(10):
        for p in range(y_train.shape[0]):
            if li == np.argmax(y_train[p]):
                ind[li] = p
                break
    cnt = 0
    for i in ind:
        arr = x_train[i]
        R = arr[0:1024].reshape(32, 32)
        G = arr[1024:2048].reshape(32, 32)
        B = arr[2048:].reshape(32, 32)
        title = classes[cnt]
        cnt += 1
        #x_train[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = np.dstack((R, G, B))
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img, interpolation='bicubic')
        ax.set_title('Category = ' + title, fontsize=15)
    plt.show()


def cross_validation(x, y, percentage=0.8):
    split = int(percentage * x.shape[0])
    #shuffle
    index = np.random.permutation(x.shape[0])
    train_idx, val_idx = index[:split], index[split:]
    x_train, x_val = x[train_idx, :], x[val_idx, :]
    y_train, y_val = y[train_idx, :], y[val_idx, :]
    print("Records in Training Dataset", x_train.shape[0])
    print("Records in Validation Dataset", x_val.shape[0])
    return x_train, y_train, x_val, y_val


#method to shuffle data(alternative solution)
def shuffle_data(x, y):
    dataset = [(i, k) for i, k in zip(x, y)]
    np.random.shuffle(dataset)
    x = [i for (i, k) in dataset]
    y = [k for (i, k) in dataset]
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


# main function
def main(choice):
    if choice == 'mnist':
        # FOR MNIST DATASET
        x_train, x_test, y_train, y_test = load_dataset_mnist()
        permutation = np.random.permutation(x_train.shape[0])
        x_train = x_train[permutation]
        y_train = y_train[permutation]
        per = np.random.permutation(x_test.shape[0])
        x_test = x_test[per]
        y_test = y_test[per]
        # visualize dataset
        ans = input('Do you want to use visualize dataset?(Yes/No):')
        if ans == 'Yes' or ans == 'YES' or ans == 'yes':
            visualize_mnist(x_train)
        #Normalize dataset
        x_train, x_test = normalize(x_train, x_test)
        x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
        x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    else:
        # FOR CIFAR-10 DATASET
        x_train, x_test, y_train, y_test = load_dataset_cifar()
        x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
        x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
        cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # visualize dataset
        ans = input('Do you want to use visualize dataset?(Yes/No):')
        if ans == 'Yes' or ans == 'YES' or ans == 'yes':
            visualize_cifar(x_train, y_train,cifar_classes)
    answer = input('Do you want to use cross-validation?(Yes/No):')
    if answer == 'Yes' or answer == 'YES' or answer=='yes':
        x_train, y_train, x_test, y_test = cross_validation(x_train, y_train)
    hidden_neurons = int(input('Enter the number of hidden neurons:'))
    learning_rate = float(input('Enter the learning rate:'))
    lamda = float(input('Enter the term of regularization:'))
    activation_function = input('Enter which activation function you want to use(softplus/tanh/cos):')
    while activation_function != 'softplus' and activation_function != 'tanh' and activation_function != 'cos':
        print('Invalid activation function \n')
        activation_function = input('Enter which activation function you want to use(softplus/tanh/cos):')
    batch_size = int(input('Enter the size of every batch:'))
    epochs = int(input('Enter the number of epochs:'))
    nn = NeuralNetwork(x_train, y_train, hidden_neurons, learning_rate, lamda, activation_function, batch_size, epochs)
    #GRADIENT CHECK HERE
    an = input('Do you want to use gradient check?(Yes/No):')
    if an == 'Yes' or an == 'YES' or an == 'yes':
        nn.gradient_check()
    nn.xavier_weights()
    nn.train()
    pred_train = nn.predict(x_train)
    train_score = nn.accuracy(pred_train, y_train)
    print('Train accuracy : %f %%' % train_score)
    pred = nn.predict(x_test)
    test_score = nn.accuracy(pred, y_test)
    print('Test accuracy : %f %%' % test_score)


ch = input('Enter the name of dataset which you want to explore(CHOICES: mnist or cifar):')
#check if name is valid
while ch != 'mnist' and ch != 'cifar' and ch != 'cifar-10':
    print('Invalid name of dataset \n')
    ch = input('Enter the name of dataset which you want to explore(CHOICES: mnist or cifar):')
start_time = time.time()
main(ch)
print('---Execution time: %s seconds ---' % (time.time()-start_time))

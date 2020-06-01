import numpy as np
import matplotlib.pyplot as plt
import random
import time
import warnings
import murmurhash
import pickle
from countminsketch_mmh3_parallel import CountMinSketch
warnings.filterwarnings("ignore")

def loss_softmax(W, X, y_ind, reg):
    loss = 0
    num_train, dim = X.shape
    scores = X.dot(W) # [N, K]
    predicted_class = np.argmax(scores,axis=1)
    scores -= np.max(scores)
    scores_exp = np.exp(scores)
    correct_scores_exp = scores_exp[list(range(num_train)), y_ind] # [N, ]
    scores_exp_sum = np.sum(scores_exp, axis=1) # [N, ]
    loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    return loss,predicted_class

def loss_grad_softmax(W, X, y_ind, reg):
    """ Compute the loss and gradients using softmax with vectorized version"""
    loss = 0
    grad = np.zeros_like(W)
    num_train, dim = np.shape(X)

    scores = X.dot(W) # [N, K]
    # Shift scores so that the highest value is 0
    scores -= np.max(scores)
    scores_exp = np.exp(scores)
    correct_scores_exp = scores_exp[list(range(num_train)), y_ind] # [N, ]
    scores_exp_sum = np.sum(scores_exp, axis=1) # [N, ]
    loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    scores_exp_normalized = (scores_exp.T / scores_exp_sum).T
    # deal with the correct class
    scores_exp_normalized[list(range(num_train)), y_ind] -= 1 # [N, K]
    grad = X.T.dot(scores_exp_normalized)
    grad /= num_train
    grad += reg * W

    return loss, grad


np.random.seed(0)
random.seed(0)

filepath = 'KDD/kdd12.tr'

data_dimension = 54686452
number_of_classes = 2
step_size = 1

cms = CountMinSketch(5000,5,2)

epoch = 1
loss = 1
mini_batch_size = 5

indicies_in_minibatch = []
indicies_in_minibatch_flat = []
values_in_minibatch = []
data_label = []

training_loss_store = []
cnt_store = []
time_store = []

start_time = time.time()
while loss > 0.01 and epoch < 2:

    line = 'start'
    print('epoch = ', epoch)

    cnt = 0
    with open(filepath) as fp:
        while line and cnt< 20242:

            line = fp.readline()
            line = line[:-1]
            line_splited = line.split(' ')
            data_label.append(int((int(line_splited[0])+1)/2))

            indicies_in_data = []
            values_in_data = []
            for feature_index, feature_value in enumerate(line_splited[1:]):
                feature,value = feature_value.split(':')
                feature = int(feature)-1
                value = float(value)
                indicies_in_data.append(feature)
                indicies_in_minibatch_flat.append(feature)
                values_in_data.append(value)

            indicies_in_minibatch.append(indicies_in_data)
            values_in_minibatch.append(values_in_data)

            if cnt % mini_batch_size == mini_batch_size - 1:

                indicies_in_minibatch_flat = list(set(indicies_in_minibatch_flat))
                indicies_in_minibatch_flat_dict = dict(zip(indicies_in_minibatch_flat,range(len(indicies_in_minibatch_flat))))

                data_vector = np.zeros((len(indicies_in_minibatch_flat) ,mini_batch_size))
                for data_ind in range(mini_batch_size):
                    for item_ind,item in enumerate(indicies_in_minibatch[data_ind]):
                        data_vector[indicies_in_minibatch_flat_dict[item],data_ind] = values_in_minibatch[data_ind][item_ind]


                X = data_vector.T
                y_ind = np.asarray(data_label)

                WW = cms.query(list( map(str,indicies_in_minibatch_flat)))
                loss, grad = loss_grad_softmax(WW, X, y_ind, reg=0)
                cms.add(list( map(str,indicies_in_minibatch_flat) ),- step_size * grad)

                indicies_in_minibatch = []
                values_in_minibatch = []
                data_label = []
                indicies_in_minibatch_flat = []

            cnt += 1

            if cnt % 1000 == 0:

                training_loss_store.append(loss)
                cnt_store.append(cnt)
                time_store.append(time.time()-start_time)
                print(cnt)
                print(loss)

    epoch += 1

fp.close()

## save the convergence trajectories
filehandler = open("results/MISSION-b5-step01-w5000-d5-loss.p", 'wb')
pickle.dump(training_loss_store, filehandler)
filehandler.close()
filehandler = open("results/MISSION-b5-step01-w5000-d5-cnt.p", 'wb')
pickle.dump(cnt_store, filehandler)
filehandler.close()
filehandler = open("results/MISSION-b5-step01-w5000-d5-time.p", 'wb')
pickle.dump(time_store, filehandler)
filehandler.close()


# finding the test loss
indicies_in_minibatch = []
indicies_in_minibatch_flat = []
values_in_minibatch = []
total_test_loss = []
data_label = []
line = 'start'
cnt = 0
filepath = 'RCV1/rcv1_test.binary'
accuracy_list = []
predicted_class_TEST = np.asarray([])
ground_truth_TEST = np.asarray([])
mini_batch_size = 10000
with open(filepath) as fp:
    while line and cnt < 677399 :

        if cnt % 10000 == 0:
            print(cnt)

        line = fp.readline()
        line = line[:-1]
        line_splited = line.split(' ')
        data_label.append(int((int(line_splited[0])+1)/2))

        indicies_in_data = []
        values_in_data = []
        for feature_index, feature_value in enumerate(line_splited[1:]):
            feature,value = feature_value.split(':')
            feature = int(feature)-1
            value = float(value)
            indicies_in_data.append(feature)
            indicies_in_minibatch_flat.append(feature)
            values_in_data.append(value)


        indicies_in_minibatch.append(indicies_in_data)
        values_in_minibatch.append(values_in_data)

        if cnt % mini_batch_size == mini_batch_size - 1:

            indicies_in_minibatch_flat = list(set(indicies_in_minibatch_flat))
            indicies_in_minibatch_flat_dict = dict(zip(indicies_in_minibatch_flat,range(len(indicies_in_minibatch_flat))))

            data_vector = np.zeros((len(indicies_in_minibatch_flat) ,mini_batch_size))
            for data_ind in range(mini_batch_size):
                for item_ind,item in enumerate(indicies_in_minibatch[data_ind]):
                    data_vector[indicies_in_minibatch_flat_dict[item],data_ind] = values_in_minibatch[data_ind][item_ind]


            X = data_vector.T
            y_ind = data_label

            WW = cms.query(list( map(str,indicies_in_minibatch_flat)))
            loss,predicted_class = loss_softmax(WW, X, y_ind, reg=0)
            total_test_loss.append(loss)

            predicted_class_TEST = np.append(predicted_class_TEST,predicted_class)
            ground_truth_TEST = np.append(ground_truth_TEST,data_label)

            indicies_in_minibatch = []
            values_in_minibatch = []
            data_label = []
            indicies_in_minibatch_flat = []

        cnt += 1


print('***')
print('average test loss is ')
print(np.mean(np.asarray(total_test_loss)))
print('accuracy')
acc_val = (predicted_class_TEST == ground_truth_TEST).sum() / len(ground_truth_TEST)
print(acc_val)

## store
file = open("results/MISSION-b5-step01-w5000-d5-log.txt","w")
file.write("Average test loss %f \n" %np.mean(np.asarray(total_test_loss)) )
file.write("Average test accuracy %f \n" % acc_val )
file.close()

filehandler = open("results/MISSION-b5-step01-w5000-d5-test-ground-truth.p", 'wb')
pickle.dump(ground_truth_TEST, filehandler)
filehandler.close()

filehandler = open("results/MISSION-b5-step01-w5000-d5-test-prediction.p", 'wb')
pickle.dump(predicted_class_TEST, filehandler)
filehandler.close()

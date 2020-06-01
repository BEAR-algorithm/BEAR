import numpy as np
from countminsketch_mmh3_parallel import CountMinSketch
import matplotlib.pyplot as plt
import random
import time
import warnings
import murmurhash
import heapq
from scipy import sparse
from scipy.sparse import linalg
import pickle
from topkheap import mytopkheap

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



def OLBFGS(grad_sp,active_set_grad,S,R,data_dimension,cc):
    mm = len(R)

    # find ro
    ro = []
    for i in range(mm):
        temp = sparse.diags(sparse.csr_matrix.diagonal((np.transpose(S[i])).dot(R[i])))
        ro.append(sparse.linalg.inv(temp))

    # initialize q
    q = grad_sp

    # first loop
    alpha = []
    for i in range(mm):
        j = mm - i - 1
        temp = ro[j] * (np.transpose(S[j])).dot(q)
        temp = sparse.diags(sparse.csr_matrix.diagonal(temp))
        q =  q - R[j] * temp
        alpha.insert(0,temp)

    denominator =  ((np.transpose(R[mm-1])).dot(R[mm-1]))
    denominator = sparse.linalg.inv( sparse.diags(sparse.csr_matrix.diagonal(denominator)))
    z = q.dot( sparse.linalg.inv(ro[mm-1])).dot(denominator)

    # second Loop
    for i in range(mm):
        beta = ro[i] * (R[i].T).dot(z)
        beta = sparse.diags(sparse.csr_matrix.diagonal(beta))
        z = z + S[i] * (alpha[i]-beta)

    return z

np.random.seed(0)
random.seed(0)

filepath = '/Users/amirali/Desktop/RCV1/rcv1_train.binary'

data_dimension = 47236
number_of_classes = 2
m = 5
mini_batch_size = 5
step_size = 0.01 # step size should scale with mini batch size, 100(1), 1(0.01)
reg_param = 0


cms = CountMinSketch(5000,5,2)

epoch = 1
loss = 1

Sm = []
Rm = []

indicies_in_minibatch = []
indicies_in_minibatch_flat = []
values_in_minibatch = []
data_label = []

training_loss_store = []
cnt_store = []
time_store = []

start_time = time.time()
total_batch_cnt = 0
while loss > 0.01 and epoch < 2:

    line = 'start'
    print('epoch = ', epoch)

    cnt = 0
    batch_cnt = 0
    with open(filepath) as fp:
        while line and cnt< 20242: ## 20242 number of training data points

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

                WW = cms.query(list(map(str,indicies_in_minibatch_flat)))
                loss1, grad1 = loss_grad_softmax(WW, X, y_ind, reg=reg_param)

                if total_batch_cnt > 0:

                    col_matrix = np.zeros(np.shape(grad1)) ## dd x cc
                    ind_matrix = np.zeros(np.shape(grad1)) ## dd x cc
                    for i in range(number_of_classes):
                        col_matrix[:,i] = i  # col
                        ind_matrix[:,i] = indicies_in_minibatch_flat # row

                    grad_sp = sparse.coo_matrix((np.matrix.flatten(grad1).tolist(),(np.matrix.flatten(ind_matrix.astype(int)).tolist(),np.matrix.flatten(col_matrix.astype(int)).tolist())),shape=(data_dimension, 2))
                    dd = OLBFGS(grad_sp,indicies_in_minibatch_flat,Sm,Rm,data_dimension,number_of_classes)
                    cms.add(list( map(str,indicies_in_minibatch_flat) ),- step_size * dd[indicies_in_minibatch_flat,:].toarray())
                    WW = cms.query(list(map(str,indicies_in_minibatch_flat)))
                    loss2, grad2 = loss_grad_softmax(WW, X, y_ind, reg=reg_param)
                    #s =  sparse.coo_matrix(- step_size * dd)
                    s = sparse.coo_matrix((np.matrix.flatten( - step_size * dd[indicies_in_minibatch_flat,:].toarray()).tolist(),(np.matrix.flatten(ind_matrix.astype(int)).tolist(),np.matrix.flatten(col_matrix.astype(int)).tolist())),shape=(data_dimension, 2))
                    r = sparse.coo_matrix((np.matrix.flatten(grad2 - grad1).tolist(),(np.matrix.flatten(ind_matrix.astype(int)).tolist(),np.matrix.flatten(col_matrix.astype(int)).tolist())),shape=(data_dimension, 2))

                # this is for the very first iteration, we do NOT do lbfgs, we only do a simple gradient
                else:
                    cms.add(list( map(str,indicies_in_minibatch_flat) ),- step_size * grad1)
                    WW = cms.query(list(map(str,indicies_in_minibatch_flat)))
                    loss2, grad2 = loss_grad_softmax(WW, X, y_ind, reg=reg_param)
                    # we now covert grad1 to sparse fromat to prepare it for S matric
                    col_matrix = np.zeros(np.shape(grad1)) ## dd x cc
                    ind_matrix = np.zeros(np.shape(grad1)) ## dd x cc
                    for i in range(number_of_classes):
                        col_matrix[:,i] = i  # col
                        ind_matrix[:,i] = indicies_in_minibatch_flat # row
                    s = sparse.coo_matrix((np.matrix.flatten(- step_size * grad1).tolist(),(np.matrix.flatten(ind_matrix.astype(int)).tolist(),np.matrix.flatten(col_matrix.astype(int)).tolist())),shape=(data_dimension, 2))
                    r = sparse.coo_matrix((np.matrix.flatten(grad2 - grad1).tolist(),(np.matrix.flatten(ind_matrix.astype(int)).tolist(),np.matrix.flatten(col_matrix.astype(int)).tolist())),shape=(data_dimension, 2))

                # prepare the S and R matrices required for LBFGS
                if total_batch_cnt < m:
                    Sm.append(s)
                    Rm.append(r)

                else:
                    Sm[0:(m-2)] = Sm[1:(m-1)]
                    Rm[0:(m-2)] = Rm[1:(m-1)]

                    Sm[m-1] = s
                    Rm[m-1] = r

                indicies_in_minibatch = []
                values_in_minibatch = []
                data_label = []
                indicies_in_minibatch_flat = []

                batch_cnt += 1
                total_batch_cnt += 1

                if total_batch_cnt % 200 == 0:
                    training_loss_store.append(loss1)
                    cnt_store.append(cnt+1)
                    time_store.append(time.time()-start_time)

                    print(cnt+1)
                    print(loss1)



            cnt += 1


    epoch += 1


filehandler = open("results/LBFGS-b5-step01-m5-w5000-d5-loss.p", 'wb')
pickle.dump(training_loss_store, filehandler)
filehandler.close()
filehandler = open("results/LBFGS-b5-step01-m5-w5000-d5-cnt.p", 'wb')
pickle.dump(cnt_store, filehandler)
filehandler.close()
filehandler = open("results/LBFGS-b5-step01-m5-w5000-d5-time.p", 'wb')
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
filepath = '/Users/amirali/Desktop/RCV1/rcv1_test.binary'
accuracy_list = []
predicted_class_TEST = np.asarray([])
ground_truth_TEST = np.asarray([])
mini_batch_size = 10000
with open(filepath) as fp:
    while line and cnt < 677399 : ## 677399 number of test data points

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
file = open("results/LBFGS-b5-step01-m5-w5000-d5-log.txt","w")
file.write("Average test loss %f \n" %np.mean(np.asarray(total_test_loss)) )
file.write("Average test accuracy %f \n" % acc_val )
file.close()

filehandler = open("results/LBFGS-b5-step01-m5-w5000-d5-test-ground-truth.p", 'wb')
pickle.dump(ground_truth_TEST, filehandler)
filehandler.close()

filehandler = open("results/LBFGS-b5-step01-m5-w5000-d5-test-prediction.p", 'wb')
pickle.dump(predicted_class_TEST, filehandler)
filehandler.close()

import numpy as np
import tensorflow as tf
import random
import sys, os
import json
import argparse
from parser import Parser
from datamanager import DataManager
from actor import ActorNetwork
from LSTM_critic import LSTM_CriticNetwork
import time
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#get parse
argv = sys.argv[1:]
parser = Parser().getParser()
args, _ = parser.parse_known_args(argv)
random.seed(args.seed)

#get data
#File name (train/dev/test) to list of strings (lines in those files) dict
dataManager = DataManager(args.dataset)


#Get the three lists of train, dev, and test data. Each list contains set of lines in these files.
train_data, dev_data, test_data = dataManager.getdata(args.grained, args.maxlenth)

#Corresponding to each word in the vocab (all files), get the 300 dim vectors. Shape: (n,300)
word_vector = dataManager.get_wordvector(args.word_vector)
#print(word_vector.shape)
#time.sleep(10)
#exit(0)

if args.fasttest == 1:
    train_data = train_data[:100]
    dev_data = dev_data[:20]
    test_data = test_data[:20]
#print ("train_data ", train_data)
#print ("dev_data", len(dev_data))
#print ("test_data", len(test_data))

def sampling_RL(sess, actor, inputs, vec, lenth, epsilon=0., Random=True):
    '''
    This method aids in movement between the actor and LSTM network for a sentence, and generates a set of states and
    actions for the sentence. After this method completes, we have a set of retained words from the sentence. No training
    happens in this method. The networks are only used and the set of states and actions are generated.
    '''
    #print("INSIDE SAMPLING RL\n")
    #print epsilon

    #current_lower_state is the initial state for the actor. This is a set of zeros, of dimension (1,600) or (1,2*dim)
    current_lower_state = np.zeros((1, 2*args.dim), dtype=np.float32)
    actions = []
    states = []
    #sampling actions

    #For each word in the sentence
    for pos in range(lenth):
        #Predict the action (the input state is both the word vector representation of the sentence and the current_lower_state)
        predicted = actor.predict_target(current_lower_state, [vec[0][pos]])
        #Append the input state (combination of word vector representation of the sentence and the state representation)
        states.append([current_lower_state, [vec[0][pos]]])
        #Epsilon greedy strategy of sampling action from the actor network (either random or argmax)
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < predicted[0] else 1)
            else:
                action = (1 if random.random() < predicted[0] else 0)
        else:
            action = np.argmax(predicted)
        #Append action in actions
        actions.append(action)
        #If the sampled action is 1, feed the current_lower_state to the critic LSTM, and generate the o/p and the new state
        #This state is again fed to the actor network in the beginning of this for loop
        if action == 1:
            out_d, current_lower_state = critic.lower_LSTM_target(current_lower_state, [[inputs[pos]]])

    Rinput = []
    #Now for every action predicted for each state (for one sentence)
    for (i, a) in enumerate(actions):
        #If action = 1
        if a == 1:
            #Append the word to Rinput
            Rinput.append(inputs[i])
    #print("######")
    #print(Rinput)
    #Rinput: all words which have been retained
    #Rlenth: number of words in the sentence which have been retained
    Rlenth = len(Rinput)
    if Rlenth == 0:
        actions[lenth-2] = 1
        Rinput.append(inputs[lenth-2])
        Rlenth = 1
    #Pad the remaining locations with 0 (thus, Rinput contains the word IDs that were retained and 0 pads)
    Rinput += [0] * (args.maxlenth - Rlenth)
    #Return the set of states and actions, along with words retained and their number
    return actions, states, Rinput, Rlenth

def train(sess, actor, critic, train_data, batchsize, samplecnt=5, LSTM_trainable=True, RL_trainable=True):
    print ("training : total ", len(train_data), "nodes.")
    random.shuffle(train_data)
    for b in range(int(len(train_data) / batchsize)):
        datas = train_data[b * batchsize: (b+1) * batchsize]
        totloss = 0.
        critic.assign_active_network()
        actor.assign_active_network()
        for j in range(batchsize):
            #prepare
            data = datas[j]

            #inputs:(maxlen,), solution:(grained,), lenth: length of sentence
            #Solution is the ground truth data for the sentence
            #inputs is a single sentence
            #lenth??
            inputs, solution, lenth = data['words'], data['solution'], data['lenth']
            #train the actor network
            if RL_trainable:
                #Set of actions, states, and loss for iterations: actionlist[1] will have actions for the first iteration
                actionlist, statelist, losslist = [], [], []
                aveloss = 0.
                for i in range(samplecnt):
                    #Get the set of actions and states from this single sentence
                    #States and actions are also vectors. Number of states = number of actions
                    #actions = [0,1,1,0,1 ...]
                    #state is a large vector of numbers
                    #Rinput has length=maxlength (contains words retained and padded 0s)
                    actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, args.epsilon, Random=True)
                    #Append set of states and actions for each iteration
                    actionlist.append(actions)
                    statelist.append(states)
                    #Get output and loss of the critic network: Rinput is generated by the actor network's sampling_RL method
                    #here, out is the output of CNet. Thus, the critic network contains both the LSTM and the CNet.
                    out, loss = critic.getloss([Rinput], [Rlenth], [solution])
                    #Rlenth is the number of retained words, and lenth is the actual sentence length (paper)
                    #The above Rlenth/lenth quantity is added to the loss (equation 6 of the paper)
                    loss += (float(Rlenth) / lenth) **2 *0.15
                    aveloss += loss
                    #Append the loss for each iteration
                    losslist.append(loss)
                #Average loss of the critic network over samplecnt iterations
                aveloss /= samplecnt
                #Total loss of the critic network over samplecnt iterations
                totloss += aveloss
                grad = None
                #Train the critic network now (if LSTM_trainable=True) based on the loss calculated, based on the existing input and ground truth.
                #This input is generated by the actor network/sampling_RL.
                if LSTM_trainable:
                    out, loss, _ = critic.train([Rinput], [Rlenth], [solution])

                #Now train the actor network based on the gradient (Policy Gradient algo)??
                for i in range(samplecnt):
                    for pos in range(len(actionlist[i])):
                        rr = [0., 0.]
                        rr[actionlist[i][pos]] = (losslist[i] - aveloss) * args.alpha
                        g = actor.get_gradient(statelist[i][pos][0], statelist[i][pos][1], rr)
                        if grad == None:
                            grad = g
                        else:
                            grad[0] += g[0]
                            grad[1] += g[1]
                            grad[2] += g[2]
                actor.train(grad)
            #If actor training not needed (RL_trainable=False), just train the critic network and calculate the total loss
            #In this case, train it based on original inputs (not actor generated)
            else:
                if len(inputs)==300:
                    out, loss, _ = critic.train([inputs], [lenth], [solution])
                    totloss += loss
                else:
                    continue
        #If actor is trained, update the parameters of both actor and critic
        if RL_trainable:
            actor.update_target_network()
            if LSTM_trainable:
                critic.update_target_network()
        #??
        else:
            critic.assign_target_network()
        #Print accuracies after every 500 batches
        if (b + 1) % 500 == 0:
            acc_test = test(sess, actor, critic, test_data, noRL= not RL_trainable)
            acc_dev = test(sess, actor, critic, dev_data, noRL= not RL_trainable)
            print ("batch ",b , "total loss ", totloss, "----test: ", acc_test, "| dev: ", acc_dev)


def test(sess, actor, critic, test_data, noRL=False):
    acc = 0
    for i in range(len(test_data)):
        #prepare
        data = test_data[i]
        inputs, solution, lenth = data['words'], data['solution'], data['lenth']

        #predict
        if noRL:
            if len(inputs)==70:
                out = critic.predict_target([inputs], [lenth])
            else:
                continue
        else:
            actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, Random=False)
            out = critic.predict_target([Rinput], [Rlenth])
        if np.argmax(out) == np.argmax(solution):
            acc += 1
    return float(acc) / len(test_data)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    #model
    #The critic network produces the structured representation (state vector) and the final sentence representation that is fed to the CNet
    critic = LSTM_CriticNetwork(sess, args.dim, args.optimizer, args.lr, args.tau, args.grained, args.maxlenth, args.dropout, word_vector)
    #The actor network produces the actions corresponding to each state
    actor = ActorNetwork(sess, args.dim, args.optimizer, args.lr, args.tau)

    #print the trainable variables
    for item in tf.trainable_variables():
        print (item.name, item.get_shape())

    saver = tf.train.Saver()

    #RLpretrain is pretraining the actor model keeping the LSTM and CNet parameters fixed
    if args.RLpretrain != '':
        pass
    #LSTMpretrain is pretraining the critic and CNet keeping the actor parameters fixed
    elif args.LSTMpretrain == '':
        sess.run(tf.global_variables_initializer())
        for i in range(0, 2):
            #Here we train only the critic and CNet parameters, as RL_trainable = False. So, actor network is not updated.
            train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt, RL_trainable=False)
            #Update the parameters of the LSTM_CriticNetwork
            critic.assign_target_network()
            #Calculate test and development accuracy for the critic model
            acc_test = test(sess, actor, critic, test_data, True)
            acc_dev = test(sess, actor, critic, dev_data, True)
            print ("LSTM_only ",i, "----test: ", acc_test, "| dev: ", acc_dev)
            saver.save(sess, "checkpoints/"+args.name+"_base", global_step=i)
        print ("LSTM pretrain OK")
    else:
        print ("Load LSTM from ", args.LSTMpretrain)
        saver.restore(sess, args.LSTMpretrain)

    print ("epsilon", args.epsilon)
    #Actor network pretraining
    if args.RLpretrain == '':
        for i in range(0, 5):
            #Here LSTM_trainable = False. So, we train the actor network, keeping the LSTM critic network and CNet static.
            train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt, LSTM_trainable=False)
            #Calculate test and development accuracy of the actor network
            acc_test = test(sess, actor, critic, test_data)
            acc_dev = test(sess, actor, critic, dev_data)
            print ("RL pretrain ", i, "----test: ", acc_test, "| dev: ", acc_dev)
            saver.save(sess, "checkpoints/"+args.name+"_RLpre", global_step=i)
        print ("RL pretrain OK")
    else:
        print ("Load RL from", args.RLpretrain)
        saver.restore(sess, args.RLpretrain)

    #Train all three components together (RL, LSTM, and CNet)
    for e in range(args.epoch):
        #Here, both LSTM_trainable and RL_trainable is true by default. So, everything is trained together
        train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt)
        #Calculate the final accuracy for validation and testing
        acc_test = test(sess, actor, critic, test_data)
        acc_dev = test(sess, actor, critic, dev_data)
        print ("epoch ", e, "----test: ", acc_test, "| dev: ", acc_dev)
        saver.save(sess, "checkpoints/"+args.name, global_step=e)

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav
from modules.util import TensorBoardSummaryWriter

import numpy as np


class LSTM_net():

    def __init__(self, obs_size, nb_hidden=128, action_size=16):

        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.action_size = action_size

        def __graph__():
            tf.reset_default_graph()

            # entry points
            features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')
            init_state_c_, init_state_h_ = ( tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2) )
            action_ = tf.placeholder(tf.int32, name='ground_truth_action')

            # input projection
            Wi = tf.get_variable('Wi', [obs_size, nb_hidden],
                                 initializer=xav())
            bi = tf.get_variable('bi', [nb_hidden],
                                 initializer=tf.constant_initializer(0.))

            # add relu/tanh here if necessary
            projected_features = tf.matmul(features_, Wi) + bi

            lstm_f = tf.contrib.rnn.LSTMCell(nb_hidden, state_is_tuple=True)

            lstm_op, state = lstm_f(inputs=projected_features, state=(init_state_c_, init_state_h_))

            # reshape LSTM's state tuple (2,128) -> (1,256)
            state_reshaped = tf.concat(axis=1, values=(state.c, state.h))
            
            
            # define action attention variables
            action_projection = tf.placeholder(tf.float32, [300, action_size], name='action_projection')
            action_one_hot = tf.placeholder(tf.float32, [action_size], name='action_one_hot')
            expanded_action_one_hot = tf.expand_dims(action_one_hot, 1)
            
            # action_encoding => scalar value
            #action_encoding = tf.tanh(tf.matmul(action_projection, expanded_action_one_hot))
            action_encoding = tf.matmul(action_projection, expanded_action_one_hot)
            action_encoding = tf.transpose(action_encoding)
            
            # Get action weights (probability distribution of each action encoding)
            W_action = tf.get_variable('W_action', [300, 1], initializer=xav())
            
            # output : 1 dimension scalar value (current system action projection value)
            # 이거 전 액션에 대한거임 변수명때문에 헷갈리지 말것.
            current_projected_action = tf.matmul(action_encoding, W_action)
            
            prev_projected_actions = tf.placeholder(tf.float32, [None, 1], name='prev_projected_actions')

            # shape : [number of prev_utter + current_utter, 1]
            projected_actions = tf.concat([prev_projected_actions, current_projected_action], 0)
            
            # shape : [1, number of prev_utter + current_utter]
            transposed_projected_actions = tf.transpose(projected_actions)
            
            # output shape : [number of prev_utter + current_utter]
            action_weights = tf.nn.softmax(transposed_projected_actions)

            prev_action_encodings = tf.placeholder(tf.float32, [None, 300], name='prev_actions')
            action_encodings = tf.concat([prev_action_encodings, action_encoding], 0)
            # output shape : (1, 300)
            weighted_sum = tf.matmul(action_weights, action_encodings)
            
            # concate lstm features with weighted sum attention feature
            concatenated_features = tf.concat([state_reshaped, weighted_sum], 1)

            # output projection
            Wo = tf.get_variable('Wo', [556, action_size],
                    initializer=xav())
            bo = tf.get_variable('bo', [action_size], 
                    initializer=tf.constant_initializer(0.))
            # get logits
            logits = tf.matmul(concatenated_features, Wo) + bo
            
            probs = tf.squeeze(tf.nn.softmax(logits))
            
            # prediction
            prediction = tf.arg_max(probs, dimension=0)

            # loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_)

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(0.2, global_step,
                                                       150000, 0.5, staircase=True)
            # train op
            train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)

            # attach symbols to self
            self.loss = loss
            self.prediction = prediction
            self.probs = probs
            self.logits = logits
            self.state = state
            self.train_op = train_op

            # attach placeholders
            self.features_ = features_
            self.init_state_c_ = init_state_c_
            self.init_state_h_ = init_state_h_
            self.action_ = action_
            # self.action_mask_ = action_mask_
            
            # attention placeholders
            self.action_projection = action_projection
            self.action_one_hot = action_one_hot
            self.prev_action_encodings = prev_action_encodings
            self.prev_projected_actions = prev_projected_actions
            
            # attention values
            self.action_encodings = action_encodings
            self.projected_actions = projected_actions
            

        # build graph
        __graph__()

        # start a session; attach to self
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        self.sess = sess

        board_root_dir = '/root/jude/project/bAbI6_HCN/tensorboard/58_shuffle/'
        self.summary_writer = TensorBoardSummaryWriter(board_root_dir, self.sess, self.prediction.graph)
        
        # set init state to zeros
        self.init_state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)

        self.prev_action_encodings_ = np.zeros([1, 300], dtype=np.float32)
        self.prev_projected_actions_ = np.zeros([1, 1], dtype=np.float32)

    # forward propagation
    def forward(self, features, action_projection, action_one_hot=np.zeros(58)):
        # forward
        probs, prediction, state_c, state_h, action_encodings, projected_actions = self.sess.run( [self.probs, self.prediction, self.state.c, self.state.h, self.action_encodings, self.projected_actions],
                feed_dict = { 
                    self.features_ : features.reshape([1,self.obs_size]), 
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_projection : action_projection,
                    self.action_one_hot : action_one_hot,
                    self.prev_action_encodings : self.prev_action_encodings_,
                    self.prev_projected_actions : self.prev_projected_actions_
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h

        self.prev_action_encodings_ = action_encodings
        self.prev_projected_actions_ = projected_actions
        # return argmax
        return prediction

    # training
    def train_step(self, features, action, action_projection, action_one_hot=np.zeros(58)):
        _, loss_value, state_c, state_h, action_encodings, projected_actions = self.sess.run( [self.train_op, self.loss, self.state.c, self.state.h, self.action_encodings, self.projected_actions],
                feed_dict = {
                    self.features_ : features.reshape([1, self.obs_size]),
                    self.action_ : [action],
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_projection : action_projection,
                    self.action_one_hot : action_one_hot,
                    self.prev_action_encodings : self.prev_action_encodings_,
                    self.prev_projected_actions: self.prev_projected_actions_
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        self.prev_action_encodings_ = action_encodings
        self.prev_projected_actions_ = projected_actions
        
        return loss_value
    
    def reset_state(self):
        # set init state to zeros
        self.init_state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)
    
    def reset_attention(self):
        self.prev_action_encodings_ = np.zeros([1, 300], dtype=np.float32)
        self.prev_projected_actions_ = np.zeros([1, 1], dtype=np.float32)
    
    # save session to checkpoint
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'ckpt_bAbI6_58/hcn.ckpt', global_step=0)
        print('\n:: saved to ckpt/hcn.ckpt \n')

    # restore session from checkpoint
    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('ckpt_bAbI6_58/')
        if ckpt and ckpt.model_checkpoint_path:
            print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('\n:: <ERR> checkpoint not found! \n')

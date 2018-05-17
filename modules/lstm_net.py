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
            self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")

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
            state_reshaped = tf.nn.dropout(state_reshaped, self.dropout)
            
            # define user utterance memory
            prev_hidden_states = tf.placeholder(tf.float32, [None, nb_hidden*2], name='prev_hidden_states')
            W_user = tf.get_variable('W_user', [nb_hidden*2, nb_hidden*2], initializer=xav())
            
            
            # (None, 256) x (256, 256) x (256, 1) => (None, 1)
            user_attention_score = tf.matmul(tf.matmul(prev_hidden_states, W_user), tf.transpose(state_reshaped))
            # (None)
            user_attention_weights = tf.nn.softmax(tf.transpose(user_attention_score))
            # (None, 256)
            user_encodings = prev_hidden_states
            # (None) x (None, 256) => (1, 256)
            user_weighted_sum = tf.matmul(user_attention_weights, user_encodings)
            user_weighted_sum = tf.nn.dropout(user_weighted_sum, self.dropout)
            
            # define action attention variables
            action_projection = tf.placeholder(tf.float32, [300, action_size], name='action_projection')
            action_one_hot = tf.placeholder(tf.float32, [action_size], name='action_one_hot')
            expanded_action_one_hot = tf.expand_dims(action_one_hot, 1)
            
            # action_encoding => (300 x 1) 현재 메모리값임
            action_encoding = tf.matmul(action_projection, expanded_action_one_hot)
            action_encoding = tf.nn.dropout(action_encoding, self.dropout)
            # (1 x 300)
            action_encoding = tf.transpose(action_encoding)
            
            
            W_action = tf.get_variable('W_action', [300, nb_hidden*2], initializer=xav())
            
            # output : 1 dimension scalar value (current system action projection value) 이거 전 액션에 대한거임 변수명때문에 헷갈리지 말것.
            # 1 x 1
            transposed_hidden_state = tf.transpose(state_reshaped) # 256 x 1
           
            # 이전 시스템 메모리값들임
            prev_action_encodings = tf.placeholder(tf.float32, [None, 300], name='prev_actions')
            # output : [None, 1]
            prev_projected_actions = tf.matmul(tf.matmul(prev_action_encodings, W_action), transposed_hidden_state)

            # shape : [number of prev_utter, 1]
            projected_actions = prev_projected_actions
            
            # shape : [1, number of prev_utter]
            transposed_projected_actions = tf.transpose(projected_actions)
            
            # output shape : [number of prev_utter]
            # Get action weights (probability distribution of each action encodings)
            action_weights = tf.nn.softmax(transposed_projected_actions)
            
            action_encodings = prev_action_encodings
            # output shape : (1, 300)
            system_weighted_sum = tf.matmul(action_weights, action_encodings)
            system_weighted_sum = tf.nn.dropout(system_weighted_sum, self.dropout)
            
            # 이 밑에 부분 3가지로 실험할 것. (1. +, 2. AVG, 3.POOLING)
            sum_features = tf.reduce_sum([state_reshaped, user_weighted_sum, system_weighted_sum], 0)
            # avg_features = tf.reduce_mean([state_reshaped, user_weighted_sum, system_weighted_sum], 0)
            # 3. pooled_features = tf.reduce_max([state_reshaped, user_weighted_sum, system_weighted_sum], 0)
            
            # output projection
            Wo = tf.get_variable('Wo', [300, action_size],
                    initializer=xav())
            bo = tf.get_variable('bo', [action_size],
                    initializer=tf.constant_initializer(0.))
            # get logits
            logits = tf.matmul(sum_features, Wo) + bo
            
            # concate lstm features with weighted sum attention feature
            # concatenated_features = tf.concat([state_reshaped, user_weighted_sum, system_weighted_sum], 1)
            # concatenated_features = tf.nn.dropout(concatenated_features, self.dropout)
            
            # # output projection
            # Wo = tf.get_variable('Wo', [300 + 256 + 256, action_size],
            #         initializer=xav())
            # bo = tf.get_variable('bo', [action_size],
            #         initializer=tf.constant_initializer(0.))
            
            # # get logits
            # logits = tf.matmul(concatenated_features, Wo) + bo
            
            probs = tf.squeeze(tf.nn.softmax(logits))
            
            # prediction
            prediction = tf.arg_max(probs, dimension=0)

            # loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_)

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(0.25, global_step,
                                                       200000, 0.8, staircase=True)
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
            
            # user attention values
            self.prev_hidden_states = prev_hidden_states
            self.user_encodings = user_encodings
            
            # attention placeholders
            self.action_projection = action_projection
            self.action_one_hot = action_one_hot
            self.prev_action_encodings = prev_action_encodings
            
            # attention values
            self.action_encoding = action_encoding
            self.action_encodings = action_encodings
            self.projected_actions = projected_actions

            self.user_attention_weights = user_attention_weights
            self.action_weights = action_weights
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
        self.prev_hidden_states_ = np.zeros([1, nb_hidden*2], dtype=np.float32)

    # forward propagation
    def forward(self, features, action_projection, action_one_hot=np.zeros(58)):
        # forward
        probs, prediction, state_c, state_h, action_encoding, action_encodings, user_encodings, user_attention_weights, action_weights = self.sess.run( [self.probs, self.prediction, self.state.c, self.state.h, self.action_encoding, self.action_encodings, self.user_encodings, self.user_attention_weights, self.action_weights],
                feed_dict = { 
                    self.features_ : features.reshape([1,self.obs_size]), 
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_projection : action_projection,
                    self.action_one_hot : action_one_hot,
                    self.prev_action_encodings : self.prev_action_encodings_,
                    self.prev_hidden_states: self.prev_hidden_states_,
                    self.dropout: 1.0
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h

        self.prev_action_encodings_ = np.concatenate((action_encodings, action_encoding), axis=0)
        current_hidden = np.concatenate((state_c, state_h), axis=1)
        self.prev_hidden_states_ = np.concatenate((user_encodings, current_hidden), axis=0)
        # return argmax
        return prediction, user_attention_weights, action_weights

    # training
    def train_step(self, features, action, action_projection, action_one_hot=np.zeros(58)):
        _, loss_value, state_c, state_h, action_encoding, action_encodings, user_encodings = self.sess.run( [self.train_op, self.loss, self.state.c, self.state.h, self.action_encoding, self.action_encodings, self.user_encodings],
                feed_dict = {
                    self.features_ : features.reshape([1, self.obs_size]),
                    self.action_ : [action],
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h,
                    self.action_projection : action_projection,
                    self.action_one_hot : action_one_hot,
                    self.prev_action_encodings : self.prev_action_encodings_,
                    self.prev_hidden_states : self.prev_hidden_states_,
                    self.dropout: 0.8
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        self.prev_action_encodings_ = np.concatenate((action_encodings, action_encoding), axis=0)
        current_hidden = np.concatenate((state_c, state_h), axis=1)
        self.prev_hidden_states_ = np.concatenate((user_encodings, current_hidden), axis=0)
        
        return loss_value
    
    def reset_state(self):
        # set init state to zeros
        self.init_state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)
    
    def reset_attention(self):
        self.prev_action_encodings_ = np.zeros([1, 300], dtype=np.float32)
        self.prev_hidden_states_ = np.zeros([1, self.nb_hidden*2], dtype=np.float32)
    
    # save session to checkpoint
    def save(self, model_name):
        saver = tf.train.Saver()
        saver.save(self.sess, 'emnlp_/' + model_name + '.ckpt', global_step=0)
        print('\n:: saved to ckpt/hcn.ckpt \n')

    # restore session from checkpoint
    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('emnlp_/')
        if ckpt and ckpt.model_checkpoint_path:
            print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('\n:: <ERR> checkpoint not found! \n')

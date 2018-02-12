# from modules.entities import EntityTracker
from modules.bow import BoW_encoder
from modules.lstm_net import LSTM_net
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker
from modules.data_utils import Data
import modules.util as util

from modules.lstm_nlu import Lstm_Nlu as EntityTracker

import numpy as np
import sys


class Trainer():

    def __init__(self):
    
        # self.net = LSTM_net(obs_size=128,
        #                     action_size=59,
        #                     nb_hidden=128)
    
        et = EntityTracker()
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed()
        print('DODOO')
        at = ActionTracker(et)
        print('Shit')
        
        self.train_dataset, train_dialog_indices = Data(et, at).train_set
        self.test_dataset, test_dialog_indices = Data(et, at).test_set
        
        print('=========================\n')
        print('length of Train dialog indices : ', len(train_dialog_indices))
        print('=========================\n')

        print('=========================\n')
        print('length of Test dialog indices : ', len(test_dialog_indices))
        print('=========================\n')

        self.dialog_indices_tr = train_dialog_indices
        self.dialog_indices_dev = test_dialog_indices

        obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features
        self.action_templates = at.get_action_templates()
        action_size = at.action_size
        nb_hidden = 128

        print('=========================\n')
        print('Action_templates: ', action_size)
        print('=========================\n')

        self.net = LSTM_net(obs_size=obs_size,
                       action_size=action_size,
                       nb_hidden=nb_hidden)
        
        self.et = et
        self.at = at
        
        action_projection = []
        for action in self.action_templates:
            action_projection.append(self.emb.encode(action))
        self.action_projection = np.transpose(action_projection)
        self.action_size = action_size

    def train(self):

        print('\n:: training started\n')
        epochs = 100
        for j in range(epochs):
            # iterate through dialogs
            num_tr_examples = len(self.dialog_indices_tr)
            loss = 0.
            for i,dialog_idx in enumerate(self.dialog_indices_tr):
                # get start and end index
                start, end = dialog_idx['start'], dialog_idx['end']
                # train on dialogue
                loss += self.dialog_train(self.train_dataset[start:end])
                # print #iteration
                sys.stdout.write('\r{}.[{}/{}]'.format(j+1, i+1, num_tr_examples))

            print('\n\n:: {}.tr loss {}'.format(j+1, loss/num_tr_examples))
            # evaluate every epoch
            accuracy = self.evaluate()
            print(':: {}.dev accuracy {}\n'.format(j+1, accuracy))
        
        self.net.save()
            # if accuracy > 0.99:
            #     self.net.save()
            #     break

    def dialog_train(self, dialog):
        # create entity tracker
        et = self.et
        et.init_entities()
        # create action tracker
        at = self.at
        # reset network
        self.net.reset_state()
        self.net.reset_attention()
        
        loss = 0.
        i = 0
        pred_list = []
        # iterate through dialog
        for (u,r) in dialog:
            i += 1
            u_ent = et.extract_entities(u)
            u_ent_features = et.context_features()
            u_emb = self.emb.encode(u)
            u_bow = self.bow_enc.encode(u)
            # concat features
            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            
            
            if i ==1:
                loss += self.net.train_step(features, r, self.action_projection)
                pred_list.append(r)
            else:
                action_one_hot = np.zeros(self.action_size)
                action_one_hot[pred_list[-1]] = 1
                loss += self.net.train_step(features, r, self.action_projection, action_one_hot)
                pred_list.append(r)
        return loss / len(dialog)

    def evaluate(self):

        dialog_accuracy = 0.
        correct_dialogue_count = 0
        
        for dialog_idx in self.dialog_indices_dev:

            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.test_dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)

            # create entity tracker
            et = self.et
            et.init_entities()
            # create action tracker
            at = self.at
            # reset network
            self.net.reset_state()
            self.net.reset_attention()

            # iterate through dialog
            correct_examples = 0
            
            pred_list = []
            i = 0
            for (u,r) in dialog:
                i += 1
                if u == 'api_call no result':
                    correct_examples += 1
                    continue
                
                if r == '<UNK>':
                    correct_examples += 1
                    continue
                
                # encode utterance
                u_ent = et.extract_entities(u)
                u_ent_features = et.context_features()
                u_emb = self.emb.encode(u)
                u_bow = self.bow_enc.encode(u)
                # concat features
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                
                if i == 1:
                    prediction = self.net.forward(features, self.action_projection)
                    pred_list.append(prediction)
                else:
                    action_one_hot = np.zeros(self.action_size)
                    action_one_hot[pred_list[-1]] = 1
                    prediction = self.net.forward(features, self.action_projection, action_one_hot)
                    pred_list.append(prediction)
                    
                correct_examples += int(prediction == r)

            if correct_examples == len(dialog):
                correct_dialogue_count += 1
                
            # get dialog accuracy
            dialog_accuracy += correct_examples / len(dialog)
        
        per_response_accuracy = dialog_accuracy / num_dev_examples * 100
        per_dialogue_accuracy = correct_dialogue_count / num_dev_examples * 100
        
        print('=============================')
        print('correct dialogue count')
        print(correct_dialogue_count)
        print('=============================\n')
        
        return per_response_accuracy, per_dialogue_accuracy



if __name__ == '__main__':
    # setup trainer
    trainer = Trainer()
    # start training
    trainer.train()

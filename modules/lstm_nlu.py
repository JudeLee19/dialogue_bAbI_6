from enum import Enum
import numpy as np

# NLU Modules should be changed for bAbI-6 Slot Tagging Model.
import rnn_nlu.interact_bAbI_6 as interact


class Lstm_Nlu:
    
    def __init__(self):
        self.entities = {
            '<area>': None,
            '<food>': None,
            '<price>': None
        }
        
        self.num_features = 3
        self.rating = None
        
        self.nlu_model = interact.Interact()

        self.EntType = Enum('Entity Type', '<area> <food> <price> <non_ent>')
    
    def init_entities(self):
        self.entities = {
            '<area>': None,
            '<food>': None,
            '<price>': None
        }
    
    def ent_type(self, ent):
        if ent == 'B-area':
            return self.EntType['<area>'].name
        elif ent == 'B-food':
            return self.EntType['<food>'].name
        elif ent == 'B-price':
            return self.EntType['<price>'].name
        else:
            return None
    
    def extract_entities(self, utterance, update=True):
        tokenized = []
        word_list = utterance.split(' ')
        slot_tagging_result = self.nlu_model.inference(utterance)[0]
        
        for i, tag in enumerate(slot_tagging_result):
            entity = self.ent_type(tag)
            if update and entity:
                self.entities[entity] = word_list[i]
                tokenized.append(entity)
            elif entity:
                tokenized.append(entity)
            else:
                tokenized.append(word_list[i])

        return ' '.join(tokenized)
    
    def context_features(self):
       keys = list(set(self.entities.keys()))
       self.ctxt_features = np.array( [bool(self.entities[key]) for key in keys],
                                   dtype=np.float32 )
       
       return self.ctxt_features
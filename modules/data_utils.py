import modules.util as util


'''
    Train

    1. Prepare training examples
        1.1 Format 'utterance \t action_template_id\n'
    2. Prepare dev set
    3. Organize trainset as list of dialogues
'''


class Data():

    def __init__(self, entity_tracker, action_tracker):
        
        self.at = action_tracker
        self.action_templates = action_tracker.get_action_templates()
        self.et = entity_tracker
        
        # prepare data
        self.train_set = self.prepare_data(type='Train')
        self.test_set = self.prepare_data(type='Test')

    def prepare_data(self, type=None):
        # get dialogs from file
        if type == 'Train':
            dialogs, dialog_indices = util.read_dialogs(with_indices=True,
                                                        file_name='/root/jude/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-trn.txt', babi_num=6)
        elif type == 'Test':
            dialogs, dialog_indices = util.read_dialogs(with_indices=True,
                                                        file_name='/root/jude/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-tst.txt', babi_num=6)
        # get utterances
        utterances = util.get_utterances(dialogs)
        # get responses
        responses = util.get_responses(dialogs)
        responses = [ self.get_template_id(response) for response in responses ]

        trainset = []
        for u,r in zip(utterances, responses):
            trainset.append((u,r))

        return trainset, dialog_indices

    def get_template_id(self, response):
        
        filtered_response = self.at.filter_response(response)
        
        if filtered_response in self.action_templates:
            return self.action_templates.index(filtered_response)
        else:
            # print(filtered_response)
            return '<UNK>'

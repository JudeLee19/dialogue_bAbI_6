import json

kb_data = json.load(open('/root/jude/data/dstc2/dstc2_traindev/scripts/config/ontology_dstc2.json'))

informable_data = kb_data['informable']

area_list = informable_data['area']
food_list = informable_data['food']
price_list = informable_data['pricerange']

filter_kb_list = []
with open('/root/jude/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt', 'r') as f_r:
    for line in f_r:
        line = line.strip()
        filter_name = line.split(' ')[1]
        filter_kb_list.append(filter_name)

filter_kb_list = set(filter_kb_list)


def read_content():
    return ' '.join(get_utterances())


def read_dialogs(with_indices=False, file_name=None, babi_num=None):
    def rm_index(row):
        return [' '.join(row[0].split(' ')[1:])] + row[1:]
    
    def babi_5_filter_(dialogs):
        filtered_ = []
        for row in dialogs:
            if row[0][:6] != 'resto_':
                filtered_.append(row)
        return filtered_
    
    def babi_6_filter_(dialogs):
        filtered_ = []
        for row in dialogs:
            if row[0].split(' ')[0] not in filter_kb_list:
                filtered_.append(row)
        return filtered_
    
    with open(file_name) as f:
        if babi_num == 5:
            dialogs = babi_5_filter_([rm_index(row.split('\t')) for row in f.read().split('\n')])
        elif babi_num == 6:
            dialogs = babi_6_filter_([rm_index(row.split('\t')) for row in f.read().split('\n')])

        # organize dialogs -> dialog_indices
        prev_idx = -1
        n = 1
        dialog_indices = []
        updated_dialogs = []
        for i, dialog in enumerate(dialogs):
            if not dialogs[i][0]:
                if len(dialogs[i]) > 1:
                    updated_dialogs.append(dialog)
                    continue
                dialog_indices.append({
                    'start': prev_idx + 1,
                    'end': i - n + 1
                })
                prev_idx = i - n
                n += 1
            else:
                updated_dialogs.append(dialog)

        if with_indices:
            return updated_dialogs, dialog_indices[:-1]

        return updated_dialogs


def get_utterances(dialogs=[]):
    # if dialogs is empty then is should read from training dataset (to create vocab)
    dialogs = dialogs if len(dialogs) else read_dialogs(file_name='/root/jude/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-trn.txt', babi_num=6)
    
    utter_list = []
    for row in dialogs:
        if len(row) == 1:
            utter_list.append(row[0])
            # continue
        else:
            utter_list.append(row[0])
    return utter_list


def get_responses(dialogs=[]):
    dialogs = dialogs if len(dialogs) else read_dialogs(file_name='/root/jude/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-trn.txt', babi_num=6)
    
    response_list = []
    for row in dialogs:
        if len(row) == 1:
            response_list.append(row[0])
            # continue
        else:
            response_list.append(row[1])
    return response_list


def get_entities():

    def filter_(items):
        return sorted(list(set([ item for item in items if item and '_' not in item ])))

    with open('data/dialog-babi-kb-all.txt') as f:
        return filter_([item.split('\t')[-1] for item in f.read().split('\n') ])

import modules.util as util
import numpy as np
import json

'''
    [Action Templates]

     1. '<rest_name> is a great restaurant',
     2. '<rest_name> is a great restaurant serving <food> food and it is in the <price> price range',
     3. '<rest_name> is a great restaurant serving <price> <food> food in the <area> of town .',
     4. '<rest_name> is a nice place in the <area> of town',
     5. '<rest_name> is a nice place in the <area> of town and the prices are <price>',
     6. '<rest_name> is a nice place in the <area> of town serving tasty <food> food',
     7. '<rest_name> is a nice restaurant in the <area> of town in the <price> price range',
     8. '<rest_name> is a nice restaurant in the <area> of town serving <food> food',
     9. '<rest_name> is in the <price> price range',
     10. '<rest_name> is in the <price> price range , and their post code is <info_post_code>',
     11.'<rest_name> is on <info_address>',
     12.'<rest_name> serves <food> food',
     13.'<rest_name> serves <food> food in the <price> price range',
     14.'Can I help you with anything else?',
     15.'Did you say you are looking for a restaurant in the <area> of town?',
     16.'Hello , welcome to the Cambridge restaurant system . You can <rest_name> for restaurants by area , price range or food type . How may I help you ?',
     17.'I am sorry but there is no <food> restaurant that matches your request',
     18.'I am sorry but there is no other <food> restaurant in the <area> of town',
     19.'I am sorry but there is no other <food> restaurant in the <price> price range',
     20.'I am sorry but there is no other <food> restaurant that matches your request',
     21."I'm sorry but there is no <food> restaurant in the <area> of town",
     22."I'm sorry but there is no <food> restaurant in the <area> of town and the <price> price range",
     23."I'm sorry but there is no restaurant serving <food> food",
     24."I'm sorry but there is no restaurant serving <price> <food> food",
     25.'Let me confirm , You are looking for a restaurant and you dont care about the price range right?',
     26.'Let me confirm , You are looking for a restaurant in the <price> price range right?',
     27.'Ok , a restaurant in any part of town is that right?',
     28.'Sorry I am a bit confused ; please tell me again what you are looking for .',
     29.'Sorry but there is no other <food> restaurant in the <price> price range and the <area> of town',
     30.'Sorry but there is no other restaurant in the <price> price range and the <area> of town',
     31.'Sorry there is no <food> restaurant in the <area> of town',
     32.'Sorry there is no <food> restaurant in the <price> price range',
     33.'Sorry there is no <price> restaurant in the <area> of town serving <food> food',
     34.'Sorry would you like <food> food or you dont care',
     35.'Sorry would you like <food> or <food> food?',
     36.'Sorry would you like something in the <area> or in the <area>',
     37.'Sorry would you like something in the <price> price range or in the <price> price range',
     38.'Sorry would you like something in the <price> price range or you dont care',
     39.'Sorry would you like the <area> of town or you dont care',
     40."Sorry, I can't hear you",
     41.'Sure , <rest_name> is on <info_address>',
     42.'The phone number of <rest_name> is <info_phone>',
     43.'The post code of <rest_name> is <info_post_code>',
     44.'The price range at <rest_name> is <price> .'f,
     45.'There are restaurants . That area would you like?',
     46.'There are restaurants in the <price> price range and the <area> of town . What type of food would you like?',
     47.'There are restaurants serving <food> food . What area do you want?',
     48.'There are restaurants serving <food> food What area do you want?',
     49.'There are restaurants serving <food> in the <price> price range . What area would you like?',
     50.'What kind of food would you like?',
     51.'What part of town do you have in mind?',
     52.'Would you like something in the <price> , <price> , or <price> price range?',
     53.'You are looking for a <food> restaurant right?',
     54.'You are looking for a restaurant is that right?',
     55.'You are looking for a restaurant serving any kind of food right?',
     56.'api_call <food> <area> <price>',
     57.'api_call no result',
     58.'you are welcome'

    [1] : food
    [2] : area
    [3] : price
'''

class ActionTracker():

    def __init__(self, ent_tracker):
        # maintain an instance of EntityTracker
        self.et = ent_tracker

        kb_data = json.load(open('/root/jude/data/dstc2/dstc2_traindev/scripts/config/ontology_dstc2.json'))

        informable_data = kb_data['informable']

        area_list = informable_data['area']
        food_list = informable_data['food']
        self.price_list = informable_data['pricerange']

        food_list.append('asian_oriental')
        food_list.append('modern_european')
        food_list.append('north_american')
        food_list.append('middle_eastern')
        food_list.append('the_americas')
        food_list.append('northern_european')

        area_list.append('west')
        area_list.append('east')
        area_list.append('south')
        area_list.append('north')
        area_list.append('centre')
        
        self.area_list = area_list
        self.food_list = food_list

        self.post_code_list = [
            'C.B 2, 1 D.P',
            'C.B 4, 3 L.E',
            'C.B 2, 1 U.F'
        ]

        self.action_unk_list = [
            '<rest_name> Fen Ditton is in the <area> part of town .',
            '<rest_name> Fen Ditton is in the <price> price range',
            '<rest_name> is in the <area> , at <info_post_code>'
        ]

        filter_kb_list = []
        with open('/root/jude/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt', 'r') as f_r:
            for line in f_r:
                line = line.strip()
                filter_name = line.split(' ')[1]
                filter_kb_list.append(filter_name)

        self.filter_kb_list = set(filter_kb_list)

        self.extra_kb_list = [
            'the cow pizza kitchen and bar',
            'the_cow_pizza_kitchen_and_bar',
            'the_good_luck_chinese_food_takeaway',
            'the good luck chinese food takeaway',
            'the_river_bar_steakhouse_and_grill',
            'the river bar steakhouse and grill'
        ]
        
        # get a list of action templates
        self.action_templates = self.get_action_templates()
        self.action_size = len(self.action_templates)
        # action mask
        self.am = np.zeros([self.action_size], dtype=np.float32)
        
    def action_mask(self):
        # get context features as string of ints (0/1)
        ctxt_f = ''.join([ str(flag) for flag in self.et.context_features().astype(np.int32) ])

        def construct_mask(ctxt_f):
            indices = self.am_dict[ctxt_f]
            for index in indices:
                self.am[index-1] = 1.
            return self.am
    
        return construct_mask(ctxt_f)
    
    def filter_response(self, res):
        response = res
        response = response.strip()
        
        # if response == 'Could you please repeat that?':
        #     a = 1
        # else:
        #     a = 0

        if 'food .' in response:
            response = response.replace('food .', 'food')

        if 'food.' in response:
            response = response.replace('food.', 'food')

        splited_response = response.split(' ')

        # Filter post_code
        for post_code in self.post_code_list:
            if post_code in response:
                response = response.replace(post_code, '<info_post_code>')

        for word in splited_response:
            if '_post_code' in word:
                response = response.replace(word, '<info_post_code>')
            elif '_address' in word:
                response = response.replace(word, '<info_address>')
            elif '_phone' in word:
                response = response.replace(word, '<info_phone>')

        for rest_name in self.extra_kb_list:
            if rest_name in response:
                response = response.replace(rest_name, '<rest_name>')

        splited_response = response.split(' ')
        filtered_list = []
        for word in splited_response:
            if word in self.filter_kb_list:
                filtered_list.append('<rest_name>')
            elif word in self.food_list:
                filtered_list.append('<food>')
            elif word in self.area_list:
                filtered_list.append('<area>')
            elif word in self.price_list:
                filtered_list.append('<price>')
            else:
                filtered_list.append(word)

        response = ' '.join(filtered_list)

        if response in self.action_unk_list:
            response = '<UNK>'

        if 'api_call' in response and 'api_call no result' != response:
            response = 'api_call <food> <area> <price>'
        
        # if a == 1:
        #     print(response)
        
        return response
        
    def get_action_templates(self):
        no_set_responses = []
        for i, response in enumerate(util.get_responses()):
            res = self.filter_response(response)
            no_set_responses.append(res)
        responses = list(set(no_set_responses))
        responses.remove('<UNK>')
        
        return sorted(responses)

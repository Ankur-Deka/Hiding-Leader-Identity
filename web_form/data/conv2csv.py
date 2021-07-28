import json
import sys
import pandas as pd

file = sys.argv[1]
print('Converting file: "{}". Ensure that there is not space/newline at the end of the file'.format(file))
# file = './web_form/data/user_data_batch_1.txt'

data_str = open(file, 'r').read()
data_list = data_str.split(']\n')
data_list = [data[1:] for data in data_list]

num_questions = 53
tot_questions = 100

df_list = [] 
for i,data in enumerate(data_list):  # list of all users
    dict_list = data.split('},')     # list of questions
    processed_dict = {}
    for j in range(num_questions):
        q_string = dict_list[j]+'}'
        # print(i,j,q_string)
        q_dict = json.loads(q_string)
        if j==0:
            processed_dict['algo_stage'] = q_dict['algo_stage']
        episode = q_dict['episode']
        keys = ['leaderName', 'choice', 'confidence', 'correct', 'score', 'latency']
        for key in keys:
            processed_dict['Ep{}_{}'.format(episode, key)] = q_dict[key]
    
    observation_dict = json.loads(dict_list[tot_questions]+'}')
    processed_dict['observation'] = observation_dict['observation']

    coupon_dict = json.loads(dict_list[tot_questions+1].split(']')[0])
    processed_dict['coupon'] = coupon_dict['coupon']
    print(coupon_dict, processed_dict['coupon'])
    
    
    df = pd.DataFrame(processed_dict, index=[i])
    df_list.append(df)

final_df = pd.concat(df_list)
csv_name = file[:-4]+'.csv'
final_df.to_csv(csv_name)
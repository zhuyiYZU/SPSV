import logging
import subprocess
import time
from itertools import product
import random
import time
from extract_sim import Extract_and_update_verbalizer
from gensim.models import KeyedVectors
import os
# 配置参数
config = {
    "dataset": ["newstitle"],  #"agnewstitle","newstitle",,"snippets"
    "batch_size": [8],
    "learning_rate": ["3e-5"], #, "2e-5", "4e-5", "5e-5"
    "shot": [20],
    # "seed": range(115),
    "seed": [120], #115,101, 120
    "template_id": [0],
    "verbalizer": ["kpt"]
}

model_start_time = time.time()

# 随机生成一个字符串
this_run_unicode = str(random.randint(0, 1e10))

# 待执行的命令模板
cmd_template = (
    "python fewshot.py --result_file ./output_fewshot.txt "
    "--dataset {dataset} --template_id {template_id} --seed {seed} "
    "--batch_size {batch_size} --shot {shot} --learning_rate {learning_rate} --verbalizer {verbalizer} "
    "--train_datasets {train_datasets} --test_datasets {test_datasets} --this_run_unicode {this_run_unicode}"
)

logging.basicConfig(level=logging.INFO)

# 加载预训练的Word2Vec模型
# Word2Vec_model = KeyedVectors.load_word2vec_format('/home/star/文档/wy/lot_prompt/data/crawl-300d-2M-subword.vec')

model_end_time = time.time()
load_model_time = model_end_time - model_start_time
print('load_model_time: ', load_model_time)


# 循环执行命令
for params in product(*config.values()):
    for s in range(0,20):
        one_start_time = time.time()
        print(s)
        train_datasets = f'./datasets/TextClassification/{params[0]}/test_20/test_0.csv' if s == 0 else f'./datasets/TextClassification/{params[0]}/test_20/test_{s}.csv'
        test_datasets = f'./datasets/TextClassification/{params[0]}/test_20/test_{s+1}.csv'

        cmd = cmd_template.format(
            dataset=params[0],
            batch_size=params[1],
            learning_rate=params[2],
            shot=params[3],
            seed=params[4],
            template_id=params[5],
            verbalizer=params[6],
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            this_run_unicode=this_run_unicode
        )

        logging.info(f"Executing command: {cmd}")
        print(cmd)

        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Command executed successfully: {cmd}")

            bert_path = './model/bert_case'
            # test_path = '/home/star/文档/wy/lot_prompt/datasets/TextClassification/agnewstitle/test_0.csv'
            if params[0] == 'agnewstitle':
                data_labels = ['politics', 'sports', 'business', 'technology']

            elif params[0] == 'newstitle':
                data_labels = ['business', 'entertainment', 'health', 'sci_tech', 'sport', 'us', 'world']

            else:
                data_labels = ['business', 'computers', 'culture-arts-entertainment', 'education-science', 'engineering',
                           'healthy', 'politics-society', 'sports']


            # output_file = f'/home/star/文档/wy/lot_prompt/scripts/TextClassification/{params[0]}/lot_verbalizer.txt'
            # verbalizer_start_time = time.time()
            # verbalizer_proce = Extract_and_update_verbalizer(bert_path, Word2Vec_model, test_datasets, data_labels,
            #                                                  output_file)
            # verbalizer_proce.write_clusters_to_txt()
            # verbalizer_end_time = time.time()
            # verbalizer_all_time = verbalizer_end_time - verbalizer_start_time
            # print(f'{params[0]}_verbalizer_time: ', verbalizer_all_time)
            # one_end_time = time.time()
            # ane_all_time = one_end_time - one_start_time
            # print(f'{params[0]}_alltime: ', ane_all_time)
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {cmd}. Error: {e}")

        time.sleep(2)

    #avg
    # 从文件中读取准确率和F1 Score，并计算平均值
    accuracy_values = []
    f1_score_values = []

    with open('metrics_data.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            # 解析每一行的准确率和F1 Score
            parts = line.strip().split(': ')
            acc_f1_values = parts[0].split(', ')
            accuracy_values.append(float(acc_f1_values[0].split('=')[1]))
            f1_score_values.append(float(acc_f1_values[1].split('=')[1]))

    # 计算准确率和F1 Score的平均值
    avg_accuracy = sum(accuracy_values) / len(accuracy_values)
    avg_f1_score = sum(f1_score_values) / len(f1_score_values)

    print(f'前20轮的平均准确率为: {avg_accuracy:.4f}')
    print(f'前20轮的平均F1 Score为: {avg_f1_score:.4f}')
    with open('output_fewshot.txt', 'a') as file:
        file.write(f'前20轮的平均准确率为: {avg_accuracy:.4f}\t\t  前20轮的平均F1 Score为: {avg_f1_score:.4f}')


    with open('metrics_data.txt', 'w') as file:
        file.write('')

    all_end_time = time.time()
    all_time = all_end_time - model_start_time
    print('all_time: ', all_time)
    os.remove(f"./ckpts/{this_run_unicode}.ckpt")

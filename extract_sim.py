import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag
import nltk
import numpy as np
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

import torch
from transformers import BertTokenizer, BertForMaskedLM
from nltk.corpus import stopwords
import time
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from gensim.models import KeyedVectors
import networkx as nx
from sklearn.cluster import SpectralClustering
# 聚类
from sklearn.cluster import KMeans


class Extract_and_update_verbalizer:
    def __init__(self, bert_path, Word2Vec_model, test_path, data_labels,output_file):
        self.bert_path = bert_path
        self.Word2Vec_model = Word2Vec_model
        self.test_path = test_path
        self.data_labels = data_labels
        self.output_file = output_file

    # 读取文件并去重，将单词转换成小写
    def extract_word(self):
        noun = []
        data = pd.read_csv(self.test_path)
        # row_count = len(data)
        with open(self.test_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, text in enumerate(f):
                text = text.split(',')
                if (len(text) == 1):
                    continue
                text = text[1].replace('"', '')
                text = text.strip()
                text1 = word_tokenize(text)
                stop_eords = set(stopwords.words('english'))
                b = [w for w in text1 if not w in stop_eords]
                for word in b:
                    tokens = nltk.word_tokenize(word)
                    tags = nltk.pos_tag(tokens)
                    for pos in tags:
                        if pos[1].startswith("JJ"):  # JJ NN
                            noun.append(pos[0])
                # print('方法：nltk，开始处理数据集%s:第%d条，共有%d条' % (i, row_count))
                # break;
            print('开始去重')
            words = list(set(noun))
            unique_words = list([word.strip().lower() for word in words])
        return unique_words


    # 为单词编码
    def encode_words(self, words):
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(words)
        return encoded_labels

    def get_cluster_label(self, word_list):
        # 统计单词出现的频率
        word_freq = Counter(word_list)
        # 获取频率最高的单词
        most_common_word = word_freq.most_common(1)[0][0]
        return most_common_word
    #聚类
    def cluster_words_with_custom_labels(self, words):
        encoded_labels = self.encode_words(words)

        # 创建一个字典，将标签映射到自定义标签
        custom_label_mapping = {label: custom_label for label, custom_label in enumerate(self.data_labels)}

        # 使用自定义标签的数量初始化K均值聚类
        kmeans = KMeans(n_clusters=len(self.data_labels))

        # 使用编码标签拟合K均值模型
        kmeans.fit(encoded_labels.reshape(-1, 1))

        # 获取聚类分配
        cluster_assignments = kmeans.labels_

        # 将聚类分配映射到自定义标签
        custom_labels_assigned = [custom_label_mapping[cluster] for cluster in cluster_assignments]

        # 创建一个DataFrame以存储原始单词和它们的自定义标签分配
        clusters = pd.DataFrame({'word': words, 'custom_label': custom_labels_assigned})

        return clusters

    # 使用标签传播算法进行分类
    def cluster_words(self, center_word, words):
        G = nx.Graph()
        words.append(center_word)
        for word in words:
            G.add_node(word, weight=np.random.rand())

        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word1 = words[i]
                word2 = words[j]
                similarity = np.random.rand()
                G.add_edge(word1, word2, weight=similarity)

        clustering = SpectralClustering(n_clusters=1, affinity='precomputed', assign_labels='discretize')
        adjacency_matrix = np.array(nx.to_numpy_array(G, weight='weight'))
        labels = clustering.fit_predict(adjacency_matrix)

        clustered_words = {}
        for word, label in zip(words, labels):
            if label not in clustered_words:
                clustered_words[label] = []
            clustered_words[label].append((word, G.nodes[word]['weight']))

        center_cluster = None
        for cluster, words_and_similarities in clustered_words.items():
            if center_word in [word for word, _ in words_and_similarities]:
                center_cluster = cluster
                break

        if center_cluster is not None:
            center_cluster_words = clustered_words[center_cluster]
            sorted_words = sorted(center_cluster_words, key=lambda x: x[1], reverse=True)
            return sorted_words
        else:
            return []

    # 损失函数
    def compute_extension_word_losses(self, model, tokenizer, template, extension_words):
        # 分词化输入句子
        tokens = tokenizer.tokenize(template)
        masked_index = tokens.index("[MASK]")  # 找到[MASK]标记的位置

        extension_losses = {}

        def compute_loss(outputs, target_ids):
            # 使用logits计算损失
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, target_ids)
            return loss

        # 遍历每个扩展词汇
        for word in extension_words:
            # 复制原始句子并将[MASK]标记替换为扩展词汇
            masked_tokens = tokens.copy()
            masked_tokens[masked_index] = word

            # 将标记转换为输入ID
            input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

            # 创建输入张量
            input_tensor = torch.tensor([input_ids])

            # 创建目标张量，维度与词汇表大小相同，并将数据类型设置为Long
            target_id = tokenizer.convert_tokens_to_ids(tokens[masked_index])
            target_tensor = torch.zeros(1, len(tokenizer.get_vocab()), dtype=torch.long)
            target_tensor[0, target_id] = 1  # 将正确标签设置为1

            # 使用BERT模型计算损失
            with torch.no_grad():
                outputs = model(input_tensor)
                loss = compute_loss(outputs, target_tensor)

            # 存储扩展词汇的损失
            extension_losses[word] = loss.item()

        # 根据损失值对扩展词汇进行排序
        sorted_extension_words = sorted(extension_losses.items(), key=lambda x: x[1], reverse=True)

        return sorted_extension_words

    # mask
    def calculate_word_probabilities(self, model, tokenizer, template, extension_words):
        # 分词化模板
        tokenized_template = tokenizer.tokenize(template)

        # 找到模板中的mask位置
        mask_indices = [i for i, token in enumerate(tokenized_template) if token == '[MASK]']

        # 将模板分词转换为输入张量
        input_ids = tokenizer.convert_tokens_to_ids(tokenized_template)
        input_tensor = torch.tensor(input_ids).unsqueeze(0)  # 添加批次维度

        # 获取BERT的预测结果
        with torch.no_grad():
            predictions = model(input_tensor)

        # 获取[mask]位置上的预测概率分布
        mask_probs = predictions.logits[0, mask_indices, :]

        # 计算每个单词的概率
        word_probabilities = {}
        for word_to_fill in extension_words:
            word_id = tokenizer.convert_tokens_to_ids(word_to_fill)
            probability = torch.nn.functional.softmax(mask_probs, dim=-1)[:, word_id]
            word_probabilities[word_to_fill] = probability.item()
        sorted_extension_words = sorted(word_probabilities.items(), key=lambda x: x[1], reverse=True)

        return sorted_extension_words

    # jaccard
    def jaccard_similarity_sort(self, category_word, word_list):
        def jaccard_similarity(w1, w2):
            set1 = set(w1)
            set2 = set(w2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union != 0 else 0  # 避免除零错误

        similarity_scores = [(word, jaccard_similarity(category_word, word)) for word in word_list]

        # 按照相似度得分降序对词和得分进行排序
        sorted_words = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        return sorted_words
        # # 打印每个词和其相似度得分
        # for word, score in sorted_words:
        #     print(f"Word: {word}, Jaccard Similarity Score: {score}")

    def find_intersection_of_top_words(self, categories_results):
        # 初始化一个列表，用于存储每个类别的前五个词
        top_5_words_by_category = []

        # 提取每个类别的前五个词
        for category_results in categories_results:
            # 对排序结果按得分进行降序排序
            category_results.sort(key=lambda x: x[1], reverse=True)
            top_5_words = [word for word, score in category_results[:10]]

            # 存储每个类别的前五个词
            top_5_words_by_category.append(top_5_words)

        # 找到所有类别前五个词的并集并去重
        intersection_of_top_words = set(top_5_words_by_category[0]).union(*top_5_words_by_category)

        return intersection_of_top_words

    def read_existing_words_from_file(self, file_path, category_word):
        existing_words_line = None
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                words = line.strip().split(',')
                if words and words[0] == category_word:
                    existing_words_line = line.strip()
                    break
        return existing_words_line

    def calculate_similarity(self, category_word, extension_words):
        # 存储扩展词与类别标签词的相似度
        similarity_scores = {}

        # 遍历扩展词数组并计算相似度
        for word in extension_words:
            try:
                # 使用Gensim的similarity方法计算相似度
                similarity = self.Word2Vec_model.similarity(category_word, word)

                # 存储相似度分数
                similarity_scores[word] = similarity
            except KeyError:
                # 处理词不在词汇表中的情况
                similarity_scores[word] = 0  # 或其他适当的默认值

        # 按相似度分数降序排序结果
        sorted_results = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        # 提取前top_n个词
        top_words = [word for word, score in sorted_results]

        return top_words

    def write_clusters_to_txt(self):
        # 初始化BERT模型和分词器

        model = BertForMaskedLM.from_pretrained(self.bert_path)
        tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        words = self.extract_word()
        # 使用修改后的cluster_words函数进行聚类
        cluster_result = self.cluster_words_with_custom_labels(words)

        # 读取原有的词汇数据
        existing_words_data = {}
        with open(self.output_file, 'r', encoding='utf-8') as existing_file:
            for line in existing_file:
                words = line.strip().split(',')
                if words:
                    category = words[0]
                    existing_words_data[category] = words[1:]

        with open(self.output_file, 'w', encoding='utf-8') as file:
            for custom_label in self.data_labels:
                # 找到该自定义标签下的单词
                words_in_cluster = list(cluster_result[cluster_result['custom_label'] == custom_label]['word'].values)

                # 标签传播
                cluster_res = self.cluster_words(custom_label, words_in_cluster)
                # 损失函数
                input_sentence = f"This is a word about [MASK], which is related to {custom_label}"
                bert_loss_res = self.compute_extension_word_losses(model, tokenizer, input_sentence, words_in_cluster)
                # mask预测
                bert_mask_res = self.calculate_word_probabilities(model, tokenizer, input_sentence, words_in_cluster)
                # jaccard
                word_list = self.jaccard_similarity_sort(custom_label, words_in_cluster)

                categories_results = [bert_loss_res, bert_mask_res, cluster_res, word_list]

                intersection_of_words = self.find_intersection_of_top_words(categories_results)

                words_in_cluster = [word for word in list(intersection_of_words) if word != custom_label]
                # print(words_in_cluster)
                # 获取已存在的词汇列表（如果存在）
                existing_words = existing_words_data.get(custom_label, [])
                # print(existing_words)
                # 合并新词汇和已存在的词汇
                final_word_list = words_in_cluster + existing_words
                cosine_result = self.calculate_similarity(custom_label, final_word_list)

                # 只保留前50个词
                final_word_list = [custom_label] + cosine_result[:50]
                # print(final_word_list)
                # 将单词以逗号分隔的形式写入文件
                line = ','.join(final_word_list)
                file.write(line + '\n')
                print('写入success')

PROCESSORS = {
    'ver_pro' : Extract_and_update_verbalizer
}

#bert_path, Word2Vec_path, test_path, data_labels,output_file
if __name__ == "__main__":
    bert_path = './model/bert_case'
    # 加载预训练的Word2Vec模型
    Word2Vec_model = KeyedVectors.load_word2vec_format('/home/star/文档/wy/lot_prompt/data/crawl-300d-2M-subword.vec')
    # Word2Vec_model = ''
    test_path = '/datasets/TextClassification/agnewstitle/test_10/test_0.csv'
    data_labels = ['politics', 'sports', 'business', 'technology']
    # data_labels = ['business','computers','culture-arts-entertainment','education-science','engineering','healthy','politics-society','sports']
    output_file = '/home/star/文档/wy/lot_prompt/scripts/TextClassification/agnewstitle/lot_verbalizer.txt'

    verbalizer_proce = Extract_and_update_verbalizer(bert_path, Word2Vec_model, test_path, data_labels, output_file)
    verbalizer_proce.write_clusters_to_txt()





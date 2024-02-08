import pandas as pd

newstitle_data = pd.read_csv('test.csv', header=None, names=['label', 'title'])

# 统计每个类别的新闻标题数量
label_counts = newstitle_data['label'].value_counts()
print(label_counts)

# 计算每个类别划分为十部分后的数量
label_counts_divided = label_counts // 20
print(label_counts_divided)

# 对每个类别进行等比划分并保存为10个文件
for i in range(20):
    divided_data = pd.DataFrame(columns=['label', 'title'])

    for label, count in label_counts_divided.items():
        # 获取原始数据中该类别的所有数据
        category_data = newstitle_data[newstitle_data['label'] == label]

        # 计算每个划分的大小
        batch_size = count
        if i == 19:
            # 最后一个划分处理余数
            batch_size += len(category_data) % 20

        # 随机抽取相应数量的数据，确保不重复
        subset = category_data.sample(batch_size, replace=False)
        divided_data = pd.concat([divided_data, subset])

    # 将结果保存为新的CSV文件
    filename = f'test_{i}.csv'
    divided_data.to_csv(filename, index=False, quoting=1, header=False)

    print(f'Saved {filename}')

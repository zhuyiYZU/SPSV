import csv
count=0
with open("train.csv", encoding='utf8') as f:
    reader = csv.reader(f, delimiter=',')
    for idx, row in enumerate(reader):
        label, headline, body = row
        text_a = headline.replace('\\', ' ')
        text_b = body.replace('\\', ' ')
        count=count+1
        print("1111111111111111111111111"+str(count))
        print(text_a)
        print(text_b)
        print("1111111111111111111111111")

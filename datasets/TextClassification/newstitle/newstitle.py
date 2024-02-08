import os
import random
lines=[]
write=[]

#business
#entertainment
#health
#sci_tech
#sport
#us
#world 1,a,c

count=8
cc=0

f_co_3 = open('train.csv', 'w', encoding='utf-8')
f_co_3.close()

dict={"business":1,"entertainment":2,"health":3,"sci_tech":4,"sport":5,"us":6,"world":7}

with open("tagmynews.txt", 'r',encoding="utf-8") as infile:
    for ii, line in enumerate(infile):
        if ii%count==0 and ii!=0:
            f_co_3 = open('train.csv', 'a', encoding='utf-8')
            a = write[0].replace(",","")
            a = a.replace("\'", "")
            a = a.replace("\"", "")
            a = a.replace("\\", "")
            a = a.replace(":", "")
            a = a.replace(".", "")

            b = write[1].replace(",","")
            b = b.replace("\'", "")
            b = b.replace("\"", "")
            b = b.replace("\\", "")
            b = b.replace(":", "")
            b = b.replace(".", "")

            f_co_3.write(str(dict[write[6]])+","+a+","+b+'\n')
            f_co_3.close()
            write=[]
            line = line.replace("\n", "")
            write.append(line)
        else:
            line = line.replace("\n","")
            write.append(line)

#with open('./test.txt', 'r', encoding='utf-8') as f:
#        for ii, line in enumerate(f):

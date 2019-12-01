#-*-coding:utf-8 -*-

import csv
with open('file.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    # 读要转换的txt文件，文件每行各词间以@@@字符分隔
    with open('C:\\Users\\1701\Desktop\\20191122 104216.txt', 'rb') as filein:
        for line in filein:
            line_list = line.decode().strip().split(",")
            print(line_list[0:5])
            A=pd.DataFrame(line_list)
            print(A.head())
            A.to_csv('C:\\Users\\1701\Desktop\\file.csv')

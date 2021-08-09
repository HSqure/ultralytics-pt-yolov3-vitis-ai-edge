# coding=utf-8
#本程序用于生成适用于VOC格式的待训练的图片文件名的清单（与标注文件对应），不带后缀
import os

"""
本函数用于查找目录下所有文件名
"""
def file_name_list(file_dir):        
    output=os.listdir(file_dir)
    return output

def _main():

    list_file = open('val.txt', 'w') #生成的清单路径

    FILE_DIR='images/val' #标注文件路径

    file_list=file_name_list(FILE_DIR) #查找标注文件目录下的所有文件名并保存在file_list里

    for x in file_list:
        list_file.write(os.getcwd() + '/images/val/' + str(x))
        list_file.write('\n')


if __name__ == '__main__':
    _main()

#额外的工具
import json
#通过json文件加载超参数
def load_config(parameters:str):
    '''加载配置文件及其中的参数'''
    path = 'E:/mdx/SelfAttention_ActionRecognition/config.json'
    with open(path) as file:
        data = json.load(file)
    return data[parameters]
#list中的str转为float
def str_to_float(list):
    #将列表中的str转换为float
    new_list = []
    for i in range(len(list)):
        new_list.append(float(list[i]))
    return new_list
#删除给定区间的关键点
def section_delete(list,start,end):
    """
    :param input_data: 输入数据
    :param start: 开始区间
    :param end: 结束区间
    """
    a = list[:start]
    b = list[end:]
    return a + b

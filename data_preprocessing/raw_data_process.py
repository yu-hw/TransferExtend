import pickle
import javalang
import os

"""
    基于原始数据  dataset.pkl, 产生网络使用的数据集
         输入：dataset_pre.pkl   输出：11个错误类型的3种文件  positive.txt  negative.txt  patch.txt
             其中 positive.txt 指 代码有错误并标记正确错误的位置 的数据  ==> 用javalang删去大于200词的数据
                  negative.txt 指 代码有错误并标记不正确错误位置 的数据  ==> 先删去标记rank，再用java删去
                  positive_patch.txt 指 代码完全正确  的数据   ==>  用javalang删去大于200词的数据，注意数据量要同步
                  negative_patch.txt
                  
    ../data/dataset_pre.pkl 文件是，处理 negative标记、词长>200的 最终数据。
    
    数据集为： 【pos,neg,pat】
      变成：【pos(有标记),patch】==> label=1
           【neg(有标记),neg(没有标记)】 ==> label=0
    
    
"""


def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
def write_pkl(obj, filepath):
    f = open(filepath,"wb")
    pickle.dump(obj,f)
    f.close()
    

dataset_pkl_path = '../data/dataset.pkl'
dataset_pkl = read_pkl(dataset_pkl_path)

#一、获得token小于200词的数据集。
#     1.将negative中的rank
#     2.原始数据集不变，删去有效字符长度超过200的数据，以patch中的行数为准
faultType = dataset_pkl.keys()
dict_data = {}
for type1 in faultType:

    data_len1 = len(dataset_pkl[type1]['positive'])
    data_len2 = len(dataset_pkl[type1]['negative'])
    data_len3 = len(dataset_pkl[type1]['patch'])
    print("未删去200以上时 代码片段总量为：",data_len1,data_len2,data_len3,data_len2)

    code_list_pos = []
    code_list_neg = []
    code_list_pos_pat = []
    code_list_neg_pat = []

    for i in range(data_len1):
        #1. 将 negative中的标记都替换掉
        data_neg = dataset_pkl[type1]["negative"][i]
        data_neg_patch = (dataset_pkl[type1]["negative"][i]).replace("rank2fixstart"," ").replace("rank2fixend"," ")
        #2.  再用javalang分词，排除大于200词的 (由于三个文件数据长度不同，为了保持数据对应维度相同，以patch为准，删去数据)
        tokens = javalang.tokenizer.tokenize(dataset_pkl[type1]["patch"][i])
        token_seq = []
        for token in tokens:
            token_seq.append( str(token.value))
        # print(len(token_seq))
        if( len(token_seq)<=200 ):
            code_list_pos.append(dataset_pkl[type1]["positive"][i])  #原始有错误标记的positive
            code_list_neg.append(data_neg) # 原始有错误 fl的 negative
            code_list_pos_pat.append(dataset_pkl[type1]["patch"][i]) # 修复后的patch
            code_list_neg_pat.append(data_neg_patch) #没有标记 错误fl的 negative
    print("最后删去200以上时 代码片段总量为：",len(code_list_pos),len(code_list_neg),len(code_list_pos_pat),len(code_list_neg_pat))

    #  以上得到了具体数据 code_list,接下来组装成pkl即可。
    dict_data[type1] =  {'positive': code_list_pos,'negative':code_list_neg,'positive_patch':code_list_pos_pat,'negative_patch':code_list_neg_pat}

print(dict_data.keys())
write_pkl(dict_data,"../data/dataset_pre.pkl")


#二、制作数据样本，生成 11种错误类型的 11 个文件夹，每个文件夹有三个txt文件，代表 'positive', 'negative', 'patch'
#     数据的形式为 list[str,str,...,str]

dataset_pre_pkl_path = '../data/dataset_pre.pkl'
dataset_pre_pkl = read_pkl(dataset_pre_pkl_path)

faultType = dataset_pre_pkl.keys()
codeType = ['positive', 'negative', 'positive_patch','negative_patch']

for type1 in faultType:
    for type1_1 in codeType:
        list1 = dataset_pre_pkl[type1][type1_1]
        print( "处理前 "+type1 + '的' + type1_1 + '的长度为：',len(list1))
        file_path = '../data/'+ type1
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        file = open( file_path + '/src_token_' + type1_1 + '.txt','w',encoding='utf-8')

        for i in range(len(list1)):
            tokens1 = javalang.tokenizer.tokenize(list1[i])
            tokens1_seq = []
            for token1 in tokens1:
                tokens1_seq.append(str(token1.value))
            # print(tokens1_seq)  # 得到的tokens1_seq样式为：['private', 'void', 'handleAuthorityURL', '(', 'List', '<',...],之后用.join()方法拼接
            string11  = ' '.join(tokens1_seq)
            #如果遇到代码中包含非法字符时，用？？替代
            string111 = string11.encode('utf-8', 'replace').decode('utf-8')

            file.write(string111)
            file.write('\n')
        file.close()

        count = len(open(file_path + '/src_token_' + type1_1 + '.txt','r').readlines())
        print("处理后的" + type1 + '的' + type1_1 + "文件的长度度为：", count)




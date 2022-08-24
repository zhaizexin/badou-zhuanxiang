import  jieba
Dict = {"经常":0.1,
        "经" :0.05,
        "有" :0.1,
        "常" :0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1,
}
#有向无环图 DAG
def calc_dag(sentence):
    DAG = {} #DAG空字典，用来储存DAG有向无环图
    N = len(sentence)
    for k in range(N):
        tmplist = []
        i = k
        frag = sentence[k]
        while i < N :
            if frag in Dict:
                tmplist.append(i)
            i += 1
            frag = sentence[k:i + 1]
        if not tmplist:
            tmplist.append(k)
        DAG[k] = tmplist
    return DAG

sentence = "经常有意见分歧"
print(calc_dag(sentence))
#{0: [0, 1], 1: [1], 2: [2, 4], 3: [3, 4], 4: [4, 6], 5: [5, 6], 6: [6]}
#0: [0, 1]代表句子中的第0个字，可以单独成词，或与第一个字一起成词
#以此类推
#这个字典中实际就存储了所有可能的切分方式的信息


#将DAG中的信息解码出来，用文本展示出所有切分方式
class DAGDecode:
    #通过两个队列来实现
    def __init__(self,sentence):
       self.sentence = sentence
       self.DAG = calc_dag(sentence)
       self.length = len(sentence)
       self.unfinish_path = [[]]#保存待解码序列的队列
       self.finish_path = []#保存解码完成的序列的队列
   #对于每一个序列，检查是否需要继续解码
    #不需要继续解码的，放入解码完成队列
    #需要继续解码的，将生成的新队列，放入待解码队列
    #path行如：【“经常”，“有”，“意见”】
    def decode_next(self,path):
        path_length = len("".join(path))
        if path_length ==self.length: #已完成解码
            self.finish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_paths = []
        for candidates in candidates:
            new_paths.append(path+[self.sentence[path_length:candidates+1]])
        self.unfinish_path += new_paths #放入待解码队列
        return

    #递归调用序列解码过程
    def decode(self):
        while self.unfinish_path !=[]:
            path = self.unfinish_path.pop()#从待解码队列中取出一个序列
            self.decode_next(path)#使用该序列进行解码
sentence = "经常有意见分歧"
dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)


#coding=utf-8
import jieba,re,os,io
from gensim.models import word2vec
import logging
#jieba.load_userdict("data\\userdict.txt")

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,filename='test_01.log')
#测试原数据
filename = 'test_01.txt'
pre,ext = os.path.splitext(filename)   #输入文件分开前缀，后缀   pre=test_01   ext=.txt
corpus = pre + '_seg' + ext    #训练语料为按行分词后的文本文件    corpus=test_01_seg.txt
fin = io.open(filename,encoding='utf8').read().strip(' ').strip('\n').replace('\n\n','\n')   #strip()取出首位空格，和换行符，用\n替换\n\n
stopwords = set(io.open('stop_list.txt',encoding='utf8').read().strip('\n').split('\n'))   #读入停用词

# 第二：分词，将训练文本中的词做处理，不能包含停用词中的词，以及长度少于等于1的词，去标点，
#去掉停用词中的词，去掉长度小于等于1的词
text = ' '.join([x for x in jieba.lcut(fin) if x not in stopwords and len(x)>1 and x != '\n'])
print(text)
#去标点，将【】里面的标点全部替换为‘，’
# results = re.sub('[（）：:？“”《》，。！·、\d ]+',' ',text)
#按行分词后存为训练语料
io.open(corpus,'w+',encoding='utf8').write(text)

#3.训练模型
# 加载语料,LineSentence用于处理分行分词语料
sentences = word2vec.LineSentence(corpus)
#用来处理按文本分词语料
#sentences1 = word2vec.Text8Corpus(corpus)
# print('=--=-=-=-=-=',sentences)
# 训练skip-gram模型; 第一个参数是训练预料，min_count是小于该数的单词会被踢出，默认值为5，
# size是神经网络的隐藏层单元数，在保存的model.txt中会显示size维的向量值。默认是100。默认window=5
#训练模型就这一句话  去掉出现频率小于2的词
model = word2vec.Word2Vec(sentences, size=12,window=25,min_count=2,workers=5,sg=1,hs=1)

# 4保存模型，以便重用
# model.save("test_01.model")
# 将模型保存成文本，model.wv.save_word2vec_format()来进行模型的保存的话，会生成一个模型文件。里边存放着模型中所有词的词向量。这个文件中有多少行模型中就有多少个词向量。
# model.wv.save_word2vec_format('test_01.model.txt','test_01.vocab.txt',binary=False)
#5词向量验证
#加载训练好的模型
model = word2vec.Word2Vec.load("test_01.model")  #加载训练好的语料模型
#计算两个句子间的相似度
list1 = ['孙悟空']
list2 = ['唐僧', '西天', '取经']
print(model.most_similar(list1))

# 计算两个词的相似度/相关程度
# role1 = ['大圣','悟空','齐天大圣','师兄','老孙','行者','孙行者','孙悟空']
# role2 = ['天蓬','猪悟能','老猪','八戒','猪八戒','呆子']
# role1 = ['天地','万物','一元']
# role2 = ['天地','百岁']
# pairs = [(x,y) for x in role1 for y in role2]
# print(pairs)  #[('天地', '天地'), ('天地', '百岁'), ('万物', '天地'), ('万物', '百岁'), ('一元', '天地'), ('一元', '百岁')]
#
# #pairs = [('观音','猪悟能'),('观音','天蓬'),('观音','八戒'),('呆子','八戒'),('天蓬','嫦娥'),('天蓬','大圣'),('天蓬','卷帘'),('八戒','姐姐')]
# for pair in pairs:
#     print("> [%s]和[%s]的相似度为：" % (pair[0],pair[1]), model.similarity(pair[0], pair[1]))   # 预测相似性
#
# # 计算某个词的相关词列表
# figures = ['如来','西天','观音','老君','师父','老孙','八戒','沙和尚','南天门','王母','天王']
# for figure in figures:
#     print("> 和[%s]最相关的词有：\n" % figure, '\n'.join([x[0].ljust(4,'　')+str(x[1]) for x in model.most_similar(figure, topn=10)


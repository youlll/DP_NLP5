import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import jieba
from tqdm import tqdm, trange
from gensim.models import Word2Vec
import numpy as np

class seq_net(nn.Module):
    def __init__(self, onehot_num):
        super(seq_net, self).__init__()
        onehot_size = onehot_num
        embedding_size = 256
        n_layer = 2
        # input [seq_length, batch, embedding_size] and output [seq_length, batch, embedding_size]
        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layer, batch_first=True)
        self.encode =torch.nn.Sequential(
            nn.Linear(onehot_size, embedding_size),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        # self.decode =torch.nn.Sequential(
        #     nn.Linear(embedding_size, 1024),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.ReLU(),
        #     nn.Linear(1024, 2048),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.ReLU(),
        #     nn.Linear(2048, onehot_size),
        #     torch.nn.Softmax()
        # )
        self.decode =torch.nn.Sequential(
            nn.Linear(embedding_size, onehot_size),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

    def forward(self, x):
        # input [seq_length, onehot_size]
        em = self.encode(x).unsqueeze(dim=1)
        # [seq_length, 1, onehot_size]
        out, (h, c) = self.lstm(em)
        res = 2*(self.decode(out[:,0,:])-0.5)
        return res


def read_data():
    datasets_root = "datasets"
    catalog = "inf.txt"
    with open(os.path.join(datasets_root, catalog), "r",encoding='ANSI') as f:
        all_files = f.readline().split(",")
        print(all_files)

    all_texts = dict()
    for name in all_files:
        with open(os.path.join(datasets_root, name+".txt"), "r",encoding='ANSI') as f:
            file_read = f.readlines()
            all_text = ""
            for line in file_read:
                line = re.sub('\s','', line)
                line = re.sub('！','。', line)
                line = re.sub('？','。', line)
                # u3002是句号
                line = re.sub('[\u0000-\u3001]','', line)
                line = re.sub('[\u3003-\u4DFF]','', line)
                line = re.sub('[\u9FA6-\uFFFF]','', line)
                all_text += line
            all_texts[name] = all_text

    return all_texts

def read_test_data():
    name = "datasets/test_data.txt"
    with open(name, "r",encoding='ANSI') as f:
        file_read = f.readlines()
        all_text = ""
        for line in file_read:
            line = re.sub('\s','', line)
            line = re.sub('！','。', line)
            line = re.sub('？','。', line)
            # u3002是句号
            line = re.sub('[\u0000-\u3001]','', line)
            line = re.sub('[\u3003-\u4DFF]','', line)
            line = re.sub('[\u9FA6-\uFFFF]','', line)
            all_text += line

    return all_text

def train():
    embed_size = 1024
    epochs = 50
    end_num = 10

    print("start read data")
    all_texts = read_data()
    all_terms = dict()

    for name in list(all_texts.keys()):
        text = all_texts[name]
        text_terms = list()
        for text_line in text.split('。'):
            seg_list = list(jieba.cut(text_line, cut_all=False)) # 使用精确模式
            if len(seg_list) < 5:
                continue
            seg_list.append("END")
            # add list
            text_terms.append(seg_list)

        all_terms[name] = text_terms
    print("end read data")

    # get word2vec
    print("start calculate embedding vector")
    text_models = dict()
    for name in list(all_terms.keys()):
        if not os.path.exists('vec_models/'+name+'.model'):
            print("Start to build ", name, " model")

            # sg=0 cbow, sg=1 skip-gram
            # size is the dim of feature
            model = Word2Vec(sentences=all_terms[name], sg=0, vector_size=embed_size, min_count=1, window=10, epochs=10)

            print("Finish to build ", name, " model")
            text_models[name] = model
            model.save('vec_models/'+name+'.model')
    print("finish calculate embedding vector")


    for name in list(all_terms.keys()):
        print("start train ", name)
        sequences = all_terms[name]
        vec_model = Word2Vec.load('vec_models/'+name+'.model')

        model = seq_net(embed_size).cuda()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
        for epoch_id in range(epochs):
            for idx in trange(0, len(sequences)//end_num - 1):
                seq = []
                for k in range(end_num):
                    seq += sequences[idx+k]

                target = []
                for k in range(end_num):
                    target += sequences[idx+end_num+k]

                input_seq = torch.zeros(len(seq), embed_size)
                for k in range(len(seq)):
                    input_seq[k] = torch.tensor(vec_model.wv[seq[k]])

                target_seq = torch.zeros(len(target), embed_size)
                for k in range(len(target)):
                    target_seq[k] = torch.tensor(vec_model.wv[target[k]])

                all_seq = torch.cat((input_seq, target_seq), dim=0)

                optimizer.zero_grad()
                out_res = model(all_seq[:-1].cuda())

                f1 = ((out_res[-target_seq.shape[0]:]**2).sum(dim=1))**0.5
                f2 = ((target_seq.cuda()**2).sum(dim=1))**0.5
                loss = (1-(out_res[-target_seq.shape[0]:] * target_seq.cuda()).sum(dim=1)/f1/f2).mean()
                loss.backward()
                optimizer.step()
                if idx % 50 == 0:
                    print("loss: ", loss.item(), " in epoch ", epoch_id, " res: ", out_res[-target_seq.shape[0]:].max(dim=1).indices, target_seq.max(dim=1).indices)

            state = {"models" : model.state_dict()}
            torch.save(state, "models/"+name+str(epoch_id)+".pth")


def test():
    embed_size = 1024
    epochs = 50
    end_num = 10

    text_name = "射雕英雄传"
    print("start read test data")
    text = read_test_data()
    text_terms = list()
    for text_line in text.split('。'):
        seg_list = list(jieba.cut(text_line, cut_all=False)) # 使用精确模式
        if len(seg_list) < 5:
            continue
        seg_list.append("END")
        # add list
        text_terms.append(seg_list)
    print("end read data")

    checkpoint = torch.load("models/"+"射雕英雄传"+str(49)+".pth")

    model = seq_net(embed_size).eval().cuda()
    model.load_state_dict(checkpoint["models"])
    vec_model = Word2Vec.load('vec_models/'+text_name+'.model')

    seqs = []
    for sequence in text_terms:
        seqs += sequence

    input_seq = torch.zeros(len(seqs), embed_size).cuda()
    result = ""
    with torch.no_grad():
        for k in range(len(seqs)):
            input_seq[k] = torch.tensor(vec_model.wv[seqs[k]])
        end_num = 0
        length = 0
        while end_num < 10 and length < 100:
            print("length: ", length)
            out_res = model(input_seq.cuda())[-2:-1]
            # key_value = vec_model.wv.most_similar(positive=np.random.rand(1,256), topn=1)
            key_value = vec_model.wv.most_similar(positive=np.array(out_res.cpu()), topn=15)
            key = key_value[np.random.randint(15)][0]
            if  key == "END":
                result += "。\n"
                end_num += 1
            else:
                result += key
            length += 1
            input_seq = torch.cat((input_seq, out_res), dim=0)

    # print(result)
    with open('result.txt', 'wb') as f:
        f.write(bytes(result.encode('utf-8')))



if __name__ == "__main__":
    # execute only if run as a script
    train()
    test()
# coding=utf-8
import time
import torch
# import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from predict import*
import os
BATCH_SIZE = 1
maxLength = 512
PATH = os.getcwd() + "\\model\\model_512_20200722.pkl"
# PATH = os.getcwd()+"/model/model_512_20200722.pkl"
PRETRAINED_MODEL_NAME = "hfl/chinese-bert-wwm"  # 指定繁簡中文 BERT-BASE 預訓練模型



# print("PyTorch 版本：", torch.__version__)
# vocab = tokenizer.vocab
# print("字典大小：", len(vocab))


class NLPDataset():
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self):
        # 大數據你會需要用 iterator=True
        self.len = 1
        self.articleLegth = maxLength
        self.model, self.tokenizer = self.load_model(PATH)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)

    def textProcess(self, text):
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        cutWordNum = 100
        word_pieces = ["[CLS]"]
        tokens = []
        I = 0
        while (len(tokens) < self.articleLegth - 1):
            if (tokens == []):
                tokens = self.tokenizer.tokenize(text[cutWordNum * I:cutWordNum * (I + 1)])
            else:
                if (len(text) > 100 * (I + 1)):
                    tokens = tokens + self.tokenizer.tokenize(text[cutWordNum * I:cutWordNum * (I + 1)])
                else:
                    tokens = tokens + self.tokenizer.tokenize(text[cutWordNum * I:])
                    break
            I += 1

        if (len(tokens) > self.articleLegth - 1):
            tokens = tokens[:self.articleLegth - 1]
        lens = self.articleLegth - len(tokens) - 1

        if (lens == 0):
            pass
        else:
            for I in range(lens):
                # padding 文章長度不足的部分
                tokens += ["[PAD]"]
        word_pieces += tokens
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segments_tensor = torch.tensor([1] * len(word_pieces), dtype=torch.long)
        tokens_tensors = torch.reshape(tokens_tensor, (1, self.articleLegth))
        segments_tensors = torch.reshape(segments_tensor, (1, self.articleLegth))
        masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

        return (tokens_tensors, segments_tensors, masks_tensors)

    def load_model(self, PATH):
        model = torch.load(PATH)
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        return model, tokenizer

    def __len__(self):
        return self.len

    def modelPredictions(self, text:str):
        predictions = None

        print("start model predit!! ", "\n")
        with torch.no_grad():
            tokens_tensors, segments_tensors, masks_tensors, = self.textProcess(text)
            tokens_tensors = tokens_tensors.cuda()
            segments_tensors = segments_tensors.cuda()
            masks_tensors = masks_tensors.cuda()
            outputs = self.model(input_ids = tokens_tensors,
                         token_type_ids = segments_tensors,
                         attention_mask = masks_tensors)

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            print(pred)
        return pred.cpu().tolist()

            # if predictions is None:
            #     predictions = pred
            # else:
            #     predictions = torch.cat((predictions, pred))
            # return predictions.cpu().tolist()

# def AML_PERSON(line):
#     nameListOutPut = list()
#     result = model.evaluate_line(sess, input_from_line(line, FLAGS.max_seq_len, tag_to_id), id_to_tag)
#     for I in range(len(result['entities'])):
#         name = result['entities'][I]["word"]
#         if name not in nameListOutPut:
#             name.encode('utf-8')
#             nameListOutPut.append(name)
#             str(nameListOutPut).replace(" ", "")
#     return nameListOutPut


if __name__ == '__main__':
    test = NLPDataset()
    text = "前桃園縣副縣長葉世文涉嫌不實請領特別費及縣政業務費共新台幣9萬8679元，遭監察院彈劾，移送公務員懲戒委員會。公懲會審理後，判決葉世文撤職並停止任用2年。監察院指出，葉世文在桃園縣副縣長任內，明知特別費及縣政業務費需以公務上實際消費單據請領，卻提供不實支出憑證及載明不實餐敘與贈禮對象等用途核銷便箋，再由桃園縣政府秘書處據以代為辦理核銷特別費4萬8030元及縣政業務費5萬649元，共計9萬8679元，損害縣政府對於經費核銷的正確性。監察院認為，葉世文有違公務員服務法規定，事證明確，情節重大，依憲法規定提案彈劾，並移送公懲會審理。公懲會判決葉世文撤職並停止任用2年。"
    person = list()
    AML = test.modelPredictions(text)
    print(AML)

    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # config = load_config(FLAGS.config_file)
    # logger = get_logger(FLAGS.log_file)
    # with open(FLAGS.map_file, "rb") as f:
    #     tag_to_id, id_to_tag = pickle.load(f)
    # with tf.Session(config=tf_config) as sess:
    #     model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)



    # tEnd = time.time()  # 計時結束
    # print("AML推論時間 : ", tEnd - tStart)
# coding=utf-8
import time
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from predict import*
import os

BATCH_SIZE = 1
maxLength = 512
PATH = os.getcwd() + "\\model\\model_512_best_loss_vs_test_f1.pkl"
# PATH = os.getcwd()+"/model/model_512_best_loss_vs_test_f1.pkl"
PRETRAINED_MODEL_NAME = "hfl/chinese-bert-wwm"  # 指定繁簡中文 BERT-BASE 預訓練模型
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)


# print("PyTorch 版本：", torch.__version__)
# vocab = tokenizer.vocab
# print("字典大小：", len(vocab))


class NLPDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, tokenizer, maxLength, text):
        # 大數據你會需要用 iterator=True
        self.len = 1
        self.text = text
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        self.articleLegth = maxLength


    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        label_tensor = None
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens = self.tokenizer.tokenize(self.text)
        lens = self.articleLegth - len(tokens) - 1
        if (lens==0):
            word_pieces += tokens[:-1]
        elif(lens<0):
            word_pieces += tokens[:self.articleLegth-1]
        else:
            for I in range(lens):
                # padding 文章長度不足的部分
                tokens += ["[PAD]"]
            word_pieces += tokens
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segments_tensor = torch.tensor([1] * len(word_pieces),
                                       dtype=torch.long)
        print("tokens_tensor.shape: ", tokens_tensor.shape)
        print("segments_tensors.shape: ", segments_tensor.shape)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
        # zero pad 到同一序列長度

    tokens_tensors = pad_sequence(tokens_tensors,
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,
                                    batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape,
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def load_model(PATH):
    model = torch.load(PATH)
    return model

#
# def AML_PERSON(line):
#     nameListOutPut = list()
#     result = model.evaluate_line(sess, input_from_line(line, FLAGS.max_seq_len, tag_to_id), id_to_tag)
#     for I in range(len(result['entities'])):
#         name = result['entities'][I]["word"]
#         if name not in nameListOutPut:
#             name.encode('utf-8')
#             nameListOutPut.append(name)
#     return nameListOutPut


def modelPredictions(text):
    predictions = None
    dataloader = mainDataProcess(BATCH_SIZE, maxLength, text)
    model = load_model(PATH)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    print("start model predit!! ", "\n")
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            #             別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            #             且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            print("tokens_tensors.shape: ", tokens_tensors.shape)
            print("segments_tensors.shape: ", segments_tensors.shape)
            print("masks_tensors.shape: ", masks_tensors.shape)
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

            return predictions.cpu().tolist()


def mainDataProcess(BATCH_SIZE, maxLength, text):
    dataSet = NLPDataset(tokenizer=tokenizer, maxLength=maxLength, text=text)
    loader = DataLoader(dataSet, batch_size=BATCH_SIZE,
                        collate_fn=create_mini_batch)
    return loader


def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not.
        And then convert your prediction to list type or raise error.

    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')





if __name__ == '__main__':
    person = list()
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # config = load_config(FLAGS.config_file)
    # logger = get_logger(FLAGS.log_file)
    #
    # with open(FLAGS.map_file, "rb") as f:
    #     tag_to_id, id_to_tag = pickle.load(f)
    # with tf.Session(config=tf_config) as sess:
    #     model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)

    text = "民進黨籍新北市議員陳科名利用擔任市議會三審召集人之便，向新北市政府工務、城鄉等局處「關切」，藉以入股土方清運業者，還向建商索取高額清運土方、承攬玄關門工程費來收取回扣，新北地檢署認陳科名所為違背為人民服務的「廉潔性」、「不可賄賂性」，11日依貪汙罪嫌將他起訴並請沒入1717餘萬元犯罪所得。土方清運及製造玄關門業者蔡怡辰在2012年間邀陳科名入股，藉以獲得建商青睞承包工程，而陳入股連振勝公司後，即和蔡怡辰議定若以議員身分取得建商發包工程，可獲每立方米20至50元、玄關門工程3％工程總價做為「付出勞務」代價。檢調發現，陳科名與蔡怡辰達成協議後，即和多家建商取得清運土方、承做玄關門的契約，隨即轉向工務、城鄉等局處股長以上主管表達關切、催辦進度，加速辦理都審作業及建使照核發進度，市府局處也因陳的關切「加速」核發程序，惟檢調並未發現市府官員有圖利或收賄之情。檢調發現，陳科名「關切」的建案包括板橋江翠北側重劃區數個建案，以及三重、中和甚至中壢地區建案，部分建商甚至還是上市公司。建商雖明知陳科名指定業者的乾、溼土清運價碼，每立方米遠比行規多出30元，但為求讓建使照快速過關，仍委由陳科名指定業者處理。陳科名到案後辯稱只是單純「選民服務」，並未涉及不法，但檢調認為陳科名承攬建商土方清運等工程後，透過關切方式加速取得使建照，甚至還藉此取得佣金，已經跨越了「選民服務」界線。新北地檢署援引最高院判決意旨指出，民意代表受託遊說應屬義務性的免費「選民服務」，涉及財產利益要求期約或接受就算貪汙，陳科名以議員之名收賄2次、不正利益12次，除依貪汙罪嫌起訴並應沒入1717餘萬元不法所得；蔡怡辰到案後坦承犯行並轉汙點證人，檢察官也請免刑或減輕其刑。"
    tStart = time.time()
    AML = modelPredictions(text)
    tEnd = time.time()  # 計時結束
    print("AML推論時間 : ", tEnd - tStart)

    if 1 in AML:
        tStart = time.time()
        # person = AML_PERSON(text)
        tEnd = time.time()  # 計時結束
        person = _check_datatype_to_list(person)
        print(person)

    else:
        person = _check_datatype_to_list(person)
        print(person)
        # print(AML)
import os
import pandas as pd
import torch
import modules.utils as utils
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from itertools import chain
import sys


def make_combined_data():
    """
    Make & Save new train data that combines train data and validation data.
    """
    DATA_DIR = os.path.join(os.path.abspath(os.getcwd()), 'data')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train/train.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'val/val.csv'))
    all_data = [train, val]
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    combined_data.to_csv(os.path.join(DATA_DIR, 'train/train_all.csv'), index=False)
    
    
class CustomDataset(Dataset):
    def __init__(self, model_name, data_dir, data_type, mode):
        self.model_name = model_name
        self.mode = mode
        self.data_dir = data_dir
        self.data_type = data_type
        self.intents = {'AS_날짜_요청': 0,'AS_날짜_질문': 1,'AS_방법_요청': 2,'AS_방법_질문': 3,'AS_비용_요청': 4,'AS_비용_질문': 5,
                        'AS_시간_질문': 6,'AS_일반_질문': 7,'결제_방식_질문': 8,'결제_수단_질문': 9,'결제_시기_질문': 10,'결제_영수증_질문': 11,
                        '결제_오류_질문': 12,'결제_일반_질문': 13,'결제_일반_확인': 14,'결제_재결제_질문': 15,'결제_추가_질문': 16,
                        '결제_취소_질문': 17,'결제_할인_질문': 18,'교환|반품|환불_방법_요청': 19,'교환|반품|환불_방법_질문': 20,
                        '교환|반품|환불_방법_확인': 21,'교환|반품|환불_비용_질문': 22,'교환|반품|환불_시간_요청': 23,
                        '교환|반품|환불_시간_질문': 24,'교환|반품|환불_일반_요청': 25,'교환|반품|환불_일반_질문': 26,
                        '교환|반품|환불_일반_확인': 27,'구매_예약_요청': 28,'구매_예약_질문': 29,'구매_제품_요청': 30,'구매_제품_질문': 31,
                        '매장_이용_요청': 32,'매장_이용_질문': 33,'매장_정보_질문': 34,'멤버십_사용_질문': 35,'멤버십_적립_질문': 36,
                        '배송_날짜_요청': 37,'배송_날짜_질문': 38,'배송_날짜_확인': 39,'배송_방법_요청': 40,'배송_방법_질문': 41,
                        '배송_방법_확인': 42,'배송_비용_질문': 43,'배송_오류_질문': 44,'배송_오류_확인': 45,'배송_일반_요청': 46,
                        '배송_일반_질문': 47,'배송_일반_확인': 48,'배송_지역_요청': 49,'배송_지역_질문': 50,'배송_택배사_질문': 51,
                        '부가서비스_날짜_요청': 52,'부가서비스_날짜_질문': 53,'부가서비스_방법_요청': 54,'부가서비스_방법_질문': 55,
                        '부가서비스_비용_요청': 56,'부가서비스_비용_질문': 57,'웹사이트_사용_질문': 58,'웹사이트_오류_질문': 59,
                        '제품_가격_비교': 60,'제품_가격_요청': 61,'제품_가격_질문': 62,'제품_가격_확인': 63,'제품_구성_요청': 64,
                        '제품_구성_질문': 65,'제품_구성_확인': 66,'제품_날짜_질문': 67,'제품_방법_요청': 68,'제품_방법_질문': 69,
                        '제품_방법_확인': 70,'제품_불량_요청': 71,'제품_불량_질문': 72,'제품_불량_확인': 73,'제품_소재_질문': 74,
                        '제품_시용_요청': 75,'제품_시용_질문': 76,'제품_용도_질문': 77,'제품_용도_확인': 78,'제품_원산지_질문': 79,
                        '제품_일반_비교': 80,'제품_일반_요청': 81,'제품_일반_질문': 82,'제품_일반_확인': 83,'제품_입고_요청': 84,
                        '제품_입고_질문': 85,'제품_재고_요청': 86,'제품_재고_질문': 87,'제품_재고_확인': 88,'제품_정보_비교': 89,
                        '제품_정보_요청': 90,'제품_정보_질문': 91,'제품_정보_확인': 92,'제품_추천_비교': 93,'제품_추천_요청': 94,
                        '제품_추천_질문': 95,'제품_추천_확인': 96,'제품_커스텀_요청': 97,'제품_커스텀_질문': 98,'제품_품질_비교': 99,
                        '제품_품질_요청': 100,'제품_품질_질문': 101,'제품_품질_확인': 102,'제품_호환_질문': 103,'제품_호환_확인': 104,
                        '포장_방식_요청': 105,'포장_방식_질문': 106,'포장_비용_질문': 107,'포장_일반_질문': 108,'행사_기간_질문': 109,
                        '행사_기간_확인': 110,'행사_날짜_질문': 111,'행사_유형_질문': 112,'행사_유형_확인': 113,'행사_일반_질문': 114,
                        '행사_일반_확인': 115,'행사_정보_요청': 116,'행사_정보_질문': 117}
        self.num_labels = len(self.intents)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load data
        self.inputs, self.labels = self.data_loader(data_dir)


    def data_loader(self, path):
        print('Loading ' + self.mode + ' dataset..')
        
        # check if preprocessed data directory exists
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()
            
        if self.data_type == 'all' and not os.path.isfile(os.path.join(path, self.mode, 'train_all.csv')):
            make_combined_data()
            
        if '/' in self.model_name:
            dev_name = self.model_name.split('/')[0]
            if dev_name == 'kykim':
                base_name = ('_' + self.model_name.split('/')[1].split('-')[0]) # bert, funnel, electra...
                if self.data_type == 'all':
                    base_name += '_all'
                dev_name += base_name
            
        if os.path.isfile(os.path.join(path, self.mode, f'{self.mode}_{dev_name}_X.pt')):
            print(f'dataset named <{self.mode}_{dev_name}> already exists')
            inputs = torch.load(os.path.join(path, self.mode, f'{self.mode}_{dev_name}_X.pt'))
            labels = torch.load(os.path.join(path, self.mode, f'{self.mode}_{dev_name}_Y.pt'))
            
        else:
            print('there is no available dataset. start preprocessing...')
            if self.data_type == 'all':
                file_path = os.path.join(path, self.mode, self.mode + '_all.csv')
            else:
                file_path = os.path.join(path, self.mode, self.mode + '.csv')
            df = utils.load_csv(file_path)
            df = df.dropna(axis=0, how='all')
            inputs = df[df.columns[2:]]
            labels = df['intent']
            print(len(labels))

            # Preprocessing
            inputs, labels = self.preprocessing(inputs, labels)
            
            # Save data
            print(f'save the dataset from {self.model_name}')
            torch.save(inputs ,os.path.join(path, self.mode, f'{self.mode}_{dev_name}_X.pt'))
            torch.save(labels ,os.path.join(path, self.mode, f'{self.mode}_{dev_name}_Y.pt'))
                
        return inputs, labels

    def pad(self, data, pad_id, max_len):
        # 기존 data에서 padding 부분을 빼고,
        # max_encoding_len 길이만큼 채우는 padding으로 교체
        padded_data = []
        for x in data:
            if max_len - len(x) > 0:
                padded_data.append(torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
            else:
                padded_data.append(x)
        return list(padded_data)

    def preprocessing(self, inputs, labels):
        print('Preprocessing ' + self.mode + ' dataset..')
        #Encoding original
        src_tensor = []
        seg_tensor = []
        for i in range(len(inputs)):
            src_tensor.append(torch.tensor(list(chain.from_iterable([self.tokenizer.encode(inputs[col][i], add_special_tokens=True) \
                                                                     for col in inputs.columns if inputs[col][i] == inputs[col][i]]))))
            clss = torch.cat([torch.where(src_tensor[i] == 2)[0], torch.tensor([len(src_tensor[i])])])
            seg_tensor.append(torch.tensor(list(chain.from_iterable( \
                [[0] * (clss[i + 1] - clss[i]).item() if i % 2 == 0 else [1] * (clss[i + 1] - clss[i]).item() \
                for i, val in enumerate(clss[:-1])]))))

        #Padding
        max_encoding_len = max(list(map(lambda x: len(x), src_tensor)))
        assert max_encoding_len < 512, 'Encoding length is longer than maximum processing length.'
        src_tensor = self.pad(src_tensor, 0, max_encoding_len)
        seg_tensor = self.pad(seg_tensor, 0, max_encoding_len)

        #Convert to list of tensor to 2d tensor
        src_tensor = torch.stack(src_tensor, dim=0)
        seg_tensor = torch.stack(seg_tensor, dim=0)
        mask_tensor = (~ (src_tensor == 0)).long()

        #Encoding labels
        label_tensor = torch.tensor(self.label_encoder(labels.values))


        #Integrate the tensor {1st dimension : {src, seg, mask}, 2nd dim : {number of samples}, 3rd dim : {encoding dimension}}
        input_tensor = torch.cat([src_tensor.unsqueeze(dim=1) , seg_tensor.unsqueeze(dim=1), mask_tensor.unsqueeze(dim=1)], dim=1)

        return input_tensor, label_tensor

    def label_encoder(self, labels):
        try:
            labels = list(map(lambda x : self.intents[x], labels))
            return labels
        except:
            assert 'Invalid intent'

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, :, :], self.labels[index]


class TestDataset(Dataset):
    def __init__(self, model_name, data_dir, mode):
        self.model_name = model_name
        self.mode = mode
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.intents = {0:'AS_날짜_요청' ,1:'AS_날짜_질문' ,2:'AS_방법_요청' ,3:'AS_방법_질문' ,4:'AS_비용_요청' ,5:'AS_비용_질문' ,
                        6:'AS_시간_질문' ,7:'AS_일반_질문' ,8:'결제_방식_질문' ,9:'결제_수단_질문' ,10:'결제_시기_질문' ,11:'결제_영수증_질문' ,
                        12:'결제_오류_질문' ,13:'결제_일반_질문' ,14:'결제_일반_확인' ,15:'결제_재결제_질문' ,16:'결제_추가_질문' ,
                        17:'결제_취소_질문' ,18:'결제_할인_질문' ,19:'교환|반품|환불_방법_요청' ,20:'교환|반품|환불_방법_질문' ,
                        21:'교환|반품|환불_방법_확인' ,22:'교환|반품|환불_비용_질문' ,23:'교환|반품|환불_시간_요청' ,
                        24:'교환|반품|환불_시간_질문' ,25:'교환|반품|환불_일반_요청' ,26:'교환|반품|환불_일반_질문' ,
                        27:'교환|반품|환불_일반_확인' ,28:'구매_예약_요청' ,29:'구매_예약_질문' ,30:'구매_제품_요청' ,31:'구매_제품_질문' ,
                        32:'매장_이용_요청' ,33:'매장_이용_질문' ,34:'매장_정보_질문' ,35:'멤버십_사용_질문' ,36:'멤버십_적립_질문' ,
                        37:'배송_날짜_요청' ,38:'배송_날짜_질문' ,39:'배송_날짜_확인' ,40:'배송_방법_요청' ,41:'배송_방법_질문' ,
                        42:'배송_방법_확인' ,43:'배송_비용_질문' ,44:'배송_오류_질문' ,45:'배송_오류_확인' ,46:'배송_일반_요청' ,
                        47:'배송_일반_질문' ,48:'배송_일반_확인' ,49:'배송_지역_요청' ,50:'배송_지역_질문' ,51:'배송_택배사_질문' ,
                        52:'부가서비스_날짜_요청' ,53:'부가서비스_날짜_질문' ,54:'부가서비스_방법_요청' ,55:'부가서비스_방법_질문' ,
                        56:'부가서비스_비용_요청' ,57:'부가서비스_비용_질문' ,58:'웹사이트_사용_질문' ,59:'웹사이트_오류_질문' ,
                        60:'제품_가격_비교' ,61:'제품_가격_요청' ,62:'제품_가격_질문' ,63:'제품_가격_확인' ,64:'제품_구성_요청' ,
                        65:'제품_구성_질문' ,66:'제품_구성_확인' ,67:'제품_날짜_질문' ,68:'제품_방법_요청' ,69:'제품_방법_질문' ,
                        70:'제품_방법_확인' ,71:'제품_불량_요청' ,72:'제품_불량_질문' ,73:'제품_불량_확인' ,74:'제품_소재_질문' ,
                        75:'제품_시용_요청' ,76:'제품_시용_질문' ,77:'제품_용도_질문' ,78:'제품_용도_확인' ,79:'제품_원산지_질문' ,
                        80:'제품_일반_비교' ,81:'제품_일반_요청' ,82:'제품_일반_질문' ,83:'제품_일반_확인' ,84:'제품_입고_요청' ,
                        85:'제품_입고_질문' ,86:'제품_재고_요청' ,87:'제품_재고_질문' ,88:'제품_재고_확인' ,89:'제품_정보_비교' ,
                        90:'제품_정보_요청' ,91:'제품_정보_질문' ,92:'제품_정보_확인' ,93:'제품_추천_비교' ,94:'제품_추천_요청' ,
                        95:'제품_추천_질문' ,96:'제품_추천_확인' ,97:'제품_커스텀_요청' ,98:'제품_커스텀_질문' ,99:'제품_품질_비교' ,
                        100:'제품_품질_요청' ,101:'제품_품질_질문' ,102:'제품_품질_확인' ,103:'제품_호환_질문' ,104:'제품_호환_확인' ,
                        105:'포장_방식_요청' ,106:'포장_방식_질문' ,107:'포장_비용_질문' ,108:'포장_일반_질문' ,109:'행사_기간_질문' ,
                        110:'행사_기간_확인' ,111:'행사_날짜_질문' ,112:'행사_유형_질문' ,113:'행사_유형_확인' ,114:'행사_일반_질문' ,
                        115:'행사_일반_확인' ,116:'행사_정보_요청' ,117:'행사_정보_질문' }
        
        # Load data
        self.inputs = self.data_loader(data_dir)
        self.conv_num = utils.load_csv(os.path.join(data_dir,'test','test.csv'))['conv_num']

    def data_loader(self, path):
        print('Loading ' + self.mode + ' dataset..')
        # check if preprocessed data directory exists
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()

        if '/' in self.model_name:
            dev_name = self.model_name.split('/')[0]
            if dev_name == 'kykim':
                base_name = ('_' + self.model_name.split('/')[1].split('-')[0]) # bert, funnel, electra...
                dev_name += base_name
                                        
        if os.path.isfile(os.path.join(path, self.mode, f'{self.mode}_{dev_name}_X.pt')):
            print(f'dataset named <{self.mode}_{dev_name}> already exists')
            inputs = torch.load(os.path.join(path, self.mode, f'{self.mode}_{dev_name}_X.pt'))

        else:
            print('there is no available dataset. start preprocessing...')
            file_path = os.path.join(path, self.mode, self.mode + '.csv')
            df = utils.load_csv(file_path)
            df = df.dropna(axis=0, how='all')
            inputs = df[df.columns[1:]]

            # Preprocessing
            inputs = self.preprocessing(inputs)
            # Save data
            print(f'save the dataset named {dev_name}')
            torch.save(inputs ,os.path.join(path, self.mode, f'{self.mode}_{dev_name}_X.pt'))
            
        return inputs

    def pad(self, data, pad_id, max_len):
        padded_data = []
        for x in data:
            if max_len - len(x) > 0:
                padded_data.append(torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
            else:
                padded_data.append(x)
        return list(padded_data)

    def preprocessing(self, inputs):
        print('Preprocessing ' + self.mode + ' dataset..')
        #Encoding original
        src_tensor = []
        seg_tensor = []
        for i in range(len(inputs)):
            src_tensor.append(torch.tensor(list(chain.from_iterable([self.tokenizer.encode(inputs[col][i], add_special_tokens=True) \
                                                                     for col in inputs.columns if inputs[col][i] == inputs[col][i]]))))
            clss = torch.cat([torch.where(src_tensor[i] == 2)[0], torch.tensor([len(src_tensor[i])])])
            seg_tensor.append(torch.tensor(list(chain.from_iterable( \
                [[0] * (clss[i + 1] - clss[i]).item() if i % 2 == 0 else [1] * (clss[i + 1] - clss[i]).item() \
                for i, val in enumerate(clss[:-1])]))))

        #Padding
        max_encoding_len = max(list(map(lambda x: len(x), src_tensor)))
        assert max_encoding_len < 512, 'Encoding length is longer than maximum processing length.'
        src_tensor = self.pad(src_tensor, 0, max_encoding_len)
        seg_tensor = self.pad(seg_tensor, 0, max_encoding_len)

        #Convert to list of tensor to 2d tensor
        src_tensor = torch.stack(src_tensor, dim=0)
        seg_tensor = torch.stack(seg_tensor, dim=0)
        mask_tensor = (~ (src_tensor == 0)).long()

        #Integrate the tensor {1st dimension : {src, seg, mask}, 2nd dim : {number of samples}, 3rd dim : {encoding dimension}}
        input_tensor = torch.cat([src_tensor.unsqueeze(dim=1) , seg_tensor.unsqueeze(dim=1), mask_tensor.unsqueeze(dim=1)], dim=1)

        return input_tensor

    def label_decoder(self, labels):
        try:
            labels = list(map(lambda x : self.intents[x], labels))
            return labels
        except:
            assert 'Invalid intent'

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index, :, :]

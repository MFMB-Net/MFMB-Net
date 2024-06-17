import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from models.missingTask.MFMB_NET.alignment_1 import Alignment
from models.missingTask.MFMB_NET.generator import Generator
from models.subNets.BertTextEncoder import BertTextEncoder

from models.missingTask.MFMB_NET.fusion_599 import Fusion

'''
这一版是t/v/a/concat直接进入classifier分类
每次数据集需要改 但我打算只拿mosi来验证单模态效果不好 简单concat效果好来说明浅层特征+需要融合
'''


class MFMB_NET(nn.Module):
    def __init__(self, args):
        super(MFMB_NET, self).__init__()
        self.args = args
        
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)

        self.batch=args.batch_size

        '''
          text_x, text_mask = text_x
        #batch,seq,dim
        #24,50,90   24,50
        #24,50,90
        #16,39,90
      
    
        

        #HNFnet 统一句子长度到和t一样，应该都是mosi mosei=50 sims=39
        audio_x, audio_mask = audio_x
        #24 375 90  24,375
        #24,500,90
        #16,400,90
 

        vision_x, vision_mask = vision_x
        #24,500,90  24,500
        #24,500,90
        #16,55,90'''
  
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.t_mosi=nn.Linear(1200,self.batch)#保证(seq*batch,)第一维相同
        self.v_mosi=nn.Linear(12000,self.batch)
        self.a_mosi=nn.Linear(9000,self.batch)
        
        self.t_mosei=nn.Linear(1200,self.batch)#保证seq相同
        self.v_mosei=nn.Linear(12000,self.batch)
        self.a_mosei=nn.Linear(12000,self.batch)
        
        self.t_sims=nn.Linear(624,self.batch)#保证seq相同
        self.v_sims=nn.Linear(880,self.batch)
        self.a_sims=nn.Linear(6400,self.batch)
        

        
        
        
        #分类器
        self.classifiert = nn.Sequential()
        self.classifiert.add_module('linear_trans_norm', nn.BatchNorm1d(self.orig_d_l))
        self.classifiert.add_module('linear_trans_hidden', nn.Linear(self.orig_d_l, self.orig_d_l//2))
        self.classifiert.add_module('linear_trans_activation', nn.ReLU())
        self.classifiert.add_module('linear_trans_drop', nn.Dropout(0.2))
        self.classifiert.add_module('linear_trans_final', nn.Linear(self.orig_d_l//2, 1))
        
        self.classifierv = nn.Sequential()
        self.classifierv.add_module('linear_trans_norm', nn.BatchNorm1d(self.orig_d_v))
        self.classifierv.add_module('linear_trans_hidden', nn.Linear(self.orig_d_v, self.orig_d_v//2))
        self.classifierv.add_module('linear_trans_activation', nn.ReLU())
        self.classifierv.add_module('linear_trans_drop', nn.Dropout(0.2))
        self.classifierv.add_module('linear_trans_final', nn.Linear(self.orig_d_v//2, 1))
        
        self.classifiera = nn.Sequential()
        self.classifiera.add_module('linear_trans_norm', nn.BatchNorm1d(self.orig_d_a))
        self.classifiera.add_module('linear_trans_hidden', nn.Linear(self.orig_d_a, self.orig_d_a//2))
        self.classifiera.add_module('linear_trans_activation', nn.ReLU())
        self.classifiera.add_module('linear_trans_drop', nn.Dropout(0.2))
        self.classifiera.add_module('linear_trans_final', nn.Linear(self.orig_d_a//2, 1))
        
        self.classifier2 = nn.Sequential()
        self.classifier2.add_module('linear_trans_norm', nn.BatchNorm1d(self.orig_d_l+ self.orig_d_a+self.orig_d_v))
        self.classifier2.add_module('linear_trans_hidden', nn.Linear(self.orig_d_l+ self.orig_d_a+self.orig_d_v, (self.orig_d_l+ self.orig_d_a+self.orig_d_v)//2))
        self.classifier2.add_module('linear_trans_activation', nn.ReLU())
        self.classifier2.add_module('linear_trans_drop', nn.Dropout(0.2))
        self.classifier2.add_module('linear_trans_final', nn.Linear((self.orig_d_l+ self.orig_d_a+self.orig_d_v)//2, 1))
                                    
    def forward(self, text, audio, vision):
        text, text_m, missing_mask_t = text
        #torch.Size([24, 3, 50]) torch.Size([24, 3, 50]) torch.Size([24, 50])
        #torch.Size([24, 3, 50]) torch.Size([24, 3, 50]) torch.Size([24, 50])
        audio, audio_m, audio_mask, missing_mask_a = audio
        #torch.Size([24, 375, 5]) torch.Size([24, 375, 5]) torch.Size([24, 375]) torch.Size([24, 375])
        #torch.Size([24, 500, 74]) torch.Size([24, 500, 74]) torch.Size([24, 500]) torch.Size([24, 500])

        vision, vision_m, vision_mask, missing_mask_v = vision
        #torch.Size([24, 500, 20]) torch.Size([24, 500, 20]) torch.Size([24, 500]) torch.Size([24, 500])
        #torch.Size([24, 500, 35]) torch.Size([24, 500, 35]) torch.Size([24, 500]) torch.Size([24, 500])
        text_mask = text[:,1,:]
        #torch.Size([24, 50])
        #torch.Size([24, 50])
        text_m = self.text_model(text_m)
        #torch.Size([24, 50, 768])
        #torch.Size([24, 50, 768])

        text = self.text_model(text)
        #torch.Size([24, 50, 768])
        #torch.Size([24, 50, 768])
        
        
        
        
        
        
        
        #简单拼接的
        text_m=text_m.reshape(-1,self.orig_d_l)#1200, 768
        text_m = self.t_mosi(text_m.permute(1,0))#768,1200-768,50
        text_m=text_m.permute(1,0)#50,768
        
        vision_m=vision_m.reshape(-1,self.orig_d_v)
        vision_m = self.v_mosi(vision_m.permute(1,0))
        vision_m=vision_m.permute(1,0)
        
        audio_m=audio_m.reshape(-1,self.orig_d_a)
        audio_m = self.a_mosi(audio_m.permute(1,0))
        audio_m=audio_m.permute(1,0)
 

        #单独t
        utterance_rep = text_m#50,768
        prediction = self.classifiert(utterance_rep)
        return prediction, torch.Tensor([0]).to(self.args.device)
        
        #单独v
        #utterance_rep = vision_m
        #prediction = self.classifierv(utterance_rep)
        #return prediction, torch.Tensor([0]).to(self.args.device)
    
        #单独a
        #utterance_rep = audio_m
        #prediction = self.classifiera(utterance_rep)
        #return prediction, torch.Tensor([0]).to(self.args.device)
        
        
        #utterance_rep = torch.cat((text_m, audio_m, vision_m), dim=1)
        #prediction = self.classifier2(utterance_rep)
        #return prediction, torch.Tensor([0]).to(self.args.device)
        
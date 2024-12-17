import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.missingTask.MFMB_NET.modules.transformer import TransformerEncoder
from einops.layers.torch import Rearrange
class MLP_block(nn.Module):
    def __init__(self,input_size,hidden_size,dropout=0.5):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size,input_size),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        x=self.net(x)
        return x

class MLP_Communicator(nn.Module):
    def __init__(self,token,channel,hidden_size,depth=2):
        super(MLP_Communicator,self).__init__()
        self.depth=depth
        self.token_mixer=nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel,hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer=nn.Sequential(
            MLP_block(input_size=token,hidden_size=hidden_size)
        )

    def forward(self,x):
        for _ in range(self.depth):
            x=x+self.token_mixer(x)
            x=x+self.channel_mixer(x)
        return x
    
class BottleAttentionNet(nn.Module):
    def __init__(self):
        super(BottleAttentionNet,self).__init__()
        self.embed_dim=90#mosi
        self.embed_dim_common=32
        self.seq=4

        self.layer_unimodal=2
        self.layer_multimodal=2
        self.Linear_common=nn.Linear(self.embed_dim,self.embed_dim_common)
        self.transformer=TransformerEncoder(embed_dim=self.embed_dim,num_heads=10,layers=4,attn_mask=False)

    def forward(self,audio,visual,text):
        #audio=self.Linear_common(audio)
        #visual=self.Linear_common(visual)

        for i in range(self.layer_unimodal):
            audio=self.transformer(audio)#seq_len,batchsize,dim

        for i in range(self.layer_unimodal):
            visual=self.transformer(visual)
            #text=self.transformer(text)

        fsn=torch.zeros(self.seq,audio.size(1),self.embed_dim).cuda()

        x=torch.cat([audio,fsn],dim=0)

        for i in range(self.layer_multimodal):
            if i==0:
                #[audio,fsn]-[fsn',visual]-[fsn'',text]-----[fsn''']
                x=self.transformer(x) #54，24，32
                x=torch.cat([x[audio.size(0):,:,:],visual],dim=0)
                x=self.transformer(x)
                x=torch.cat([x[:self.seq,:,:],text],dim=0)
                x = self.transformer(x)
            else:
                x=self.transformer(x)

        x=x[:self.seq,:,:]
        return x

class GRUencoder(nn.Module):
    """Pad for utterances with variable lengths and maintain the order of them after GRU"""
    def __init__(self, embedding_dim, utterance_dim, num_layers):
        super(GRUencoder, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=utterance_dim,
                          bidirectional=True, num_layers=num_layers)

    def forward(self, utterance, utterance_lens):
        """Server as simple GRU Layer.
        Args:
            utterance (tensor): [utter_num, max_word_len, embedding_dim]
            utterance_lens (tensor): [utter_num]
        Returns:
            transformed utterance representation (tensor): [utter_num, max_word_len, 2 * utterance_dim]
        """
        utterance_embs = utterance.transpose(0,1)

        # SORT BY LENGTH.
        sorted_utter_length, indices = torch.sort(utterance_lens, descending=True)
        _, indices_unsort = torch.sort(indices)
        
        s_embs = utterance_embs.index_select(1, indices)

        # PADDING & GRU MODULE & UNPACK.
        utterance_packed = pack_padded_sequence(s_embs, sorted_utter_length.cpu())
        utterance_output = self.gru(utterance_packed)[0]
        utterance_output = pad_packed_sequence(utterance_output, total_length=utterance.size(1))[0]

        # UNSORT BY LENGTH.
        utterance_output = utterance_output.index_select(1, indices_unsort)

        return utterance_output.transpose(0,1)

class C_GATE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, drop):
        super(C_GATE, self).__init__()

        # BI-GRU to get the historical context.
        self.gru = GRUencoder(embedding_dim, hidden_dim, num_layers)
        # Calculate the gate.
        self.cnn = nn.Conv1d(in_channels= 2 * hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        # Linear Layer to get the representation.
        self.fc = nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim)
        # Utterance Dropout.
        self.dropout_in = nn.Dropout(drop)
        
    def forward(self, utterance, utterance_mask):
        """Returns:
            utterance_rep: [utter_num, utterance_dim]
        """
        add_zero = torch.zeros(size=[utterance.shape[0], 1], requires_grad=False).type_as(utterance_mask).to(utterance_mask.device)
        utterance_mask = torch.cat((utterance_mask, add_zero), dim=1)
        utterance_lens = torch.argmin(utterance_mask, dim=1)

        transformed_ = self.gru(utterance, utterance_lens) # [batch_size, seq_len, 2 * hidden_dim]

        gate = F.sigmoid(self.cnn(transformed_.transpose(1, 2)).transpose(1, 2))  # [batch_size, seq_len, 1]
        gate_x = torch.tanh(transformed_) * gate # [batch_size, seq_len, 2 * hidden_dim]

        utterance_rep = torch.tanh(self.fc(torch.cat([utterance, gate_x], dim=-1))) # [batch_size, seq_len, hidden_dim]

        utterance_rep = torch.max(utterance_rep, dim=1)[0] # [batch_size, hidden_dim]

        # UTTERANCE DROPOUT
        utterance_rep = self.dropout_in(utterance_rep) # [utter_num, utterance_dim]

        return utterance_rep

class GATE_F(nn.Module):
    def __init__(self, args):
        super(GATE_F, self).__init__()
        
        self.text_encoder = C_GATE(args.fusion_t_in, args.fusion_t_hid, args.fusion_gru_layers, args.fusion_drop)
        self.audio_encoder = C_GATE(args.fusion_a_in, args.fusion_a_hid, args.fusion_gru_layers, args.fusion_drop)
        self.vision_encoder = C_GATE(args.fusion_v_in, args.fusion_v_hid, args.fusion_gru_layers, args.fusion_drop)

        self.audio_visual_model = BottleAttentionNet()

       
        #1.For mosi
        self.seq_v_mosi=nn.Linear(500,50)
        self.seq_a_mosi=nn.Linear(375,50)
        
        
        #2.mosei
        self.seq_v_mosei=nn.Linear(500,50)
        self.seq_a_mosei=nn.Linear(500,50)
        
        
        #stack
        self.common_size=64
        self.dim=self.common_size
        self.batchnorm=nn.BatchNorm1d(2,affine=False)
        self.MLP_Communicator1=MLP_Communicator(self.dim,2,hidden_size=64,depth=1)
        self.MLP_Communicator2 = MLP_Communicator(self.dim, 2, hidden_size=64, depth=1)
        self.args=args
        self.batch_size=self.args.batch_size
        #self.batch_size=24 
        #stack
        self.t_stack_linear=nn.Linear(36,self.common_size)
        self.v_stack_linear = nn.Linear(48,self.common_size)
        self.a_stack_linear = nn.Linear(20,self.common_size)
        
        
        # 1.classification
        self.classifier1 = nn.Sequential()
        self.classifier1.add_module('linear_trans_norm', nn.BatchNorm1d(self.common_size * 4+90))
        self.classifier1.add_module('linear_trans_hidden', nn.Linear(self.common_size * 4+90, args.cls_hidden_dim))
        self.classifier1.add_module('linear_trans_activation', nn.LeakyReLU())
        self.classifier1.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier1.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))

        # 2.concat all for classification
        self.classifier2 = nn.Sequential()
        self.classifier2.add_module('linear_trans_norm', nn.BatchNorm1d(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid+ self.common_size*4+90))
        self.classifier2.add_module('linear_trans_hidden', nn.Linear(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid  +self.common_size*4+90, args.cls_hidden_dim))
        self.classifier2.add_module('linear_trans_activation', nn.ReLU())
        self.classifier2.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier2.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))
     
        

    def forward(self, text_x, audio_x, vision_x):
        text_x, text_mask = text_x

        #mosi mosei
        text_x_fusion=text_x.permute(1,0,2)#50,24,90 #seq,batchsize,dim
        #text_x_fusion=text_x_fusion.permute(2,0,1)#seq_len,batchsize,dim 50,24,90
  
        audio_x, audio_mask = audio_x
        audio_x_fusion=self.seq_a_mosi(audio_x.permute(0,2,1))#24,90,375--24,90,50-
        audio_x_fusion=audio_x_fusion.permute(2,0,1)#seq_len,batchsize,dim 50,24,90

        vision_x, vision_mask = vision_x
        vision_x_fusion=self.seq_v_mosi(vision_x.permute(0,2,1))
        vision_x_fusion=vision_x_fusion.permute(2,0,1)#50,24,90

        audio_visual_fusion=self.audio_visual_model(audio_x_fusion,vision_x_fusion,text_x_fusion)#4，24，90
        audio_visual_fusion=audio_visual_fusion[-1]#24,90

        #C_GATE
        text_rep = self.text_encoder(text_x, text_mask)
        text_rep_common=self.t_stack_linear(text_rep)
     
        audio_rep = self.audio_encoder(audio_x, audio_mask)
        audio_rep_common = self.a_stack_linear(audio_rep)
     
        vision_rep = self.vision_encoder(vision_x, vision_mask)
        vision_rep_common = self.v_stack_linear(vision_rep)

 
        stack_rep_ta=torch.stack((text_rep_common,audio_rep_common),dim=0) #(2,batch_size,common_size) 2,24,64
        stack_rep_tv = torch.stack((text_rep_common, vision_rep_common), dim=0)#2,24,64

        stack_rep_ta = self.batchnorm(stack_rep_ta.permute(1,0,2))# batchsize,channanl,commonsize 24,2,64
        stack_rep_tv = self.batchnorm(stack_rep_tv.permute(1,0,2))  # 24,2,64

        #Adjust dim for classification
        #stack_rep_tv = stack_rep_tv.reshape(24, -1)
        stack_rep_tv=stack_rep_tv.reshape(self.batch_size,-1)
        stack_rep_ta = stack_rep_ta.reshape(self.batch_size, -1)
        

        #utterance_rep = torch.cat((stack_rep_tv, stack_rep_ta,audio_visual_fusion), dim=1)
        #return self.classifier1(utterance_rep)

        utterance_rep = torch.cat((text_rep, audio_rep, vision_rep,stack_rep_tv,stack_rep_ta,audio_visual_fusion), dim=1)


        
        return self.classifier2(utterance_rep)

       
      
        

MODULE_MAP = {
    'c_gate': GATE_F,
}

class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()

        select_model = MODULE_MAP[args.fusionModule]

        self.Model = select_model(args)

    def forward(self, text_x, audio_x, vision_x):

        return self.Model(text_x, audio_x, vision_x)

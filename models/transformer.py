"""

PyTorch Transformer model for the time-series imputation task.

Some part of the code is from https://github.com/WenjieDu/SAITS.

"""



# Created by Wenjie Du <wenjay.du@gmail.com>

# License: GPL-v3

import pickle

import numpy as np
import pandas as pd

import torch

import torch.cuda

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader

from scipy.spatial.distance import cosine

from scipy.special import kl_div


from pypots.data.base import BaseDataset

from pypots.data.dataset_for_mit import DatasetForMIT

from pypots.data.integration import mcar, masked_fill

from pypots.imputation.base import BaseNNImputer

from pypots.utils.metrics import cal_mae

from sklearn import metrics

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



device = torch.device("cuda:1"  if torch.cuda.is_available() else 'cpu')
 
class AUCPR():
    def __init__(self, *args, **kwargs):
        self.y_true = None
        self.y_pred = None

    def add(self, pred, y_true):
        if self.y_pred is None:
            self.y_pred = pred
        else:
            self.y_pred = np.concatenate([self.y_pred, pred])

        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate([self.y_true, y_true])

    def get(self):
        (precisions, recalls, thresholds) = metrics.precision_recall_curve( 
            self.y_true, self.y_pred)
            
        auprc = metrics.auc(recalls, precisions)
        #auprc=metrics.average_precision_score(self.y_true, self.y_pred)
        return auprc

    def save(self, name):
        fname = name + ".pkl"
        with open(fname, 'wb') as f:
            pickle.dump((self.y_pred, self.y_true), f, pickle.HIGHEST_PROTOCOL)

def calculate_similarity_matrix(data):
    num_variables = data.shape[0]
    similarity_matrix = np.zeros((num_variables, num_variables))

    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            similarity_matrix[i, j] = 1 - cosine(data[i,:], data[j,: ])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix
    
def adjust_matrix(matrix):
    min_val = np.min(matrix)
    if min_val < 0:
        constant = abs(min_val)
        adjusted_matrix = matrix + constant
    else:
        adjusted_matrix = matrix
    return adjusted_matrix
    

def calculate_kl_divergence(p, q):
    kl_divergence = kl_div(p, q)
    return kl_divergence

class ScaledDotProductAttention(nn.Module):

    """Scaled dot-product attention"""



    def __init__(self, temperature, attn_dropout=0.1):

        super().__init__()

        self.temperature = temperature

        self.dropout = nn.Dropout(attn_dropout)



    def forward(self, q, k, v, attn_mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if attn_mask is not None:

            attn = attn.masked_fill(attn_mask == 1, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module): 

    """original Transformer multi-head attention"""



    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):

        super().__init__()



        self.n_head = n_head

        self.d_k = d_k

        self.d_v = d_v



        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)

        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)

        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)



        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout) 

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)



    def forward(self, q, k, v, attn_mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)



        # Pass through the pre-attention projection: b x lq x (n*dv)

        # Separate different heads: b x lq x n x dv

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)

        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)

        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)



        # Transpose for attention dot product: b x n x lq x dv  # [batch_size * n_head, seq_length, d_model/n_head]

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)



        if attn_mask is not None:

            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim

            attn_mask = attn_mask.unsqueeze(0).unsqueeze(

                1

            )  # For batch and head axis broadcasting.



        v, attn_weights = self.attention(q, k, v, attn_mask)



        # Transpose to move the head dimension back: b x lq x n x dv

        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv) #

        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        v = self.fc(v)

        return v, attn_weights





class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):

        super().__init__()

        self.w_1 = nn.Linear(d_in, d_hid)

        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

        self.dropout = nn.Dropout(dropout)



    def forward(self, x):

        residual = x

        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))

        x = self.dropout(x)

        x += residual

        return x





class EncoderLayer(nn.Module):

    def __init__(

        self,

        d_time,

        d_feature,

        d_model,

        d_inner,

        n_head,

        d_k,

        d_v,

        dropout=0.1,

        attn_dropout=0.1,

        diagonal_attention_mask=False,

    ):

        super().__init__()



        self.diagonal_attention_mask = diagonal_attention_mask

        self.d_time = 59# d_time

        self.d_feature = d_feature



        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout) 

        self.dropout = nn.Dropout(dropout)

        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

        



    def forward(self, enc_input):

        if self.diagonal_attention_mask:

            mask_time = torch.eye(self.d_time).to(enc_input.device)

        else:

            mask_time = None



        residual = enc_input

        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745

        enc_input = self.layer_norm(enc_input)

        enc_output, attn_weights = self.slf_attn(

            enc_input, enc_input, enc_input, attn_mask=mask_time

        )

        enc_output = self.dropout(enc_output)

        enc_output += residual



        enc_output = self.pos_ffn(enc_output)

        return enc_output, attn_weights





class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):

        super().__init__()

        # Not a parameter

        self.register_buffer(

            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)

        )



    @staticmethod

    def _get_sinusoid_encoding_table(n_position, d_hid):

        """Sinusoid position encoding table"""



        def get_position_angle_vec(position):

            return [

                position / np.power(10000, 2 * (hid_j // 2) / d_hid)

                for hid_j in range(d_hid)

            ]



        sinusoid_table = np.array(

            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]

        )

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i

        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)



    def forward(self, x):
        #print("x",x.shape,x.size(1))  # 32 59,256
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_q = nn.Linear(input_size, hidden_size)
        self.linear_k = nn.Linear(input_size, hidden_size)
        self.linear_v = nn.Linear(input_size, hidden_size)
        self.trans = nn.Linear(256, 512)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        B, T, F = x.size()
        Q = self.linear_q(x)  # [B, T, H]
        K = self.linear_k(x)  # [B, T, H]
        V = self.linear_v(x)  # [B, T, H]
       # Q = Q.view(B, T, -1, self.hidden_size)  # [B, T, S, H]
       # K = K.view(B, T, self.hidden_size, -1)  # [B, T, H, S]
       # V = V.view(B, T, -1, self.hidden_size)  # [B, T, S, H]
        S = torch.matmul(Q, K.transpose(1, 2))  # [B, T, S, S]
        S = S / self.hidden_size**0.5
        S = self.softmax(S)
        S = self.dropout(S)
        A = torch.matmul(S, V)  # [B, T, S, H]
        #print("A size",x.shape,A.shape) 32 48 512   32  48 256
        #A = A.view(B, T, -1)  # [B, T, F']
        x = x + self.trans(A)
        x = self.layer_norm(x)  
        return x
class PositiveWeightMatrix(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PositiveWeightMatrix, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        # 
        #nn.init.xavier_uniform_(self.weights)
        #nn.init.constant_(self.bias, 0.1)  # 

    def forward(self, input_tensor):
        weighted_input = torch.matmul(input_tensor, self.weights.t()) + self.bias
        positive_weights = torch.sigmoid(weighted_input)  # 
        #print(positive_weights,positive_weights.shape)
        return positive_weights


class _TransformerEncoder(nn.Module):

    def __init__(

        self,

        n_layers,

        d_time,

        d_feature,

        d_model,

        d_inner,

        n_head,

        d_k,

        d_v,

        dropout,

        ORT_weight=1,

        MIT_weight=1,

    ):

        super().__init__()

        self.n_layers = n_layers

        actual_d_feature = d_feature * 2

        self.ORT_weight = ORT_weight

        self.MIT_weight = MIT_weight
        
        self.layer_norm = nn.LayerNorm(256) #add


        self.layer_stack = nn.ModuleList(

            [

                EncoderLayer(

                    d_time,

                    actual_d_feature,

                    d_model,

                    d_inner,

                    n_head,

                    d_k,

                    d_v,

                    dropout,

                    0,

                    False,

                )

                for _ in range(n_layers)

            ]

        )


        self.var_embedding = nn.Linear(48, d_model) 
        
        self.embedding = nn.Linear(actual_d_feature, d_model)

        self.position_enc = PositionalEncoding(d_model, n_position=d_time) 
        
        self.attn= SelfAttention(59*2, 256)

        self.dropout = nn.Dropout(p=dropout)

        self.reduce_dim = nn.Linear(d_model, d_feature)
        
        self.var_reduce_dim = nn.Linear(d_model, 48)

        self.dense = nn.Linear(768, d_model)
        
        self.dense59 = nn.Linear(768, 59)
        
        self.dense48 = nn.Linear(768, 48)
        
        self.dense200to256=nn.Linear(200, 256)
        
        self.dense256 = nn.Linear(768, 256)

        self.hist_reg = nn.Linear(d_model, 48)

        self.text_matrix_conv = nn.Conv1d(in_channels=768, out_channels=59, kernel_size=12)
        
        self.align =nn.Linear(768, 48) 
        
        self.c59_48=nn.Linear(59,48)
        
        self.fusion = nn.Linear(59*2, d_feature)    ## old 59*2  step 256*2
        
        self.fusion256 = nn.Linear(256*2, 256)
        
        self.add =nn.Linear(60, d_feature)
        
        #self.W = nn.Parameter(torch.Tensor(1, 128))
        self.w = nn.Linear(768, 256, bias=False)
        
        self.attn= SelfAttention(256*2, 256)
            
        self.proj0 =nn.Linear(256, 256) 
        
        self.proj = nn.Linear(59, 59)
        
        self.proj1=nn.Linear(59*2, 59)
        
        self.dense2 = nn.Linear(768, 48)
        
        self.dense128to59=nn.Linear(128,59)
        
        self.dense256to59=nn.Linear(256,59)
        
        self.dense256to48=nn.Linear(256,48)
        
        self.Mh_attn = MultiHeadAttention(6, 768, 128, 128, dropout)

        self.pos_ffn = PositionWiseFeedForward(768, 128, dropout)
        
        
        self.lstm = nn.LSTM(59, 256*2, 1, batch_first=True, bidirectional=True)
        
        self.FinalFC=nn.Linear(256, 1, bias=False)  
        
        self.final_act = torch.sigmoid
        
        self.criterion= nn.BCEWithLogitsLoss()
        
        self.out = nn.Linear(d_feature, 1, bias=False)
        
        self.out2 = nn.Linear(2*d_feature, 1, bias=False)
        
        self.out256 = nn.Linear(d_model, 1, bias=False) 
        
        self.bicross=nn.Bilinear(59,59,59)

        self.bicross256 =nn.Bilinear(256,256,256)
        
        self.classifier =nn.Linear(d_feature, 2)


    def impute(self, inputs, x_vectors, text_emb, labels):
        X, masks = inputs["X"], inputs["missing_mask"]
        
        # 1. 变量嵌入和位置编码
        input_var = self.var_embedding(torch.transpose(X,1,2).contiguous()) # [32, 59, 256]       
        enc_output = self.dropout(input_var)

        # 2. Transformer编码器处理时间序列
        for encoder_layer in self.layer_stack:
            enc_output, _ = encoder_layer(enc_output)            

        # 3. 文本和变量的注意力融合
        # 计算文本向量和变量向量之间的注意力得分
        scores = torch.matmul(x_vectors, torch.transpose(text_emb, 1, 2).contiguous()) #[batch,59,768] x [batch,768,128]
        attn = torch.softmax(scores, dim=-1) #[batch,59,128] 
        # 基于注意力权重融合文本信息
        text_fused = torch.bmm(attn, text_emb) #[batch,59,768]
        # 将文本特征转换到相同维度
        text_fused256 = self.dense256(text_fused) #[batch,59,256]

        # 4. 对比学习损失计算
        var_flatten = torch.reshape(text_fused256, (-1, text_fused256.size(-1)))   # [batch*59, 256]
        ts_flatten = torch.reshape(enc_output, (-1, enc_output.size(-1))) # [batch*59, 256]
        
        # 计算文本和时间序列表示之间的对比损失
        projs = torch.cat((var_flatten, ts_flatten)) # [batch*59*2, 256]
        b = var_flatten.shape[0] 
        n = b*2
        
        # 计算相似度矩阵
        contr_logits = projs @ projs.t()
        contrmask = torch.eye(n,device = device).bool()
        contr_logits = contr_logits[~contrmask].reshape(n, n - 1)    
        contr_logits /= 0.05 # 温度系数
        
        # 计算对比损失
        contr_labels = torch.cat(((torch.arange(b, device = device) + b - 1), torch.arange(b, device=device)), dim=0)
        contr_loss = F.cross_entropy(contr_logits, contr_labels, reduction='sum')
        contr_loss /= n*n

        # 5. 双线性融合
        # 对文本和时间序列特征进行层归一化
        text_fused256 = self.layer_norm(text_fused256)
        enc_output = self.layer_norm(enc_output)
        
        # 计算双线性注意力
        bilinear_fusion = self.bicross256(text_fused256, enc_output)  
        a = torch.softmax(torch.bmm(text_fused256,torch.transpose(enc_output,2,1).contiguous()), dim=-1)  
        baf = torch.bmm(a, bilinear_fusion)

        # 6. 最终预测
        # 使用最后一个时间步进行分类预测
        y_h = self.out256(fusion[:,-1,:]).squeeze(-1)
        Probs = self.final_act(y_h)

        # 7. 重建时间序列数据
        imputed_data = masks * X + (1 - masks) * enc_output

        return imputed_data, enc_output, contr_loss, y_h



    def forward(self,inputs,x_vectors,text_emb,labels):

        X, masks = inputs["X"], inputs["missing_mask"]

        
        x_vectors=x_vectors.to(device) #x_vectors [64,59,768]
        
        #print("text_emb",text_emb.shape) # 5 128 768

        scores = torch.matmul(x_vectors, torch.transpose(text_emb, 1, 2).contiguous())#[64,59,768]  [64,768,128]
        
        attn = torch.softmax(scores, dim=-1) #[64,59,128]
        
        text_fused = torch.bmm(attn, text_emb) #[64,59,768]
        
        text_fused= self.dense59(torch.transpose(self.c59_48(torch.transpose(text_fused,2,1).contiguous()),2,1))

        
        text_fused=text_fused.double()  
             
        text_intact, text_fused, m_mask, indicating_mask = mcar(text_fused, 0.5)
        
        #text_intact=torch.from_numpy(text_intact)
    
        text_fused = masked_fill(text_fused, 1 - m_mask, np.nan)
        
        m_mask = ~torch.isnan(text_fused)
        
        text_fused = torch.nan_to_num(X)    
        
        
        imputed_data, learned_presentation,y_loss,contr_loss,PredScores,TrueLabels,PredLabels = self.impute(inputs,x_vectors,text_emb,labels)
        
        #print("h,label",learned_presentation.shape,PredLabels.shape) #[32, 48, 59],[32,1]
        #with open('learned_presentation.pkl', 'wb') as f:
        #    pickle.dump(learned_presentation.detach().cpu().numpy(), f)  
            
            
          
        with open('PredLabels.pkl', 'wb') as f:
            pickle.dump(PredLabels.detach().cpu().numpy(), f) 
        #print("pre label",PredLabels) 
        
        

        reconstruction_loss = cal_mae(learned_presentation, X, masks)
         
        rec_txt_loss=cal_mae(learned_presentation,text_fused, m_mask)
        
        #print("learned_presentation,text_fused, m_mask", learned_presentation,text_fused, m_mask)
        
        #print("reconstruction_loss ,rec_txt_loss",reconstruction_loss,rec_txt_loss)

        # have to cal imputation loss in the val stage; no need to cal imputation loss here in the tests stage

        imputation_loss = cal_mae(

            learned_presentation, inputs["X_intact"], inputs["indicating_mask"]

        ) 
        
        imp_txt_loss = cal_mae(

            learned_presentation, text_intact, indicating_mask

        )
        
        #print("impu loss ,losstxt",imputation_loss,imp_txt_loss)
        
        
       # print("imputation_loss y",imputation_loss,y_loss)  #imp_txt_loss+rec_txt_loss+  
       
        #print("before loss",reconstruction_loss+y_loss)       0.5*contr_loss+print("con loss",contr_loss)    reconstruction_loss  
         
        loss = contr_loss+reconstruction_loss+imputation_loss+y_loss # reconstruction_loss+imputation_loss+ reconstruction_loss+imputation_loss+   reconstruction_loss+imputation_loss+ self.ORT_weight * reconstruction_loss + self.MIT_weight * imputation_loss #+ y_loss     



        return {

            "imputed_data": imputed_data,

            "reconstruction_loss": reconstruction_loss,

            "imputation_loss": imputation_loss,

            "loss": loss,
            
            "PredScores":PredScores,
            
            "TrueLabels":TrueLabels,
            
            "PredLabels":PredLabels, 

        }





class Transformer(BaseNNImputer):

    def __init__(

        self,

        n_steps,

        n_features,

        n_layers,

        d_model,

        d_inner,

        n_head,

        d_k,

        d_v,

        dropout,

        ORT_weight=1, 

        MIT_weight=1,

        learning_rate=3e-4,#3e-4, # 1e-3 

        epochs=100,

        patience=10, 

        batch_size=5,             

        weight_decay=1e-5,#1e-5,  #1e-5 best: 1e-4 

        device=None,

    ):

        super().__init__(

            learning_rate, epochs, patience, batch_size, weight_decay, device

        )



        self.n_steps = n_steps

        self.n_features = n_features

        # model hype-parameters

        self.n_layers = n_layers

        self.d_model = d_model

        self.d_inner = d_inner

        self.n_head = n_head

        self.d_k = d_k

        self.d_v = d_v

        self.dropout = dropout

        self.ORT_weight = ORT_weight

        self.MIT_weight = MIT_weight



        self.model = _TransformerEncoder(

            self.n_layers,

            self.n_steps,

            self.n_features,

            self.d_model,

            self.d_inner,

            self.n_head,

            self.d_k,

            self.d_v,

            self.dropout,

            self.ORT_weight,

            self.MIT_weight,

        )

        self.model = self.model.to(self.device)

        self._print_model_size()



    def fit(self, train_X,x_vectors,text_emb,labels, val_X=None):

        train_X = self.check_input(self.n_steps, self.n_features, train_X)

        if val_X is not None:

            val_X = self.check_input(self.n_steps, self.n_features, val_X)



        training_set = DatasetForMIT(train_X,x_vectors,text_emb,labels)

        training_loader = DataLoader(

            training_set, batch_size=self.batch_size, shuffle=True

        )

        if val_X is None:

            self._train_model(training_loader)

        else:

            val_X_intact, val_X, val_X_missing_mask, val_X_indicating_mask = mcar(

                val_X, 0.2

            )

            val_X = masked_fill(val_X, 1 - val_X_missing_mask, np.nan)

            val_set = DatasetForMIT(val_X)

            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

            self._train_model(

                training_loader, val_loader, val_X_intact, val_X_indicating_mask

            )



        self.model.load_state_dict(self.best_model_dict) 

        self.model.eval()  # set the model as eval status to freeze it.

        return self
        
   

    def assemble_input_data(self, data):

        """Assemble the input data into a dictionary.



        Parameters

        ----------

        data : list

            A list containing data fetched from Dataset by Dataload.



        Returns

        -------

        inputs : dict

            A dictionary with data assembled.

        """



        indices, X_intact, X, missing_mask, indicating_mask = data

       # print("data?",data,len(data))  5  right

        inputs = {

            "X": X,

            "X_intact": X_intact,

            "missing_mask": missing_mask,

            "indicating_mask": indicating_mask,

        }

       # print("inputs",inputs)

        return inputs



    def impute(self, X,x_vectors,text_emb,labels):

        X = self.check_input(self.n_steps, self.n_features, X)

        self.model.eval()  # set the model as eval status to freeze it.

        test_set = DatasetForMIT(X,x_vectors,text_emb,labels)#BaseDataset(X)

        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        imputation_collector = []
        prob_collector=[]
        aucpr_collector=[]
        predlabel_collector=[]

        #x_vectors=x_vectors.to(device)
        
        PredScores = torch.tensor([]).to("cuda:1")
        TrueLabels = torch.tensor([]).to("cuda:1")
        PredLabels = torch.tensor([]).to("cuda:1")
        aucpr_obj = AUCPR()  

        with torch.no_grad(): 

            for idx, (data,x_vectors,text_emb,labels) in enumerate(test_loader):

                inputs = {"X": data[2], "missing_mask": data[3]} #??"X": data[1], "missing_mask": data[2]->"X": data[2], "missing_mask": data[3]

                #print("inputs",inputs)#"data?",data,

                imputed_data, _ ,y_loss,contr_loss,predscores,truelabels,predlabels = self.model.impute(inputs,x_vectors,text_emb,labels)

                imputation_collector.append(imputed_data) 
                prob_collector.append(predscores)#predscores
                predlabel_collector.append(predlabels)
                 
                with torch.no_grad():
            
                  PredScores = torch.cat([PredScores,predscores])
                  TrueLabels = torch.cat([TrueLabels,truelabels])
                  PredLabels = torch.cat([PredLabels,predlabels])
                #print("shape",predscores.shape,labels.shape)
                aucpr_obj.add(predscores.detach().cpu(), labels.detach().cpu())   
                torch.cuda.empty_cache()
                #aucpr1 = aucpr_obj.get()    
                
            f1 = metrics.f1_score(TrueLabels.detach().cpu(), PredLabels.detach().cpu(), average= "weighted")            
            roc_macro = roc_auc_score(TrueLabels.detach().cpu() , PredScores.detach().cpu() , average='macro')            
            aucpr2 = aucpr_obj.get()
           #print("true label",y_h,Probs) 
            print(" 1 train 2 test roc_macro,aucpr_obj",roc_macro,f1,aucpr2)  #f1,f2,aucpr1 
            torch.cuda.empty_cache()

        imputation_collector = torch.cat(imputation_collector)
        prob_collector = torch.cat(prob_collector)
        predlabel_collector=torch.cat(predlabel_collector)
        torch.cuda.empty_cache()
        print("shape",len(imputation_collector))
        print("shape2 ",len(prob_collector)) 
        
        #return imputation_collector,prob_collector,predlabel_collector
        return imputation_collector.cpu().detach().numpy(),prob_collector.cpu().detach().numpy(),predlabel_collector.cpu().detach().numpy() 


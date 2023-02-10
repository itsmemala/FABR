import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        config = BertConfig.from_pretrained(args.bert_model)
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False

        self.relu=torch.nn.ReLU()
        # self.mcl = MCL(args,taskcla)
        # self.ac = AC(args,taskcla)
        if self.args.mlp_depth==1:
            self.mlp = torch.nn.Linear(args.max_seq_length*args.bert_hidden_size,args.bert_hidden_size) # Input Shape: bert output shape (flattened)
        elif self.args.mlp_depth==2:
            self.mlp = torch.nn.Linear(args.max_seq_length*args.bert_hidden_size,128)
            self.mlp2 = torch.nn.Linear(128,args.bert_hidden_size)

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.bert_hidden_size,n))


        print('BERT (Fixed) + GRU + KAN')


        return

    def forward(self,t, input_ids, segment_ids, input_mask, which_type,s=None,my_debug=0):
        # sequence_output, pooled_output = \
            # self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        res = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = res['last_hidden_state'] # Added this to fix err: 'str' object has no attribute 'size'

        sequence_output_flattened = torch.flatten(sequence_output,start_dim=1,end_dim=-1)
        # print(sequence_output_flattened.shape)
        # print(sequence_output_flattened.shape[0])
        # print(sequence_output_flattened.shape[1])
        assert len(sequence_output_flattened.shape)==2 and sequence_output_flattened.shape[0]==input_ids.shape[0] and sequence_output_flattened.shape[1]==self.args.max_seq_length*self.args.bert_hidden_size
        if self.args.mlp_depth==1:
            mcl_hidden = self.mlp(sequence_output_flattened)
        elif self.args.mlp_depth==2:
            mcl_hidden_temp = self.mlp(sequence_output_flattened)
            mcl_hidden_temp = self.relu(mcl_hidden_temp)
            mcl_hidden = self.mlp2(mcl_hidden_temp)
        h=self.relu(mcl_hidden)

        #loss ==============
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        if my_debug==2:
            # print('activations size (current batch):',h.size())
            return h, None
        return y

    # def get_view_for(self,n,mask):
        # if n=='mcl.gru.rnn.weight_ih_l0':
            # # print('not none')
            # return mask.data.view(1,-1).expand_as(self.mcl.gru.rnn.weight_ih_l0)
        # elif n=='mcl.gru.rnn.weight_hh_l0':
            # return mask.data.view(1,-1).expand_as(self.mcl.gru.rnn.weight_hh_l0)
        # elif n=='mcl.gru.rnn.bias_ih_l0':
            # return mask.data.view(-1).repeat(3)
        # elif n=='mcl.gru.rnn.bias_hh_l0':
            # return mask.data.view(-1).repeat(3)
        # return None



# class AC(nn.Module):
    # def __init__(self,args,taskcla):
        # super().__init__()

        # self.gru = GRU(
                    # embedding_dim = args.bert_hidden_size,
                    # hidden_dim = args.bert_hidden_size,
                    # n_layers=1,
                    # bidirectional=False,
                    # dropout=0.5,
                    # args=args)

        # self.efc=torch.nn.Embedding(args.num_task,args.bert_hidden_size)
        # self.gate=torch.nn.Sigmoid()


    # def mask(self,t,s=1):
        # gfc=self.gate(s*self.efc(torch.LongTensor([t]).cuda()))
        # return gfc


# class MCL(nn.Module):
    # def __init__(self,args,taskcla):
        # super().__init__()

        # self.gru = GRU(
                    # embedding_dim = args.bert_hidden_size,
                    # hidden_dim = args.bert_hidden_size,
                    # n_layers=1,
                    # bidirectional=False,
                    # dropout=0.5,
                    # args=args)

# class GRU(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, n_layers,
                 # bidirectional, dropout, args):
        # super().__init__()

        # self.rnn = nn.GRU(embedding_dim,
                           # hidden_dim,
                           # num_layers=n_layers,
                           # bidirectional=bidirectional,
                           # dropout=dropout,
                           # batch_first=True)
        # self.args = args

    # def forward(self, x):
        # output, hidden = self.rnn(x)
        # hidden = hidden.view(-1,self.args.bert_hidden_size)
        # output = output.view(-1,self.args.max_seq_length,self.args.bert_hidden_size)

        # return output,hidden


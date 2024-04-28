import torch
import copy
import math

import torch.nn as nn
import torch.nn.functional as F

# Set GPU Check 
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Set the CUDA device to use (e.g., GPU with index 0)
    device = torch.device("cuda")   # Now you can use CUDA operations
else:
    device = torch.device("cpu") 

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class ConditionalSublayerConnection(nn.Module):
    def __init__(self, gamma,beta,d_model=1024, dropout = 0.1, rm_num_slots=2, rm_d_model=1024): #rm_num_slots = 3
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(gamma,beta,d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, att_mask, sublayer, memory): 
        #print("x shape: ",x.shape)
        norm_states = self.norm(x,memory)
        #print('Norm States shape: ',norm_states.shape)
        # if layer_past: #Generation process
        #     #Retrieve only the last layer in last dim
        #     att_mask = att_mask[:,:,:,-1:]
        mha_output = sublayer(norm_states,att_mask)
        #print('Output of MHA: ',mha_output.shape)
        return x + self.dropout(mha_output)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h=16, d_model=1024, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        #print("Query Key Value sizes in attention: ",query.size(),key.size(),value.size())    
        d_k = key.size(-1) #query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            #print("Scores: ",scores.shape,scores.dtype)
            #print("Mask: ",mask.shape, mask.dtype)
            fill_value = torch.finfo(scores.dtype).min #0 #-1e4  #-1e9
            scores = torch.where(~mask, fill_value, scores)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None: 
            p_attn = dropout(p_attn)
        #print("p_attn :",p_attn.shape)
        #print("value: ",value.shape)
        '''
        Multiplication of 4d tensors A[a1,a2,a3,a4] and B[b1,b2,b3,b4]
        Then   a1 == b1 and a2 == b2
        Last 2 dims compatible for matmul ie a4 == b3
        '''
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None, present = None, layer_past = None): #layer_past: to det if gen process or not ; if present -> concatenate with  
        #print("Query Key Value sizes in MHA fwd: ",query.size(),key.size(),value.size())  

        # Use linear layer on the same device
        self.linears[-1] = self.linears[-1].to(query.device) 
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # if layer_past is not None: #Generation process - Further stages
        #     past_key, past_value = present
        #     #Remove last 2 layers of past_key and past_value along the 2nd last dim
        #     past_key = past_key[..., :-2, :]
        #     past_value = past_value[..., :-2, :]
        #     #Adds extra layers alongisde 2nd last dim only by 1 in each fwd call (like attn mask)
        #     key = torch.cat((past_key,key),dim=-2)
        #     value = torch.cat((past_value,value),dim=-2)
        #     #print("Key Value sizes in MHA fwd in gen process after adding key values in further generation: ",key.size(),value.size()) 

        if layer_past is not None: #Generation process - Further stages
            past_key, past_value = layer_past
            #Remove first 2 layers of past_key and past_value along the 2nd last dim
            past_key = past_key[..., 1:, :]
            past_value = past_value[..., 1:, :]
            #Adds extra layers alongisde 2nd last dim only by 1 in each fwd call (like attn mask)
            key = torch.cat((past_key,key),dim=-2)
            value = torch.cat((past_value,value),dim=-2)
            #print("Key Value sizes in MHA fwd in gen process after adding key values in further generation: ",key.size(),value.size()) 
        
        # if use_cache is True and layer_past is not None: #Update in further generations
        #     present = (key,value)
        # elif use_cache is True and layer_past is None: #Return previous present (from pseudo)
        #     present = present
        # else:
        #     present = None

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        #print("X shape in MHA: ",x.shape)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        #print("X shape after transpose: ",x.shape)
        l = self.linears[-1](x)
        return l

class ConditionalLayerNorm(nn.Module):
    def __init__(self, gamma, beta, d_model=1024, rm_num_slots=2, rm_d_model=1024, eps=1e-6): #rm_num_slots=3
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(gamma) #nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(beta) #nn.Parameter(torch.zeros(d_model))
        #print(self.gamma.shape, self.beta.shape)
        
        self.rm_d_model = rm_d_model 
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory): 
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory).permute(0, 1, 2)
        delta_beta = self.mlp_beta(memory).permute(0, 1, 2)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        # print(f"Shapes of tensors:\nGamma Hat {gamma_hat.shape}\nBeta Hat {beta_hat.shape}\nDelta Gamma {delta_gamma.shape}\nDelta Beta {delta_beta.shape}\n")
        # print(f"Mean : Mean Shape \n {mean.shape} \n Std Deviation : \n Std Dev Shape {std.shape}")
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        op = gamma_hat * (x - mean) / (std + self.eps) + beta_hat
        
        #file = open("checkRM/run20-120-sublayerconn.txt", 'a')  #File to write word hidden states during diff stages
        #file.write("\n-----------------------------------------------------------------------------------------------------------------------------------------------------------\n")  #SHORTENED FILE CHANGES
        #file.write(f"Shapes of tensors:\nGamma Hat {gamma_hat.shape}\nBeta Hat {beta_hat.shape}\nDelta Gamma {delta_gamma.shape}\nDelta Beta {delta_beta.shape}\n")
        #file.write(f"Mean : Mean Shape \n {mean.shape} \n Std Deviation : \n Std Dev Shape {std.shape}\n")
        #file.write("No of Nan in output tensor after CondLayerNorm: "+str(torch.sum(torch.isnan(op)).item())+"\n")
        #file.write("\n-----------------------------------------------------------------------------------------------------------------------------------------------------------\n")  #SHORTENED FILE CHANGES
        
        """with open('checkRM/example-ln_1-last.txt', 'a') as #file:
            #file.write("No of Nan in output tensor after CondLayerNorm: "+str(torch.sum(torch.isnan(op)).item())+"\n")"""
        return op
    
class RelationalMemory(nn.Module):

    def __init__(self, num_slots=2, d_model=1024, num_heads=8): #num_slots=3
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, device, batch_size):
        #comm
        eye_tensor = torch.eye(self.num_slots).to(device) 
        memory = torch.stack([eye_tensor] * batch_size).to(device)
      
        #memory = torch.stack([torch.eye(self.num_slots).to(device)])
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff)).to(device)
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory 

    def forward_step(self, input, memory):
        #Make sure initialzied tensors are in same device as memory
        self.attn = self.attn.to(memory.device) 
        self.mlp = self.mlp.to(memory.device) 
        self.W = self.W.to(memory.device)
        self.U = self.U.to(memory.device)  
        
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1) 
        #print(f"Devices: {memory.device}\n{q.device}\n{k.device}\n{v.device}")
        q, k, v = [t.to(memory.device) for t in (q, k, v)]

        attn = self.attn(q, k, v) #Past layers not taken into consideration
        next_memory = memory + attn
        ml_n = self.mlp(next_memory)
        next_memory = next_memory + ml_n

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)
        return next_memory

    def forward(self, inputs, memory):
        #inputs = inputs.to(memory.device)
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


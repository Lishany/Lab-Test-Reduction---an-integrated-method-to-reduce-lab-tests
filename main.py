import numpy as np
KKK = 1
main_device_id = 2
penalty = 1.5
#load training data and test data
data_dir = '../../Processed_data/'
TRAIN_vital_data = np.load(data_dir+'TRAIN_vital_data_all.npy')
TRAIN_test_data_fill = np.load(data_dir+'TRAIN_test_data.npy').astype('float')
TRAIN_visit_mask_all = np.load(data_dir+'TRAIN_visit_mask_all.npy')
TRAIN_visit_times_all = np.load(data_dir+'TRAIN_visit_times_all.npy')
TRAIN_abnormal_mask = np.load(data_dir+'TRAIN_abnormal_mask.npy')
TRAIN_not_nan_mask = np.load(data_dir+'TRAIN_not_nan_mask.npy')
TRAIN_person_f_list = np.load(data_dir+'TRAIN_person_f_list.npy')
TEST_vital_data = np.load(data_dir+'TEST_vital_data_all.npy')
TEST_test_data_all = np.load(data_dir+'TEST_test_data_all.npy',allow_pickle=True).astype('float')
TEST_visit_mask_all = np.load(data_dir+'TEST_visit_mask_all.npy')
TEST_visit_times_all = np.load(data_dir+'TEST_visit_times_all.npy')
TEST_abnormal_mask = np.load(data_dir+'TEST_abnormal_mask.npy')
TEST_not_nan_mask = np.load(data_dir+'TEST_not_nan_mask.npy')
TEST_person_f_list = np.load(data_dir+'TEST_person_f_list.npy')
print(len(TRAIN_vital_data),len(TEST_vital_data))
print(np.sum(TEST_visit_times_all==1))
valid_id = []
for i in range(len(TRAIN_visit_times_all)):
    if TRAIN_visit_times_all[i] >1:
        valid_id.append(i)
valid_id = np.array(valid_id)
TRAIN_vital_data = TRAIN_vital_data[valid_id]
TRAIN_test_data_fill = TRAIN_test_data_fill[valid_id]
TRAIN_visit_mask_all = TRAIN_visit_mask_all[valid_id]
TRAIN_visit_times_all = TRAIN_visit_times_all[valid_id]
TRAIN_abnormal_mask = TRAIN_abnormal_mask[valid_id]
TRAIN_not_nan_mask = TRAIN_not_nan_mask[valid_id]
TRAIN_person_f_list = TRAIN_person_f_list[valid_id]
valid_id = []
for i in range(len(TEST_visit_times_all)):
    if TEST_visit_times_all[i] >1:
        valid_id.append(i)
valid_id = np.array(valid_id)
TEST_vital_data = TEST_vital_data[valid_id]
TEST_test_data_all = TEST_test_data_all[valid_id]
TEST_visit_mask_all = TEST_visit_mask_all[valid_id]
TEST_visit_times_all = TEST_visit_times_all[valid_id]
TEST_abnormal_mask = TEST_abnormal_mask[valid_id]
TEST_not_nan_mask = TEST_not_nan_mask[valid_id]
TEST_person_f_list = TEST_person_f_list[valid_id]
print(len(TRAIN_person_f_list),len(TEST_person_f_list))
normal_range_M_F = [[[23,30],[7,20],[8.5,10.3],[97,107],[0.6,1.2],[13.5,17.5],[1.5,2.5],[2.5,4.5],[150,400],[3.6,5.2],[135,145],[4.5,11.0]],
                    [[23,30],[7,20],[8.5,10.3],[97,107],[0.5,1.1],[12.0,15.5],[1.5,2.5],[2.5,4.5],[150,400],[3.6,5.2],[135,145],[4.5,11.0]]]

normal_range_M_F = np.array(normal_range_M_F)
normal_range_M_F = normal_range_M_F.transpose(0,2,1)
scale_s = ((normal_range_M_F[:,1,:]-normal_range_M_F[:,0,:])/2)
mean_s = ((normal_range_M_F[:,1,:]+normal_range_M_F[:,0,:])/2)
ttt = TEST_test_data_all-np.mean(normal_range_M_F[TEST_person_f_list[:,0]],1).reshape(-1,1,12)
ttt[TEST_not_nan_mask==0]=0
TEST_test_data_all_rescale = ttt.copy()
ttt = TRAIN_test_data_fill-np.mean(normal_range_M_F[TRAIN_person_f_list[:,0]],1).reshape(-1,1,12)
ttt[TRAIN_not_nan_mask==0]=0
TRAIN_test_data_all_rescale = ttt

import torch
torch.cuda.is_available()
#WINDOW = 5
TEST_NUM = 12
import os
if torch.cuda.is_available():
    is_cuda = True
else:
    is_cuda = False 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

device_ids = [main_device_id]
torch.cuda.set_device(main_device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device_ids = [4]
if not is_cuda:
    device = torch.device("cpu")
    device_ids = []
print(device)
normal_range_M_F = [[[23,30],[7,20],[8.5,10.3],[97,107],[0.6,1.2],[13.5,17.5],[1.5,2.5],[2.5,4.5],[150,400],[3.6,5.2],[135,145],[4.5,11.0]],
                    [[23,30],[7,20],[8.5,10.3],[97,107],[0.5,1.1],[12.0,15.5],[1.5,2.5],[2.5,4.5],[150,400],[3.6,5.2],[135,145],[4.5,11.0]]]

normal_range_M_F = torch.FloatTensor(normal_range_M_F).to(device)
normal_range_M_F,normal_range_M_F.size()

normal_range_M_F = normal_range_M_F.transpose(1,2)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
scale_contant = (1.0/torch.FloatTensor(np.mean(scale_s,0))).to(device)

#model 

class TrainDataset(Dataset):
    def __init__(self, TRAIN_vital_data, TRAIN_test_data_fill,TRAIN_visit_mask_all,TRAIN_visit_times_all, TRAIN_abnormal_mask,TRAIN_not_nan_mask,CUT_LEN,TRAIN_person_f_list, TEST_NUM):
        self.ALLCOUNT = len(TRAIN_test_data_fill)
        self.TRAIN_vital_data = torch.FloatTensor(TRAIN_vital_data)
        
        self.TRAIN_test_data_fill = torch.FloatTensor(TRAIN_test_data_fill)
        self.TRAIN_visit_mask_all = torch.FloatTensor(TRAIN_visit_mask_all)
        self.TRAIN_visit_times_all = torch.FloatTensor(TRAIN_visit_times_all)
        self.TRAIN_abnormal_mask = torch.FloatTensor(TRAIN_abnormal_mask)
        self.TRAIN_not_nan_mask = torch.FloatTensor(TRAIN_not_nan_mask)
        self.TRAIN_person_f_list = torch.LongTensor(TRAIN_person_f_list)
        self.CUT_LEN = CUT_LEN
        self.TEST_NUM = TEST_NUM
        self.Miss_vital_data = torch.FloatTensor(TRAIN_vital_data)
        self.Miss_vital_data[:,:,:6][self.Miss_vital_data[:,:,:6]>0] = 1
        self.Miss_vital_data[:,:,6:] = self.Miss_vital_data[:,:,:6]
        self.Miss_vital_data = 1- self.Miss_vital_data
        self.Miss_vital_data[self.TRAIN_visit_mask_all==0]=0
        self.TRAIN_nan_mask = torch.FloatTensor(TRAIN_not_nan_mask)
        self.TRAIN_nan_mask = 1- self.TRAIN_nan_mask
        self.TRAIN_nan_mask[self.TRAIN_visit_mask_all==0] = 0

    def __len__(self):
        return self.ALLCOUNT

    def __getitem__(self, idx):
        vital_data = self.TRAIN_vital_data[idx]
        miss_vital_data = self.Miss_vital_data[idx]
        input_data = self.TRAIN_test_data_fill[idx]
        visit_mask = self.TRAIN_visit_mask_all[idx]
        visit_times = self.TRAIN_visit_times_all[idx]
        abnormal_mask = self.TRAIN_abnormal_mask[idx]
        not_nan_mask = self.TRAIN_not_nan_mask[idx]
        nan_mask = self.TRAIN_nan_mask[idx]
        pfeature_batch = self.TRAIN_person_f_list[idx]
        sample = {'vital_batch':vital_data,'miss_vital_mask':miss_vital_data,'nan_mask':nan_mask,'input_data': input_data, 'visit_mask': visit_mask, 'visit_times': visit_times,'abnormal_mask': abnormal_mask, 'not_nan_mask': not_nan_mask, 'pfeature_batch':pfeature_batch}
        return sample
    
class Simple_model(nn.Module):
    def __init__(self,vital_num, window_cnn,person_fnum,fnum_NUM,input_dim, hidden_dim1,hidden_dim2, test_num, dropout_prob, nlay1 = 1, nlay2 = 1,uniform_init = False):
        super(Simple_model, self).__init__()
        self.window_cnn = window_cnn
        self.test_num = test_num
        self.vital_num = vital_num
        self.hidden_cnn = test_num*4
        self.person_fnum = person_fnum
        self.fnum_NUM = fnum_NUM
        self.emb_person_dim = 4
        self.person_emb = nn.Embedding(fnum_NUM,self.emb_person_dim)
        self.cnn_gen_nan = nn.Conv2d(2,self.hidden_cnn,(window_cnn*2+1,test_num+self.vital_num),stride = 1,padding = (window_cnn,0))
        self.cnn_gen_nan2 = nn.Conv2d(1,self.hidden_cnn,(test_num*4,window_cnn*2+1),stride = 1,padding = (0,window_cnn))
        self.linear_cnn = nn.Linear(self.hidden_cnn+person_fnum*self.emb_person_dim, test_num)
        
        window_cnn_2 = 2
        self.window_cnn_2 = window_cnn_2
        
        self.lstm_gen_prob = nn.LSTM(test_num*2+self.vital_num*2, hidden_dim1, num_layers= nlay1, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim1+person_fnum*self.emb_person_dim, test_num)
        self.lstm_gen_next_visit = nn.LSTM(test_num*2+self.vital_num*2, hidden_dim2,num_layers = nlay2, batch_first=True)        
        self.linear2 = nn.Linear(hidden_dim2+person_fnum*self.emb_person_dim, test_num)
        
        if uniform_init:
            nn.init.xavier_uniform_(self.person_emb.weight)
            nn.init.xavier_uniform_(self.cnn_gen_nan.weight)
            nn.init.xavier_uniform_(self.cnn_gen_nan2.weight)
            nn.init.xavier_uniform_(self.linear_cnn.weight)
            
            for names in self.lstm_gen_prob._all_weights:
                for name in filter(lambda n: "weight" in n, names):
                    weight = getattr(self.lstm_gen_prob, name)
                    nn.init.xavier_uniform_(weight)
            for names in self.lstm_gen_next_visit._all_weights:
                for name in filter(lambda n: "weight" in n, names):
                    weight = getattr(self.lstm_gen_next_visit, name)
                    nn.init.xavier_uniform_(weight)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, vital_input_data,miss_mask,nan_mask,batch_input_data,nnan_mask,person_data_id,colla_prob1 = 0.0,colla_prob2 = 0.0): #person_data_id (B,person_fnum)
        person_data = self.person_emb(person_data_id) # (B,person_fnum, 2)
        person_data = person_data.view(person_data.size()[0],-1)
        person_data = torch.Tensor.repeat(person_data.unsqueeze(1),[1,batch_input_data.size()[1],1])
        batch_input_data_colla = batch_input_data.clone()
        nan_mask0 = nan_mask.clone()
        temp_input = torch.cat([batch_input_data_colla,vital_input_data],2)
        mask_input = torch.cat([nan_mask0,miss_mask],2)
        temp_input = torch.stack([temp_input,mask_input],1)
        
        
        cnn_pre = self.cnn_gen_nan(temp_input).squeeze(3) #B * testnum*4 * len_visit * 1
        cnn_pre = torch.relu(cnn_pre)
        cnn_pre = self.cnn_gen_nan2(cnn_pre.unsqueeze(1)).squeeze(2) # B * testnum*4 * 1 * len_visit
        cnn_pre = torch.relu(cnn_pre)
        cnn_pre = cnn_pre.transpose(1,2) # B * len_visit * testnum*4
        
        cnn_pre = torch.cat([person_data,cnn_pre],2)
        
        cnn_pre = self.linear_cnn(cnn_pre) # B * len_visit * TEST_NUM
        loss2 = torch.sum(((cnn_pre-batch_input_data)*scale_contant)[nnan_mask>0]**2)/torch.sum(nnan_mask>0)
        
        cnn_pre[nnan_mask>0] = batch_input_data[nnan_mask>0]
        
        temp_input = torch.cat([cnn_pre,vital_input_data],2)
        mask_input = torch.cat([nan_mask,miss_mask],2)
        cnn_pre_new = torch.cat([temp_input,mask_input],2)
        
        cnn_pre_next,_ = self.lstm_gen_next_visit(cnn_pre_new)
        cnn_pre_next = torch.cat([person_data,cnn_pre_next],2)
        next_temp0 = self.linear2(cnn_pre_next)
        prob_temp, _ = self.lstm_gen_prob(cnn_pre_new)
        prob_temp = torch.cat([person_data,prob_temp],2)
        prob_temp = self.linear1(prob_temp)
        
        prob_temp = torch.sigmoid(KKK*prob_temp)
        new_input_batch = torch.cat([cnn_pre[:,:1,:],((1-prob_temp)*next_temp0)[:,:-1,:]+(prob_temp[:,:-1,:])*cnn_pre[:,1:,:]],1)
        
        temp_input = torch.cat([new_input_batch,vital_input_data],2)
        prob_mask = torch.cat([nan_mask[:,:1,:],1-prob_temp[:,:-1,:]],1)
        mask_input = torch.cat([prob_mask,miss_mask],2)
        new_input_batch = torch.cat([temp_input,mask_input],2)  
        
        
        next_temp_mix,_ = self.lstm_gen_next_visit(new_input_batch)
        next_temp_mix = torch.cat([person_data,next_temp_mix],2)
        next_temp_mix = self.linear2(next_temp_mix)
        
        loss = torch.sum(((next_temp0-next_temp_mix)*scale_contant)**2)/next_temp0.size()[0]
        return((prob_temp, next_temp_mix,loss,loss2))
    
    def get_cnn_result(self, batch_input_data,person_data_id,vital_input_data,miss_mask,nan_mask): #person_data_id (B,person_fnum)
        person_data = self.person_emb(person_data_id) # (B,person_fnum, 2)
        person_data = person_data.view(person_data.size()[0],-1)
        person_data = torch.Tensor.repeat(person_data.unsqueeze(1),[1,batch_input_data.size()[1],1])
        
        temp_input = torch.cat([batch_input_data,vital_input_data],2)
        mask_input = torch.cat([nan_mask,miss_mask],2)
        temp_input = torch.stack([temp_input,mask_input],1)
        
        cnn_pre = self.cnn_gen_nan(temp_input).squeeze(3) #B * testnum*4 * len_visit * 1
        cnn_pre = torch.relu(cnn_pre)
        cnn_pre = self.cnn_gen_nan2(cnn_pre.unsqueeze(1)).squeeze(2) # B * testnum*4 * 1 * len_visit
        cnn_pre = torch.relu(cnn_pre)
        cnn_pre = cnn_pre.transpose(1,2) # B * len_visit * testnum*4
        
        cnn_pre = torch.cat([person_data,cnn_pre],2)
        
        cnn_pre = self.linear_cnn(cnn_pre) # B * len_visit * TEST_NUM
        return(cnn_pre)
    
    def get_next(self,batch_input_data,prob_mask,vital_input_data,miss_mask,person_data_id):
        person_data = self.person_emb(person_data_id) # (B,person_fnum, 2)
        person_data = person_data.view(person_data.size()[0],-1)
        person_data = torch.Tensor.repeat(person_data.unsqueeze(1),[1,batch_input_data.size()[1],1])
        
        temp_input = torch.cat([batch_input_data,vital_input_data],2)
        mask_input = torch.cat([prob_mask,miss_mask],2)
        cnn_pre_new = torch.cat([temp_input,mask_input],2)
        
        cnn_pre_next,_ = self.lstm_gen_next_visit(cnn_pre_new)
        cnn_pre_next = torch.cat([person_data,cnn_pre_next],2)
        next_temp0 = self.linear2(cnn_pre_next)
        prob_temp, _ = self.lstm_gen_prob(cnn_pre_new)
        prob_temp = torch.cat([person_data,prob_temp],2)
        prob_temp = self.linear1(prob_temp)
        
        prob_temp = torch.sigmoid(KKK*prob_temp)
        
        return(prob_temp,next_temp0)

scale_s_tensor = torch.FloatTensor(scale_s).to(device)

BATCHSIZE = 512
nw_all = 0
CUT_LEN = 30
train_dataset = TrainDataset(TRAIN_vital_data,TRAIN_test_data_all_rescale,TRAIN_visit_mask_all,TRAIN_visit_times_all, TRAIN_abnormal_mask,TRAIN_not_nan_mask,CUT_LEN,TRAIN_person_f_list, TEST_NUM)
train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE,shuffle=True, num_workers=nw_all,drop_last=True,pin_memory = True)

test_dataset = TrainDataset(TEST_vital_data,TEST_test_data_all_rescale,TEST_visit_mask_all,TEST_visit_times_all,TEST_abnormal_mask,TEST_not_nan_mask,CUT_LEN,TEST_person_f_list, TEST_NUM)
test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE*4,shuffle=False, num_workers=nw_all,drop_last=False,pin_memory = True)
input_dim= TEST_NUM
person_fnum = 2
window_cnn = 1
vital_num = 12
fnum_NUM = 43
hidden_dim1=50
hidden_dim2=50
test_num=TEST_NUM
dropout_prob=0.0
nlay1 = 2
nlay2 = 2
uniform_init = True
b1 = 1000
b2 = 100
learning_rate= 0.0001
lambda_weight_decay = 0.000001
NUM_EPOCH = 10001
recal_k = 10
n_print = 30

simplemodel=Simple_model(vital_num,window_cnn,person_fnum,fnum_NUM,input_dim, hidden_dim1,hidden_dim2, test_num, dropout_prob, nlay1, nlay2,uniform_init).to(device)
optimizer = torch.optim.Adam(simplemodel.parameters(), lr=learning_rate, weight_decay=lambda_weight_decay)
simplemodel.train()

#load imputation model
model_temp = torch.load('./imputation_model4.pkl')
state_dict_temp = {}
for x in simplemodel.state_dict():
    #if x in recover_model.state_dict():
    if x in model_temp:
        if 'next' in str(x):
            state_dict_temp[x] = simplemodel.state_dict()[x].data
        else:
            state_dict_temp[x] = model_temp[x].data
    else:
        state_dict_temp[x] = simplemodel.state_dict()[x].data
simplemodel.load_state_dict(state_dict_temp)
simplemodel.person_emb.weight.requires_grad = False
simplemodel.cnn_gen_nan.weight.requires_grad= False
simplemodel.cnn_gen_nan.bias.requires_grad= False
simplemodel.cnn_gen_nan2.weight.requires_grad= False
simplemodel.cnn_gen_nan2.bias.requires_grad= False
simplemodel.linear_cnn.weight.requires_grad= False
simplemodel.linear_cnn.bias.requires_grad= False


def return_loss(batch_input_data,gender_batch, prob_temp,next_temp,visit_mask, ab_mask, nnan_mask, b1,b2):
    visit_times_all = torch.sum(visit_mask[:,1:])
    visit_mask = (visit_mask[:,1:]>0)
    
    prob_temp = prob_temp[:,:-1,:]
    scale_temp = scale_s_tensor[gender_batch]
    mean_temp = torch.FloatTensor(mean_s).to(device)[gender_batch]
    
    next_temp = next_temp[:,:-1,:]
    ab_mask = ab_mask[:,1:,:]
    nnan_mask = nnan_mask[:,1:,:]
    batch_input_data_temp = batch_input_data[:,1:,:]
    
    range_temp = torch.abs(batch_input_data_temp) - scale_temp.unsqueeze(1) 
    range_temp = range_temp[visit_mask]
    
    left_range_temp = (batch_input_data_temp - scale_temp.unsqueeze(1))/2.0
    left_range_temp = left_range_temp[visit_mask]
    right_range_temp = (batch_input_data_temp + scale_temp.unsqueeze(1))/2.0
    right_range_temp = right_range_temp[visit_mask]
    
    diff = torch.abs(next_temp-batch_input_data_temp)
    diff = diff[visit_mask]
    next_temp = next_temp[visit_mask]
    prob_temp = prob_temp[visit_mask]
    nnan_mask = nnan_mask[visit_mask]
    ab_mask = ab_mask[visit_mask]
       
    signal = torch.relu((next_temp - left_range_temp)*(next_temp-right_range_temp))*torch.relu(-range_temp)
    signal[signal>0] = 1 
    diff2 = torch.max(diff, torch.abs(range_temp)/3) *ab_mask
    diff2_temp = torch.relu(diff-torch.abs(range_temp)/3) *ab_mask
    diff2_temp[diff2_temp>0] = 1
    
    wrong_temp = signal+ diff2_temp
    
    diff9 = wrong_temp*nnan_mask
    
    diff_final = (diff*scale_contant)**2*signal + (diff2*scale_contant)**2    
    
    diff_final = (5*(diff_final+1)*((1-prob_temp))+ (diff_final))*diff9 
      
    diff_final_final = diff_final[nnan_mask>0]
    loss_3 = -torch.sum(torch.log(1.000001-prob_temp))
    
    loss_3 = loss_3/visit_times_all 
    loss_4 = torch.sum(diff_final_final)/(1+diff_final_final.size()[0])
    return(loss_3,loss_4)

#model training
NUM_EPOCH = 10001
n_print = 100
point = 0.5
for n_epoch in range(NUM_EPOCH):
    simplemodel.train()
    torch.cuda.empty_cache()
    train_cost = 0
    input_iter = iter(train_dataloader)
    for i_train, TV_samples in enumerate(input_iter,0):
        batch_input_data,pfeature_batch, visit_mask, ab_mask, nnan_mask = TV_samples['input_data'].to(device),TV_samples['pfeature_batch'].to(device), TV_samples['visit_mask'].to(device), TV_samples['abnormal_mask'].to(device), TV_samples['not_nan_mask'].to(device)
        nan_mask,vital_batch,vital_miss_mask = TV_samples['nan_mask'].to(device),TV_samples['vital_batch'].to(device),TV_samples['miss_vital_mask'].to(device)
        
        optimizer.zero_grad()
        prob_temp, next_temp,loss_1,loss_2 = simplemodel(vital_batch,vital_miss_mask,nan_mask,batch_input_data,nnan_mask,pfeature_batch)
        #print(batch_input_data[0][1:3])
        #print(next_temp[0][:2])
        loss_3,loss_4= return_loss(batch_input_data,pfeature_batch[:,0], prob_temp,next_temp,visit_mask, ab_mask, nnan_mask, b1,b2)
        loss = (penalty*loss_3+10*loss_4+ 8*loss_1+0*loss_2).mean()
        if i_train%60==0 and n_epoch % n_print == 0:
            print(i_train,loss_1.data.item(),loss_2.data.item(),loss_3.data.item(),loss_4.data.item())
        #print(loss_5.data)
        #print(loss)
        train_cost += loss.data
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        #break
    #print(n_epoch,'train cost: ',train_cost.item()/(i_train+1))
    if n_epoch % n_print == 0:
        simplemodel.eval()
        print('-'*50)
        print(n_epoch,'train cost: ',train_cost.item()/(i_train+1))
        
        #loss_all = []
        true_ab_count = 0
        nnan_count = 0
        pre_do_count = 0
        pre_do_on_nnan_count = 0
        pre_ab_on_all_count = 0
        pre_ab_on_nan_count = 0
        pre_ab_p1_t1_on_nnan_count = 0
        notdoing_p0_t1_on_nan_count = 0
        notdoing_p1_t0_on_nan_count = 0
        notdoing_p_same_t_on_nan_count = 0
        notdoing_p1_t1_on_nan_count = 0
        bias_test_mean = torch.zeros([1,TEST_NUM])
        bias_test_mean_sum = torch.zeros([1,TEST_NUM])
        bias_test_mean_not = torch.zeros([1,TEST_NUM])
        bias_test_mean_sum_not = torch.zeros([1,TEST_NUM])
        for i_train, TV_samples in enumerate(test_dataloader,0):
            #batch_input_data, visit_mask,visit_times, ab_mask, nnan_mask = TV_samples['input_data'].to(device), TV_samples['visit_mask'].to(device),TV_samples['visit_times'].to(device), TV_samples['abnormal_mask'].to(device), TV_samples['not_nan_mask'].to(device)
            batch_input_data, pfeature_batch, visit_mask, visit_times, ab_mask, nnan_mask = TV_samples['input_data'],TV_samples['pfeature_batch'].to(device), TV_samples['visit_mask'],TV_samples['visit_times'], TV_samples['abnormal_mask'], TV_samples['not_nan_mask'].to(device)
            
            nan_mask,vital_batch,vital_miss_mask = TV_samples['nan_mask'].to(device),TV_samples['vital_batch'].to(device),TV_samples['miss_vital_mask'].to(device)
            first_temp = simplemodel.get_cnn_result(batch_input_data[:,:1].to(device),(pfeature_batch),vital_batch[:,:1],vital_miss_mask[:,:1],nan_mask[:,:1])
            
            batch_input_data = batch_input_data.to(device)
            start_batch = batch_input_data.clone()[:,:1,:]
            start_batch[start_batch==0] = first_temp[start_batch==0]
            temp_batch = batch_input_data.clone()[:,:1,:]
            temp_batch[temp_batch==0] = first_temp[temp_batch==0]
            nnan_mask_next = nnan_mask[:,1:,:].to(device)
            prob_nnan_temp = nan_mask[:,:1,:].clone()
            
            temp_prob = []
            temp_next_list = []
            #loss = []
            for j in range(CUT_LEN-1):
                #prob_temp_test,next_temp = simplemodel.get_next(temp_batch,prob_nnan_temp,pfeature_batch)
                prob_temp_test,next_temp = simplemodel.get_next(temp_batch,prob_nnan_temp,vital_batch[:,:(j+1)],vital_miss_mask[:,:(j+1)],pfeature_batch)
                prob_temp0 = prob_temp_test[:,-1:,:].data.clone()
                temp_prob.append(prob_temp_test[:,-1,:])
                temp_next_list.append(next_temp[:,-1,:].data.clone())
                prob_temp0[prob_temp0<point] = 0
                prob_temp0 = (nnan_mask_next[:,j:(j+1),:])*prob_temp0 #need do
                prob_temp0[prob_temp0>0] = 1
                new_next = (1-prob_temp0)*next_temp[:,-1:,:]+(prob_temp0)*batch_input_data.clone()[:,(j+1):(j+2),:]
                temp_batch = torch.cat([temp_batch,new_next],1)
                prob_nnan_temp = torch.cat([prob_nnan_temp,1-prob_temp0],1)
             
            prob_temp_test,next_temp = simplemodel.get_next(temp_batch,prob_nnan_temp,vital_batch,vital_miss_mask,pfeature_batch)

            temp_prob.append(prob_temp_test[:,-1,:])
            temp_next_list.append(next_temp[:,-1,:].data.clone())
            
            final_prob = torch.zeros(prob_temp_test.size())
            for j in range(final_prob.size()[1]):
                final_prob[:,j] = temp_prob[j]
            final_next = torch.zeros(next_temp.size())
            for j in range(final_next.size()[1]):
                final_next[:,j] = temp_next_list[j]
            
            prob_temp_test = final_prob.cpu()
            next_temp = final_next.cpu()
                
            
            visit_mask = visit_mask[:,1:] >0
            
            pre_do = prob_temp_test[:,:-1,:].data.cpu()[visit_mask]
            pre_do[pre_do>=point] = 1
            pre_do[pre_do<point] = 0
            
            pre_do_tempsum = torch.sum(pre_do)
            pre_do_count += pre_do_tempsum.item()
            nnan = nnan_mask[:,1:,:].data[visit_mask]
            
            pre_do_on_nnan_count += torch.sum(pre_do[nnan>0]).item()
            
            true_ab = ab_mask[:,1:,:].data[visit_mask]
            
            true_ab_tempsum = torch.sum(true_ab)
            true_ab_count += true_ab_tempsum.item()
            
            nnan_tempsum = torch.sum(nnan)
            nnan_count += nnan_tempsum.item()
            
            next_temp_ab = next_temp.data.cpu()[:,:-1,:]
            
            
            scale_temp = scale_s_tensor[pfeature_batch[:,0]].cpu()
            next_temp_ab = torch.abs(next_temp_ab) - scale_temp.unsqueeze(1)
            next_temp_ab[next_temp_ab<=0] = 0
            next_temp_ab[next_temp_ab>0] = 1
            next_temp_ab = next_temp_ab[visit_mask]
            
            pre_ab_on_all_count += torch.sum(next_temp_ab).item()
            pre_ab_on_nan_count += torch.sum(next_temp_ab[nnan>0]).item()
            
            pre_ab_ab = next_temp_ab*true_ab
            pre_ab_ab = pre_ab_ab[nnan>0]
            
            pre_ab_p1_t1_on_nnan_count +=torch.sum(pre_ab_ab).item()
            
            pre_ab_same = next_temp_ab-true_ab
            pre_ab_same_on_notdoing = pre_ab_same[nnan>0][pre_do[nnan>0]==0]
            
            notdoing_p0_t1_on_nan_count += torch.sum(pre_ab_same_on_notdoing==-1).item()
            notdoing_p1_t0_on_nan_count += torch.sum(pre_ab_same_on_notdoing==1).item()
            notdoing_p_same_t_on_nan_count += torch.sum(pre_ab_same_on_notdoing==0).item()
            notdoing_p1_t1_on_nan_count += torch.sum(pre_ab_ab[pre_do[nnan>0]==0]).item()
            
            tempp = (next_temp[:,:-1,:].data+torch.FloatTensor(mean_s)[pfeature_batch[:,0].cpu()].unsqueeze(1))[visit_mask]
            tempp_mask = nnan_mask[:,1:,:][visit_mask].cpu()
            #print(tempp.size())
            #print(tempp[:3])
            tempp_not = (1-pre_do)*tempp
            tempp_mask_not = (1-pre_do)*tempp_mask
            tempp_not[tempp_mask_not==0] = 0
            
            tempp[tempp_mask==0]=0
            bias_test_mean += torch.sum(tempp,0)
            bias_test_mean_sum += torch.sum(tempp_mask,0)
            bias_test_mean_not += torch.sum(tempp_not,0)
            bias_test_mean_sum_not += torch.sum(tempp_mask_not,0)
            
            #break
        bias_test_mean_sum[bias_test_mean_sum==0] = 1
        bias_test_mean = bias_test_mean/bias_test_mean_sum
        bias_test_mean_sum_not[bias_test_mean_sum_not==0] = 1
        bias_test_mean_not = bias_test_mean_not/bias_test_mean_sum_not
        
        print(bias_test_mean)
        print(bias_test_mean_not)
        print('true_ab_count',true_ab_count)
        print('nnan_count',nnan_count)
        print('pre_do_count',pre_do_count)
        print('pre_do_on_nnan_count',pre_do_on_nnan_count)
        print('pre_ab_on_all_count',pre_ab_on_all_count)
        print('pre_ab_on_nan_count',pre_ab_on_nan_count)
        print('pre_ab_p1_t1_on_nnan_count',pre_ab_p1_t1_on_nnan_count)
        print('all_p0t0_p0t1_p1t0_p1t1',nnan_count-pre_ab_on_nan_count-true_ab_count+pre_ab_p1_t1_on_nnan_count,true_ab_count-pre_ab_p1_t1_on_nnan_count,pre_ab_on_nan_count-pre_ab_p1_t1_on_nnan_count,pre_ab_p1_t1_on_nnan_count)
        print('notdoing_p0t0_p0t1_p1t0_p1t1',notdoing_p_same_t_on_nan_count-notdoing_p1_t1_on_nan_count,notdoing_p0_t1_on_nan_count,notdoing_p1_t0_on_nan_count,notdoing_p1_t1_on_nan_count)
        print(pre_do_on_nnan_count/nnan_count)
        if notdoing_p_same_t_on_nan_count+notdoing_p0_t1_on_nan_count+notdoing_p1_t0_on_nan_count != 0:
            print(notdoing_p_same_t_on_nan_count/(notdoing_p_same_t_on_nan_count+notdoing_p0_t1_on_nan_count+notdoing_p1_t0_on_nan_count))
        else:
            print(1.0)
        print(1-(notdoing_p0_t1_on_nan_count+notdoing_p1_t0_on_nan_count)/nnan_count)
print('--END--')



#Test
#test response 

#choose point
bbb = [0.000001,0.00001,0.0001,0.001,0.01]+list(np.arange(1,21)/20)
#choose point
temp_ab_test = []
temp_ab_test_seperate = []
for point in bbb:
    simplemodel.eval()
    print(point)
    #loss_all = []
    true_ab_count = 0
    nnan_count = 0
    pre_do_count = 0
    pre_do_on_nnan_count = 0
    pre_ab_on_all_count = 0
    pre_ab_on_nan_count = 0
    pre_ab_p1_t1_on_nnan_count = 0
    notdoing_p0_t1_on_nan_count = 0
    notdoing_p1_t0_on_nan_count = 0
    notdoing_p_same_t_on_nan_count = 0
    notdoing_p1_t1_on_nan_count = 0
    bias_test_mean = torch.zeros([1,TEST_NUM])
    bias_test_mean_sum = torch.zeros([1,TEST_NUM])
    bias_test_mean_not = torch.zeros([1,TEST_NUM])
    bias_test_mean_sum_not = torch.zeros([1,TEST_NUM])
    pre_ab_same_12_all = []
    visit_mask_all = []
    for i_train, TV_samples in enumerate(test_dataloader,0):
        #batch_input_data, visit_mask,visit_times, ab_mask, nnan_mask = TV_samples['input_data'].to(device), TV_samples['visit_mask'].to(device),TV_samples['visit_times'].to(device), TV_samples['abnormal_mask'].to(device), TV_samples['not_nan_mask'].to(device)
        batch_input_data, pfeature_batch, visit_mask, visit_times, ab_mask, nnan_mask = TV_samples['input_data'],TV_samples['pfeature_batch'].to(device), TV_samples['visit_mask'],TV_samples['visit_times'], TV_samples['abnormal_mask'], TV_samples['not_nan_mask'].to(device)

        nan_mask,vital_batch,vital_miss_mask = TV_samples['nan_mask'].to(device),TV_samples['vital_batch'].to(device),TV_samples['miss_vital_mask'].to(device)
        first_temp = simplemodel.get_cnn_result(batch_input_data[:,:1].to(device),(pfeature_batch),vital_batch[:,:1],vital_miss_mask[:,:1],nan_mask[:,:1])

        batch_input_data = batch_input_data.to(device)
        start_batch = batch_input_data.clone()[:,:1,:]
        start_batch[start_batch==0] = first_temp[start_batch==0]
        temp_batch = batch_input_data.clone()[:,:1,:]
        temp_batch[temp_batch==0] = first_temp[temp_batch==0]
        nnan_mask_next = nnan_mask[:,1:,:].to(device)
        prob_nnan_temp = nan_mask[:,:1,:].clone()

        temp_prob = []
        temp_next_list = []
        #loss = []
        for j in range(CUT_LEN-1):
            #prob_temp_test,next_temp = simplemodel.get_next(temp_batch,prob_nnan_temp,pfeature_batch)
            prob_temp_test,next_temp = simplemodel.get_next(temp_batch,prob_nnan_temp,vital_batch[:,:(j+1)],vital_miss_mask[:,:(j+1)],pfeature_batch)
            prob_temp0 = prob_temp_test[:,-1:,:].data.clone()
            temp_prob.append(prob_temp_test[:,-1,:])
            temp_next_list.append(next_temp[:,-1,:].data.clone())
            prob_temp0[prob_temp0<point] = 0
            prob_temp0 = (nnan_mask_next[:,j:(j+1),:])*prob_temp0 #need do
            prob_temp0[prob_temp0>0] = 1
            new_next = (1-prob_temp0)*next_temp[:,-1:,:]+(prob_temp0)*batch_input_data.clone()[:,(j+1):(j+2),:]
            #new_next = torch.cat([person_data,new_next],2)
            #print('new_next',new_next)
            temp_batch = torch.cat([temp_batch,new_next.clone()],1)
            prob_nnan_temp = torch.cat([prob_nnan_temp,1-prob_temp0],1)

        #prob_temp_test,next_temp = simplemodel.get_next(temp_batch,prob_nnan_temp,pfeature_batch)
        prob_temp_test,next_temp = simplemodel.get_next(temp_batch,prob_nnan_temp,vital_batch,vital_miss_mask,pfeature_batch)

        temp_prob.append(prob_temp_test[:,-1,:])
        temp_next_list.append(next_temp[:,-1,:].data.clone())

        final_prob = torch.zeros(prob_temp_test.size())
        for j in range(final_prob.size()[1]):
            final_prob[:,j] = temp_prob[j]
        final_next = torch.zeros(next_temp.size())
        for j in range(final_next.size()[1]):
            final_next[:,j] = temp_next_list[j]

        prob_temp_test = final_prob.cpu()
        next_temp = final_next.cpu()


        visit_mask = visit_mask[:,1:] >0

        pre_do = prob_temp_test[:,:-1,:].data.cpu()[visit_mask]
        pre_do[pre_do>=point] = 1
        pre_do[pre_do<point] = 0

        pre_do_tempsum = torch.sum(pre_do)
        pre_do_count += pre_do_tempsum.item()
        nnan = nnan_mask[:,1:,:].data[visit_mask]

        pre_do_on_nnan_count += torch.sum(pre_do[nnan>0]).item()

        true_ab = ab_mask[:,1:,:].data[visit_mask]

        true_ab_tempsum = torch.sum(true_ab)
        true_ab_count += true_ab_tempsum.item()

        nnan_tempsum = torch.sum(nnan)
        nnan_count += nnan_tempsum.item()

        next_temp_ab = next_temp.data.cpu()[:,:-1,:]


        scale_temp = scale_s_tensor[pfeature_batch[:,0]].cpu()
        next_temp_ab = torch.abs(next_temp_ab) - scale_temp.unsqueeze(1)
        next_temp_ab[next_temp_ab<=0] = 0
        next_temp_ab[next_temp_ab>0] = 1
        next_temp_ab = next_temp_ab[visit_mask]

        pre_ab_on_all_count += torch.sum(next_temp_ab).item()
        pre_ab_on_nan_count += torch.sum(next_temp_ab[nnan>0]).item()

        pre_ab_ab = next_temp_ab*true_ab
        pre_ab_ab = pre_ab_ab[nnan>0]

        pre_ab_p1_t1_on_nnan_count +=torch.sum(pre_ab_ab).item()

        pre_ab_same = next_temp_ab-true_ab
        pre_ab_same_on_notdoing = pre_ab_same[nnan>0][pre_do[nnan>0]==0]
        pre_ab_same_12 = pre_ab_same.clone()
        pre_ab_same_12[pre_do == 1] = -1000
        pre_ab_same_12[nnan==0] = 1000
        pre_ab_same_12_all.append(pre_ab_same_12)
        visit_mask_all.append(visit_mask)
        #visit_mask

        

        notdoing_p0_t1_on_nan_count += torch.sum(pre_ab_same_on_notdoing==-1).item()
        notdoing_p1_t0_on_nan_count += torch.sum(pre_ab_same_on_notdoing==1).item()
        notdoing_p_same_t_on_nan_count += torch.sum(pre_ab_same_on_notdoing==0).item()
        notdoing_p1_t1_on_nan_count += torch.sum(pre_ab_ab[pre_do[nnan>0]==0]).item()

        #tempp = (next_temp[:,:-1,:].data+torch.FloatTensor(mean_s)[pfeature_batch[:,0].cpu()].unsqueeze(1))[visit_mask]
        tempp = (torch.abs(next_temp[:,:-1,:].data-batch_input_data.cpu()[:,1:,:]).data)[visit_mask]
        tempp_mask = nnan_mask[:,1:,:][visit_mask].cpu()
        #print(tempp.size())
        #print(tempp[:3])
        tempp_not = (1-pre_do)*tempp
        tempp_mask_not = (1-pre_do)*tempp_mask
        tempp_not[tempp_mask_not==0] = 0

        tempp[tempp_mask==0]=0
        bias_test_mean += torch.sum(tempp,0)
        bias_test_mean_sum += torch.sum(tempp_mask,0)
        bias_test_mean_not += torch.sum(tempp_not,0)
        bias_test_mean_sum_not += torch.sum(tempp_mask_not,0)

        #break
    pre_ab_same_12_all = torch.cat(pre_ab_same_12_all,0)
    visit_mask_all = torch.cat(visit_mask_all,0)
    temp_ab_test_seperate.append(torch.stack([torch.sum(pre_ab_same_12_all==0,0),torch.sum(torch.abs(pre_ab_same_12_all)<10,0)],0))
    bias_test_mean_sum[bias_test_mean_sum==0] = 1
    bias_test_mean = bias_test_mean/bias_test_mean_sum
    bias_test_mean_sum_not[bias_test_mean_sum_not==0] = 1
    bias_test_mean_not = bias_test_mean_not/bias_test_mean_sum_not
    if notdoing_p_same_t_on_nan_count+notdoing_p0_t1_on_nan_count+notdoing_p1_t0_on_nan_count != 0:
        temp_ab_test.append((point,pre_do_on_nnan_count, pre_do_on_nnan_count/nnan_count,notdoing_p_same_t_on_nan_count/(notdoing_p_same_t_on_nan_count+notdoing_p0_t1_on_nan_count+notdoing_p1_t0_on_nan_count),1-(notdoing_p0_t1_on_nan_count+notdoing_p1_t0_on_nan_count)/nnan_count))
    else:
        temp_ab_test.append((point,pre_do_on_nnan_count, pre_do_on_nnan_count/nnan_count,1.0,1-(notdoing_p0_t1_on_nan_count+notdoing_p1_t0_on_nan_count)/nnan_count))
    #break
print('--END--')


#draw plot
#doing_pro = [w[2] for w in temp_ab]
#pre_pro = [w[3] for w in temp_ab]
doing_pro_test = [w[2] for w in temp_ab_test]
pre_pro_test = [w[3] for w in temp_ab_test]
import matplotlib.pyplot as plt
#plt.plot(doing_pro, pre_pro,'r.') 
plt.plot(doing_pro_test, pre_pro_test,'b.') 
#plt.plot(x,visit_times_all[nan_sort_id]/30)
plt.xlabel('Doing proportion')
plt.ylabel('Precision on not-doing tests')
my_x_ticks = np.arange(0, 1.0001, 0.1)
my_y_ticks = np.arange(0.7, 1, 0.05)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show() 

import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

KKK = 1
main_device_id = 4
device_ids = [4]

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
TRAIN_test_data_all_rescale = ttt.copy()
import torch
torch.cuda.is_available()
WINDOW = 5
TEST_NUM = 12
import os
if torch.cuda.is_available():
    is_cuda = True
else:
    is_cuda = False 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

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
scale_contant = (1.0/torch.FloatTensor(np.mean(scale_s,0))).to(device)

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
NUM_EPOCH = 101
recal_k = 10
n_print = 30


class Recover_model(nn.Module):
    def __init__(self, vital_num,window_cnn,person_fnum,fnum_NUM,input_dim, hidden_dim1,hidden_dim2, test_num, dropout_prob, nlay1 = 1, nlay2 = 1,uniform_init = False):
        super(Recover_model, self).__init__()
        self.window_cnn = window_cnn
        self.vital_num = vital_num
        self.test_num = test_num
        self.hidden_cnn = test_num*4
        self.person_fnum = person_fnum
        self.fnum_NUM = fnum_NUM
        self.emb_person_dim = 4
        self.person_emb = nn.Embedding(fnum_NUM,self.emb_person_dim)
        self.cnn_gen_nan = nn.Conv2d(2,self.hidden_cnn,(window_cnn*2+1,test_num+self.vital_num),stride = 1,padding = (window_cnn,0))
        self.cnn_gen_nan2 = nn.Conv2d(1,self.hidden_cnn,(test_num*4,window_cnn*2+1),stride = 1,padding = (0,window_cnn))
        self.linear_cnn = nn.Linear(self.hidden_cnn+person_fnum*self.emb_person_dim, test_num)
        
        if uniform_init:
            nn.init.xavier_uniform_(self.person_emb.weight)
            nn.init.xavier_uniform_(self.cnn_gen_nan.weight)
            nn.init.xavier_uniform_(self.cnn_gen_nan2.weight)
            nn.init.xavier_uniform_(self.linear_cnn.weight)  
            
    def forward(self, vital_input_data,miss_mask,nan_mask,batch_input_data,nnan_mask,person_data_id,colla_prob1 = 0.1,colla_prob2 = 0.1): #person_data_id (B,person_fnum)
        
        person_data = self.person_emb(person_data_id) # (B,person_fnum, 2)
        person_data = person_data.view(person_data.size()[0],-1)
        person_data = torch.Tensor.repeat(person_data.unsqueeze(1),[1,batch_input_data.size()[1],1])
        batch_input_data_colla = batch_input_data.clone()
        mask_cnn = torch.rand(batch_input_data_colla.size())+colla_prob1
        batch_input_data_colla[mask_cnn>1.0] = 0.0
        nan_mask0 = nan_mask.clone()
        nan_mask0[mask_cnn>1.0] = 1.0
        
        #merged_data = torch.cat([person_data,batch_input_data_colla],2) # there are colla_prob1 collated
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
        
        return(loss2)

recover_model=Recover_model(vital_num,window_cnn,person_fnum,fnum_NUM,input_dim, hidden_dim1,hidden_dim2, TEST_NUM, dropout_prob, nlay1, nlay2,uniform_init).to(device)
optimizer_rc = torch.optim.Adam(recover_model.parameters(), lr=learning_rate, weight_decay=lambda_weight_decay)

NUM_EPOCH = 801
for n_epoch in range(NUM_EPOCH):
    torch.cuda.empty_cache()
    train_cost = 0
    input_iter = iter(train_dataloader)
    for i_train, TV_samples in enumerate(input_iter,0):
        batch_input_data,pfeature_batch, visit_mask, ab_mask, nnan_mask = TV_samples['input_data'].to(device),TV_samples['pfeature_batch'].to(device), TV_samples['visit_mask'].to(device), TV_samples['abnormal_mask'].to(device), TV_samples['not_nan_mask'].to(device)
        nan_mask,vital_batch,vital_miss_mask = TV_samples['nan_mask'].to(device),TV_samples['vital_batch'].to(device),TV_samples['miss_vital_mask'].to(device)
        optimizer_rc.zero_grad()
        loss = recover_model(vital_batch,vital_miss_mask,nan_mask,batch_input_data,nnan_mask,pfeature_batch)
        #print(loss)
        train_cost += loss.data
        loss.backward()
        optimizer_rc.step()
        torch.cuda.empty_cache()
    #print(train_cost)
    if n_epoch%50==0:
        print(train_cost/(i_train+1))


torch.save(recover_model.state_dict(), 'imputation_model4.pkl')

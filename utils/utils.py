import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from classifiers import ResNet
from classifiers import LSTM
from classifiers import InceptionTime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

import multiprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
import seaborn as sn

def accuracy_end_to_end(model,x_train,y_train,x_test,y_test,x_val=None,y_val=None,batch_size=64,max_epochs=50,default_root_dir=None):
    if 'sklearn' in str(type(model)) or 'sktime' in str(type(model)) or 'tslearn' in str(type(model)): 
        return accuracy_score(y_test,model.fit(x_train,y_train).predict(x_test))
    else:
        x_val = x_test
        y_val = y_test        
        
        return accuracy_score(y_test,model.fit(x_train,y_train,x_val,y_val,default_root_dir=default_root_dir).predict(x_test))
    
    
def init_model(model_name=None,model=None,nb_classes=2 ,lr =0.001, lr_factor = 0.5, lr_patience=50,lr_reduce=True, batch_size=64, earlystopping=False, et_patience=10, max_epochs=50, gpu = [0], default_root_dir=None):
    if model_name == None:
        return model
    elif model_name != None:
        if model_name == 'ResNet':
            return ResNet.Classifier_ResNet(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, gpu = gpu, default_root_dir=default_root_dir)
        elif model_name == 'LSTM':
            return LSTM.Classifier_LSTM(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, gpu = gpu, default_root_dir=default_root_dir)
        elif model_name == 'InceptionTime':
            return InceptionTime.Classifier_InceptionTime(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, gpu = gpu, default_root_dir=default_root_dir)
        elif model_name == 'KNN_DTW':
            return KNN_DTW.Classifier_KNN_DTW()
        
import multiprocessing

def find_cycles_idx_by_patient_idx(patient_idx,cycle_end_idx):
    
    if patient_idx == 0:
        return np.arange(0,cycle_end_idx[patient_idx])
    else:
        return np.arange(cycle_end_idx[patient_idx-1],cycle_end_idx[patient_idx])
    
def patients_idx_to_cycles_idx(patients_idx,cycle_end_idx):
    cycles_idx = []
    for i in patients_idx:
        cycles_idx.append(find_cycles_idx_by_patient_idx(i,cycle_end_idx))
        
    return np.concatenate(cycles_idx)

def find_patients_idx_by_cycles_idx(cycles_idx,cycle_end_idx):
    patients_idx = []
    for i in cycles_idx:
        patients_idx.append(cycle_idx_to_patient_idx(i,cycle_end_idx))
    return patients_idx

def cycle_idx_to_patient_idx(cycle_idx,cycle_end_idx):
    return np.where(cycle_idx>cycle_end_idx)[0][-1]+1

class GridSearchCV:
    def __init__(self, model,para_list,n_splits=7,n_process=10,default_root_dir=None):
        self.model = model
        
        self.ML = 'sklearn' in str(model) or 'sktime' in str(model) or 'tslearn' in str(model)
        
        self.para_list = list(ParameterGrid(para_list))
        self.n_splits = n_splits
        self.n_process = n_process
        self.default_root_dir = default_root_dir
        
    def fit(self, x_data_cv, y_data_cv, X,y,cycle_end_idx):
        print(self.ML)
        if self.ML:
            
            pool = multiprocessing.Pool(processes=self.n_process)
            print('Multi-process:',self.n_process)

        kf = StratifiedKFold(n_splits=self.n_splits)

        self.accuracy_mean_list = []
        self.accuracy_std_list = []

        self.accuracy_list = []

        for para in self.para_list:

            accuracy = []
            for train_index, test_index in kf.split(x_data_cv,y_data_cv):
                
                x_tr = X[patients_idx_to_cycles_idx(x_data_cv[train_index],cycle_end_idx)]
               
                y_tr = y[patients_idx_to_cycles_idx(x_data_cv[train_index],cycle_end_idx)]
                
                x_te = X[patients_idx_to_cycles_idx(x_data_cv[test_index],cycle_end_idx)]
                y_te = y[patients_idx_to_cycles_idx(x_data_cv[test_index],cycle_end_idx)]
                                                                    
                if self.ML:
                    accuracy.append( pool.apply_async(accuracy_end_to_end,args=(self.model(**para),x_tr, y_tr,x_te,y_te)))
                elif not self.ML:   
                    accuracy.append( accuracy_end_to_end(self.model(**para),x_tr, y_tr,x_te,y_te,default_root_dir=self.default_root_dir))
            if self.ML:
                accuracy = [i.get() for i in accuracy]

            self.accuracy_list.append(accuracy)

            self.accuracy_mean_list.append(np.mean(accuracy))
            self.accuracy_std_list.append(np.std(accuracy))

            print('parameter:',para)
            print('accuracy:',np.mean(accuracy),"±",np.std(accuracy))
            print()
        if self.ML:
            pool.close() 
            pool.join()

        self.best_accuracy_mean = self.accuracy_mean_list[np.argmax(self.accuracy_mean_list)]
        self.best_accuracy_std = self.accuracy_std_list[np.argmax(self.accuracy_mean_list)]
        self.best_para = self.para_list[np.argmax(self.accuracy_mean_list)]
        print('------------------------')
        print('best parameter:',self.best_para)
        print('accuracy:',self.best_accuracy_mean,"±",self.best_accuracy_std)

        self.accuracy_list = np.array(self.accuracy_list)


def error_patient_cycle(x_test,y_test,y_d,cycle_end_idx,pre_list,threshold=1,nb_classes=2):
    
    
    error_order_idx = [[],[]]
    error_cycles_idx = []

    y_test = y_test.astype(int)

    cum_list = np.cumsum([len(find_cycles_idx_by_patient_idx(i,cycle_end_idx)) for i in x_test])
    
    
    n_subject = np.zeros(nb_classes,dtype=int)
    
    
    n_cycle = 0
    n_patient_well_pre = [0,0]
    
    for i in range(len(x_test)):
        
        if isinstance(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)],np.ndarray):
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        else:
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy().astype(int)


        y_true = y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        
        print('-----------------------------')
        print('Subject No.',i)
        print('cycle_n=',len(find_cycles_idx_by_patient_idx(i,(cum_list))))
        compre_list,compre_list_count = np.unique(y_true ==y_pre,return_counts=True)
        
        n_error = 0
        if False in compre_list:
            n_error = compre_list_count[np.where(compre_list==False)[0][0]]
            
        print('n_error=',n_error)
        print()
        
        n_subject_one = np.zeros(nb_classes,dtype=int)
        
        
        if n_error:
            n_subject[int(y_test[i])]+=1
            n_subject_one[int(y_test[i])] += n_error

            n_cycle += n_error
        
        print('error in label '+str(y_test[i])+'=', n_subject_one[int(y_test[i])])    
        print('pre_list',y_pre)

        print('y_test[i]',y_test[i])
        print('majority:', np.argmax(np.bincount(y_pre)))
        
        if y_test[i] == np.argmax(np.bincount(y_pre)) and n_error!=0:
            n_patient_well_pre[y_test[i]]+=1
            
            print('well predict for class '+str(y_test[i]))
            
        elif n_error==0:
            n_patient_well_pre[y_test[i]]+=1
            
            print('well predict for no error')
        else:
            print('majority error for label No.',i)
            error_order_idx[y_test[i]].append(i)
            error_cycles_idx.append(np.where(y_pre!=y_test[i]))
            
        

    print('- Origin -')   
    for i in range(nb_classes):
        print('n_subject_'+str(i)+'=',len(y_test[y_test==i]))
    print('n_subject:',len(y_test))
    print('n_patient_well_pre_majority:',n_patient_well_pre)
    print('n_cycles:',len(patients_idx_to_cycles_idx(x_test,cycle_end_idx)))

    print('- Error -')
    for i in range(nb_classes):
        print('n_subject_'+str(i)+'=',n_subject[i])

    print('n_subject:',np.sum(n_subject))
    print('n_cycles:',n_cycle)
    print('error patient idx:', error_order_idx)
    print('error_cycles_idx:', error_cycles_idx)
    return error_order_idx, n_patient_well_pre

def prediction_subject(x_test,y_test,y_d,cycle_end_idx,pre_list):
    
    y_test = y_test.astype(int)
    cum_list = np.cumsum([len(find_cycles_idx_by_patient_idx(i,cycle_end_idx)) for i in x_test])
    pre_subject_list = []

    for i in range(len(x_test)):

        if isinstance(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)],np.ndarray):
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        else:
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy().astype(int)


        pre_subject_list.append(np.argmax(np.bincount(y_pre)))

        
    return pre_subject_list
    
    
def print_result_bundle_i_cycles(out,y_d,x_test,cycle_end_idx,label_list,save=None,n_model=10,binary=True):
    ac_list = []
    f1_list = []
    recall_list = []

    recall_list = []

    specificity_list = []

    auc_list = []
    c_matrix_list = [] 

    for i in range(n_model):
        pre_lab_test = [ np.argmax(i) for i in out[i]]
        ac_list.append(accuracy_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))
        c_matrix_list.append(confusion_matrix(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))
        
        if binary==True:

            f1_list.append(f1_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))

            recall_list.append(recall_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))

            specificity_list.append(recall_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test, pos_label=0))

            auc_list.append(roc_auc_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))




    print("Accuracy:")
    print("Accuracy_list:", ac_list)
    print("Accuracy_mean:", np.mean(ac_list))
    print("Accuracy_std:", np.std(ac_list))
    print("---")
    
    print("Confusion matrix")
    print("Confusion matrix mean:", np.mean(c_matrix_list, axis=0))
    print("Confusion matrix std:", np.std(c_matrix_list, axis=0))
    print("---")
    
    if binary==True:

        print("F1")
        print("F1_list:",(f1_list))
        print("F1_mean:",np.mean(f1_list))
        print("F1_std:",np.std(f1_list))
        print("---")

        print("Sensitivity")
        print("Sensitivity list:", recall_list)
        print("Sensitivity mean:",np.mean(recall_list))
        print("Sensitivity std:",np.std(recall_list))
        print("---")

        print("Specificity")
        print("Specificity list:",specificity_list)
        print("Specificity mean:",np.mean(specificity_list))
        print("Specificity std:",np.std(specificity_list))
        print("---")


        print("AUC")
        print("AUC list:", auc_list)
        print("AUC mean:",np.mean(auc_list))
        print("AUC std:",np.std(auc_list))


    
    

#     df_cm = pd.DataFrame(np.mean(c_matrix_list, axis=0),index = label_list, columns = label_list )
#     plt.figure(dpi=100)
#     sn.set(font_scale=1.4) # for label size
#     sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
#     # plt.title("Confusion matrix ResNet", fontsize =20)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
    
#     if save!=None:
#         plt.savefig(save+"_mean_cycles",bbox_inches = 'tight',facecolor ="w")
        
    
#     plt.show()
#     plt.close()

#     df_cm = pd.DataFrame(np.std(c_matrix_list, axis=0),index = label_list, columns = label_list )
#     plt.figure(dpi=100)
#     sn.set(font_scale=1.4) # for label size
#     sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
    
#     if save!=None:
#         plt.savefig(save+"_std_cycles",bbox_inches = 'tight',facecolor ="w")
   
#     plt.show()
#     plt.close()
    
    if binary==True:
    
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(recall_list), np.std(recall_list)], [np.mean(specificity_list), np.std(specificity_list)], [np.mean(f1_list),np.std(f1_list)],[np.mean(auc_list),np.std(auc_list)]]
    else: 
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(f1_list),np.std(f1_list)]]

def print_result_bundle_i_subjects(out,y_d,x_test, y_test,cycle_end_idx,label_list,save=None ,n_model=10, binary=True):
    pre_subject_list = []
    for i in range(n_model):
        pre_subject = prediction_subject(x_test,y_test,y_d,cycle_end_idx,np.array([ np.argmax(j) for j in out[i]]))
        pre_subject_list.append(pre_subject)
        com = pre_subject == y_test
        print("Error subject index of model "+str(i)+" :",np.where(com==False))
        
    
    ac_list = []
    f1_list = []
    recall_list = []

    recall_list = []

    specificity_list = []

    auc_list = []
    c_matrix_list = [] 

    for i in range(n_model):
        pre_lab_test = pre_subject_list[i]
        ac_list.append(accuracy_score(y_test,pre_lab_test))
        c_matrix_list.append(confusion_matrix(y_test,pre_lab_test))
        
        if binary==True:
            f1_list.append(f1_score(y_test,pre_lab_test))

            recall_list.append(recall_score(y_test,pre_lab_test))

            specificity_list.append(recall_score(y_test,pre_lab_test, pos_label=0))

            auc_list.append(roc_auc_score(y_test,pre_lab_test))

        
        
    
    print("Accuracy:")
    print("Accuracy_list:",ac_list)
    print("Accuracy_mean:",np.mean(ac_list))
    print("Accuracy_std:",np.std(ac_list))
    print("---")
    
    print("Confusion matrix")
    print("Confusion matrix mean:",np.mean(c_matrix_list, axis=0))
    print("Confusion matrix std:",np.std(c_matrix_list, axis=0)) 
    print("---")
    
    if binary==True:

        print("F1")
        print("F1_list:",(f1_list))
        print("F1_mean:",np.mean(f1_list))
        print("F1_std:",np.std(f1_list))
        print("---")

        print("Sensitivity")
        print("Sensitivity list:", recall_list)
        print("Sensitivity mean:",np.mean(recall_list))
        print("Sensitivity std:",np.std(recall_list))
        print("---")

        print("Specificity")
        print("Specificity list:",specificity_list)
        print("Specificity mean:",np.mean(specificity_list))
        print("Specificity std:",np.std(specificity_list))
        print("---")


        print("AUC")
        print("AUC list:", auc_list)
        print("AUC mean:",np.mean(auc_list))
        print("AUC std:",np.std(auc_list))


        
#     df_cm = pd.DataFrame(np.mean(c_matrix_list, axis=0),index = label_list, columns = label_list )
#     plt.figure(dpi=100)
#     sn.set(font_scale=1.4) # for label size
#     sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
    
    
#     if save!=None:
#         plt.savefig(save+"_mean_subjects",bbox_inches = 'tight',facecolor ="w")
        
#     plt.show()
#     plt.close()
    
#     df_cm = pd.DataFrame(np.std(c_matrix_list, axis=0),index = label_list, columns = label_list )
#     plt.figure(dpi=100)
#     sn.set(font_scale=1.4) # for label size
#     sn.heatmap(df_cm, annot=True, fmt='.2f', annot_kws={"size": 16}) # font size
#     # plt.title("Confusion matrix ResNet", fontsize =20)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
    
    
#     if save!=None:
#         plt.savefig(save+"_std_subjects",bbox_inches = 'tight',facecolor ="w")
        
        
#     plt.show()
#     plt.close()
    
    if binary==True:
    
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(recall_list), np.std(recall_list)], [np.mean(specificity_list), np.std(specificity_list)], [np.mean(f1_list),np.std(f1_list)],[np.mean(auc_list),np.std(auc_list)]]
    else: 
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(f1_list),np.std(f1_list)]]
    
def merge_mean_std(mean_std):
    return str('%.2f'%mean_std[0])+"±"+str('%.2f'%mean_std[1])

def output_csv(output_list):
    list_str = []
    for i in output_list:
        list_str.append(merge_mean_std(i))
    return list_str

def eval_model(D_name,Net_name,X_d,y_d,x_test,y_test,cycle_end_idx,nb_classes=2):
    
    list_f1 = []
    list_accuracy = []

    import os
    filePath = "./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    os.listdir(filePath)

    for i in os.listdir(filePath):
        if 'model' in i:
            print(i)
            classifer = init_model(model_name=Net_name,nb_classes=nb_classes).load( "./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+str(i),nb_classes=nb_classes)
            list_accuracy.append(accuracy_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)], classifer.predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)])))
            list_f1.append(f1_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)], classifer.predict(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]),average='micro'))
    return list_accuracy, "%.4f" %(np.mean(list_accuracy)) +"±"+ "%.4f" %(np.std(list_accuracy)),list_f1 , "%.4f" %(np.mean(list_f1)) +"±"+ "%.4f" %(np.std(list_f1))


def batch_predict(D_name,Net_name, images,model_path,nb_classes=2):
    pred_set = Dataset_torch(images,with_label=False)
    data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=64,num_workers=4)
    
    
    classifer = init_model(model_name=Net_name,nb_classes=nb_classes).load(model_path, nb_classes=nb_classes)
    
    trainer = pl.Trainer(gpus=[0])
    pred = trainer.predict(model=classifer.model,dataloaders = data_loader_pred)
    pred = torch.cat(pred)
    return pred.detach().cpu().numpy()

class Dataset_torch(Dataset):

    def __init__(self, data,with_label=True):
        self.with_label =  with_label
        
        if self.with_label:
            self.data_x, self.data_y = data
        else:
            self.data_x = data
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.with_label:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]
        
def eval_model_5_bundle(D_name,Net_name,X_d,y_d,x_test,y_test,cycle_end_idx,idx=None):
    
    list_f1 = []
    list_accuracy = []

    import os
    
    
    
    filePath = "./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    model_list = os.listdir(filePath)
    for idx_tmp,i in enumerate(model_list):
        if not i.startswith('model'):
            model_list.remove(i)
    print("len(model_list)",len(model_list))
    proba_sum_list = []
    if idx==None:
        for index in range(10):
            print(index)
            proba_sum = np.zeros([len(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]),2])
            for j in range(5):

                print(model_list[index])

                proba_sum +=batch_predict(D_name,Net_name, X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],"./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j])/5
            proba_sum_list.append(proba_sum)
    else:
            index = idx
            print(index)
            proba_sum = np.zeros([len(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]),2])
            for j in range(5):

                print(model_list[index])

                proba_sum +=batch_predict(D_name,Net_name, X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],"./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j])/5
            proba_sum_list = (proba_sum)
    return proba_sum_list

def train_model(D_name,Net_name,X_d,y_d,x_train,x_val,y_train,y_val,cycle_end_idx,nb_classes=2):
    a_list = []
    for i in range(5):
        print("train number -", i)
        classifer = init_model(model_name=Net_name,lr_reduce=True, earlystopping=True, lr_patience=1, batch_size=64,lr=0.001, max_epochs=50,gpu=[0],default_root_dir="./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/",nb_classes=nb_classes)
        classifer.fit(X_d[patients_idx_to_cycles_idx(x_train,cycle_end_idx)], y_d[patients_idx_to_cycles_idx(x_train,cycle_end_idx)], X_d[patients_idx_to_cycles_idx(x_val,cycle_end_idx)], y_d[patients_idx_to_cycles_idx(x_val,cycle_end_idx)],ckpt_monitor='val_accuracy')
        print("train number end -", i)
        
def load_dataset_v1(dir_dataset,d_file_list,channel_first=True,flatten=True,shuffle=False,random_state=0):
    index = [0,1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,22,23,24,29]
    
    x_list = []
    
    for i in d_file_list:
    
        npzfile = np.load(dir_dataset + i)

        x_all = npzfile['Input']

        x = x_all[:,:,index]
        if channel_first:
            x = x.transpose(0,2,1)
            
        if flatten == True:
            x = x.reshape([x.shape[0],-1])
        
        x_list.append(x)

    nb_classes = len(d_file_list)
    
    return x_list, nb_classes

def generate_data_for_train(x_list, idx_file_list,order):
    X_d = np.concatenate(x_list)
    y_d = []
    idx_list = []
    Flag = True
    
    for idx,i in enumerate(idx_file_list):
        
        if Flag == True:
            idx_0 = pd.read_csv(i, header=None)[2].to_numpy()
            idx_list.append(idx_0)
            y_d.append(np.zeros((idx_0[-1]))+order[idx])
            
        else:
            idx_new = pd.read_csv(i, header=None)[2].to_numpy()
            idx_list.append(idx_new+idx_list[-1][-1])
            y_d.append(np.zeros((idx_new[-1]))+order[idx])

        Flag = False
            
            
    cycle_end_idx = np.concatenate(idx_list)
 
    patient_index_range = np.arange(len(cycle_end_idx))    
    
    patient_class_list = []
    for idx,i in enumerate(idx_list):
        
        patient_class_list.append(np.zeros(len(i))+order[idx])
    
    patient_class = np.concatenate(patient_class_list)
    y_d = np.concatenate(y_d)    
    
    x_train, x_test, y_train, y_test = train_test_split(patient_index_range, patient_class, test_size=0.4, stratify=patient_class,random_state=0)
    x_val, x_test, y_val, y_test = train_test_split(x_test,y_test, test_size=0.75, random_state=0, stratify=y_test)
       
    return x_train, x_test, y_train, y_test, x_val, x_test, y_val, y_test, cycle_end_idx, X_d, np.array(y_d)

def plot_cycle_with_idx(n_subject, name_dataset, RL_list, cycle_end_idx, X_d):

    cycles_idx = find_cycles_idx_by_patient_idx(n_subject,cycle_end_idx)
    X_d_all = X_d[cycles_idx]
    d_mean_all = np.mean(X_d_all,0)

    fig, axs = plt.subplots(6, 7, figsize=(70, 40))

    title_list = ['F/E','Ab/Ad','Rot I/E']

    class_list = [name_dataset+' - subject No.'+str(n_subject)]



    for i,ax in enumerate(axs.flatten(), start=0):
        if i<6:
            axs.flatten()[i].axis('off')
            axs.flatten()[i].text(0.15,0.1,title_list[i%3],size=60,weight='bold')
            if i%3==1:
                axs.flatten()[i].text(0.0,0.6,RL_list[(i-1)//3],size=60,weight='bold')

    list_right = ['','Pelvis','Hip','Knee','Ankle','Foot']
    for idx,i in enumerate([6,13,20,27,34,41]):  
        axs.flatten()[i].axis('off')
        axs.flatten()[i].text(0.1,0.5,list_right[idx],size=60,weight='bold')

    for i in range(7):
        ht_line = plt.axes([0.065, 0.9-i*0.133, 0.89,0.01])
        ht_line.plot([0,1],[0.5,0.5],linewidth = 10,c='black')
        ht_line.axis('off')
        if i == 0:
            ht_line.text( 2.7/7,0.3,class_list[0],size=60,weight='bold')

    for i in range(8):
        linewidth = 5
        high_line = 0.73

        if i in [0,6,7]:
            linewidth_use = 10
            high_line_use = 0.87
        elif i in [3,9]:
            linewidth_use = 10
            high_line_use = 0.8
        else:
            linewidth_use = linewidth
            high_line_use = high_line

        t_line = plt.axes([0.1+i*0.116, 0.07, 0.01, high_line_use])
        t_line.plot([0.5,0.5],[0,1],linewidth = linewidth_use,c='black')
        t_line.axis('off')


    plt.subplots_adjust(wspace=0.5, 
                        hspace=0.5)


    for idx,i in enumerate([7,8,9,14,15,16,21,22,23,28,37 , 7+3,8+3,9+3,14+3,15+3,16+3,21+3,22+3,23+3,28+3,37+3 ]):

        if i in [7,14,21,28, 7+3,14+3,21+3,28+3]:
            axs.flatten()[i].set_ylim(-20,70)
            axs.flatten()[i].set_yticks([-20,0,20,40,70], size = 50)
        else:
            axs.flatten()[i].set_ylim(-45,45)
            axs.flatten()[i].set_yticks([-45,-25,-10,0,10,25,45], size = 50)
        for idx_jj,jj in enumerate(X_d_all):
            if idx_jj == 0:
                label = 'Cycles'
            else:
                label = None

            tmp = axs.flatten()[i].plot(range(101),jj[idx], label=label,c='b',linewidth=1)


        tmp = axs.flatten()[i].plot(range(101),d_mean_all[idx],label='Avg',c='r',linewidth=3)
        plt.subplots_adjust(left=None, bottom=None, right=0.91, top=0.9,hspace=0.45,wspace=0.3)
        axs.flatten()[i].spines['bottom'].set_linewidth(3)
        axs.flatten()[i].spines['left'].set_linewidth(3)
        axs.flatten()[i].spines['right'].set_linewidth(3)
        axs.flatten()[i].spines['top'].set_linewidth(3)

        axs.flatten()[i].set_ylabel('Angle',weight= 'bold',size= 20)
        axs.flatten()[i].set_xlabel('% gait cycle',weight= 'bold',size= 20)
        axs.flatten()[i].legend(prop = {'size':20})

        axs.flatten()[i].tick_params(labelsize=20)

    for i in np.arange(1,7)*7-1:
        axs.flatten()[i].axis('off')

    for i in [29,30,35,36]:
        for j in range(2):
            axs.flatten()[i+j*3].axis('off')
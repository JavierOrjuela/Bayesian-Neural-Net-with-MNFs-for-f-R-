### Code written by Hector J. Hortua for the paper Bayesian deep learning for cosmic volumes with modified gravity arXiv:2309.00612 
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tqdm import tqdm
import sys
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#cloned from https://github.com/janosh/tf-mnf
from tf_mnf import models  # ,ROOT
from  tf_mnf import layers as tfmnflayer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path,PurePath
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import zipfile
import argparse
import glob
from tensorflow.keras.layers import Input,Conv3D, BatchNormalization,Dropout,LeakyReLU,Flatten,Dense,GlobalAveragePooling3D,MaxPooling3D,AveragePooling3D,GlobalMaxPool3D
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from classification_models_3D.tfkeras import Classifiers
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras
RANDOM=123
gpus = tf.config.experimental.list_physical_devices('GPU')

parser = argparse.ArgumentParser(description='model3D')
parser.add_argument('-bs', '--batch_size', type=int, help='batch',default=8)
parser.add_argument('-epochs', '--epochs', type=int, help='epochs',default=50*5)
parser.add_argument('-lr', '--lr', type=float, help='lr',default=1e-3)
parser.add_argument('-path', '--path', type=str, help='pathdir',default='tf-mnf/logs3Dpsall1/')
#parser.add_argument('-model', '--model', type=str, help='model',default='resnet18')
parser.add_argument('-varpk', '--varpk', type=str, help='variable pk used',default='pk')
args = parser.parse_args()


APPLY_LOG=True
VARIABLE_PK=args.varpk  #'k3pk1'
BATCH_SIZE=args.batch_size
epochs=args.epochs
model_used=VARIABLE_PK
checkpoint_filepath = args.path+'{}'.format(model_used)
save_df=args.path+'{}/df'.format(model_used)
if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)
if not os.path.exists(save_df):
    os.makedirs(save_df)


data = pd.read_csv('latin_hypercube_params.txt', sep="\s", header=0)
array= np.arange(0,2500,1)
data['filename'] = array.tolist()

image_File_64=''
def omega_bh2(x):    
    return x["omega_b"] * x["h"]* x["h"] 
def omega_mh2(x):    
    return x["omega_m"] * x["h"]* x["h"]
def S8(x):    
    return x["sigma_8"] * (x["omega_m"]/0.3)**0.5

data['filename_path']=data.apply(
         lambda row: '/pk/pks/pofk_fR_'+str(int(row.filename))+'_z0.000_CDM.txt',axis=1)


df= pd.read_csv(data['filename_path'][0],delim_whitespace=True,skiprows=1,header=None,names=["k","pk","kmean","k3pk1"])

filename_64=data['filename_path']
#filename_32=data['filename']
filename_names=['filename_path']
#"omega_b","sigma_8","h","n_s"]#["omega_m","omega_b","h","n_s","sigma_8"]
params=['Om', 'h', 'sigma8', 'fR0_scaled']
len_params=len(params)

directory_folder='boxes_interpo/'
CIC_path=directory_folder+'CIC/density_field_subsample_test_CIC/'
sub_folders_CIC = [int(name) for name in os.listdir(CIC_path) if os.path.isdir(os.path.join(CIC_path, name))]
Test_LSS= data[data['filename'].isin(sub_folders_CIC)]
data= data[~data['filename'].isin(sub_folders_CIC)]



removes=[]
def min_values(df,image_File,valiable_pk='pk',apply_log=False):
     try:
         imagedf=pd.read_csv(image_File+df,delim_whitespace=True,skiprows=1,header=None,names=["k","pk","kmean","k3pk1"])
         imagedf['k3pk']=imagedf['pk']*imagedf['k']**3
         imagedf=imagedf.iloc[:85]
         if apply_log:
                imagedf['log_pk']=np.log10(imagedf['pk'])
                valiable_pk='log_pk'
         image=imagedf[valiable_pk]
         min_val=image.min()
         return min_val
     except:
         print("Error! Could not load encoder for feature ", Path(df).parts[0])
         removes.append(int(Path(df).parts[0]))
         return None
    
def max_values(df,image_File,valiable_pk='pk',apply_log=False):
     try:
         imagedf=pd.read_csv(image_File+df,delim_whitespace=True,skiprows=1,header=None,names=["k","pk","kmean","k3pk1"])
         imagedf['k3pk']=imagedf['pk']*imagedf['k']**3
         imagedf=imagedf.iloc[:85]
         if apply_log:
                imagedf['log_pk']=np.log10(imagedf['pk'])
                valiable_pk='log_pk'
         image=imagedf[valiable_pk]
         max_val=image.max()
         return max_val
         
     #import pdb;pdb.set_trace()
     except:
         print("Error! Could not load encoder for feature ", Path(df).parts[0])
         return None

        

shuffle_data= data.sample(frac=1,random_state=RANDOM).reset_index(drop=True)
label_df=shuffle_data
Train_LSS, Validation_LSS = train_test_split(label_df, test_size=0.1,random_state=RANDOM)
Data_train_length =Train_LSS.shape[0]

features= Train_LSS[params].values
num_feratures=len(params)
scaler_data_max_=np.array([0.49977939, 0.89993787 ,0.99983828 ,0.59985074])
scaler_data_min_=np.array([0.10018186 ,0.50018637 ,0.60001234 ,0.40007124])

removes=[]

min_val_64=-1.0
max_val_64=1978.0874

def normalization_data_64(array):
    thres=1.
    return (array-min_val_64)/(max_val_64-min_val_64)

def normalization_features(feat):
    return (feat-scaler_data_min_)/(scaler_data_max_-scaler_data_min_)

def normalization_features_mean(feat):
  return feat*(scaler_data_max_-scaler_data_min_)+scaler_data_min_
def normalization_features_var(feat):
  return feat*(scaler_data_max_-scaler_data_min_)**2


def load_arrays(path,valiable_pk=VARIABLE_PK,apply_log=APPLY_LOG):
  #import pdb;pdb.set_trace()
  imagedf=pd.read_csv(path.numpy().decode("utf-8"),delim_whitespace=True,skiprows=1,header=None,names=["k","pk","kmean","k3pk1"])
  imagedf['k3pk']=imagedf['pk']*imagedf['k']**3  
  imagedf=imagedf.iloc[:85]
  if apply_log:
      imagedf['log_pk']=np.log10(imagedf['pk'])
      valiable_pk='log_pk'
  image=imagedf[valiable_pk]
  return image

def tf_data_array(filenames1, features):
  [feat,]= tf.py_function(normalization_features, [features], [tf.float64])
  [image_64,] = tf.py_function(func= load_arrays,  inp=[filenames1], Tout=[tf.float64])
  image_64.set_shape(ma_64.shape,)
  feat.set_shape(features.shape)
  return image_64,  feat

val_pk=VARIABLE_PK
a_log=APPLY_LOG
ma_64=pd.read_csv(image_File_64+filename_64[1],delim_whitespace=True,skiprows=1,header=None,names=["k","pk","kmean","k3pk1"])
ma_64['k3pk']=ma_64['pk']*ma_64['k']**3
ma_64=ma_64.iloc[:85]
if a_log:
      ma_64['log_pk']=np.log10(ma_64['pk'])
      val_pk='log_pk'
ma_64=ma_64[val_pk]


def datasets_iteration(dataset_row,files_mod, training=True):
    filename_64=dataset_row[files_mod[0]]
    features=dataset_row[params].values
    dataset = tf.data.Dataset.from_tensor_slices((filename_64,features))
    print(dataset)
    if training:
        dataset =dataset.shuffle(len(dataset_row),seed=RANDOM)
    dataset = dataset.map(tf_data_array).batch(BATCH_SIZE)
    return dataset


Train_dataset= datasets_iteration(Train_LSS,filename_names)
Validation_dataset= datasets_iteration(Validation_LSS,filename_names)
Test_dataset= datasets_iteration(Test_LSS,filename_names,False)
for i,j in Train_dataset.take(1):
    print(i.shape,j.shape)
    shaped=i.shape


from  tf_mnf import layers as tfmnflayer

model1 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(shaped[-1],)),
    tfmnflayer.MNFDense(64),
    tf.keras.layers.Activation('relu'),
    tfmnflayer.MNFDense(64),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.BatchNormalization(),
    tfmnflayer.MNFDense(64),
    tf.keras.layers.Activation('relu'),
    tfmnflayer.MNFDense(tfp.layers.MultivariateNormalTriL.params_size(len_params)),
    tfp.layers.MultivariateNormalTriL(len_params)])
model1.summary()

def loss_fn(labels, preds):
    cross_entropy = -preds.log_prob(labels)
    entropic_loss = tf.reduce_mean(cross_entropy)
    kl_loss = (model1.layers[-2].kl_div()+model1.layers[-4].kl_div()+model1.layers[-7].kl_div()+model1.layers[-9].kl_div()) / Data_train_length
    loss = entropic_loss +  kl_loss

    return loss


adam = tf.optimizers.Adam(args.lr)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min' ,factor=0.9,verbose=1,
                              patience=5, min_lr=1e-7)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200,mode='min',restore_best_weights=True,
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath+'/checkpoint/checkpoints',
    save_weights_only=True,
    monitor='val_mse',
    mode='min',
    save_best_only=True)

model1.compile(loss=loss_fn, optimizer=adam, metrics=["mse"])

mnf_hist = model1.fit(
     Train_dataset, epochs=epochs,
           validation_data=Validation_dataset,  callbacks=[reduce_lr,callback,model_checkpoint_callback],verbose=1)

model1.evaluate(Test_dataset)



def get_samples(Test_dataset_=Test_dataset,mnf_lenet_=model1, filepath=''):
    for x, y_ in Test_dataset.take(1):
          y =normalization_features_mean(y_) 
          samples_= mnf_lenet_(x).sample(10000)
          samples_=normalization_features_mean(samples_)
          np.save(filepath+"_samples_.npy", samples_.numpy())
          np.save(filepath+"_targetsamples_.npy", y.numpy())
    return None

get_samples(Test_dataset_=Test_dataset,mnf_lenet_=model1, filepath=save_df+'/Test_dataset')































































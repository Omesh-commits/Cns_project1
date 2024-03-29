# -*- coding: utf-8 -*-
"""CNS_Project2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HMyVVwz7YokLCBwWWZ_fm9UPaCopUgKC
"""

!pip install -qq -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans
import sys
sys.path.append('/content/src/cleverhans')
import cleverhans

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop,adam

from cleverhans.attacks import FastGradientMethod ,SaliencyMapMethod
from cleverhans.utils_tf import model_train,model_eval,batch_eval
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes

import tensorflow as tf
from tensorflow.python.platform import flags

from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,roc_curve,auc,f1_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.svm import SVC,LinearSVC

import matplotlib.pyplot as plt

import cleverhans.model
from cleverhans.attacks import *

pip install py-flags

from flags import *

FLAGS=flags.FLAGS

####Delete all flags before declare#####

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.delattr(keys)
        del_all_flags(tf.flags.FLAGS)

plt.style.use('bmh')

FLAGS=flags.FLAGS
flags = tf.app.flags 
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('nb_epochs',120,'Number of epochs to train model')
flags.DEFINE_integer('batch_size',128,'Size of training batches')
flags.DEFINE_float('learning_rate',0.01,'Learning rate for training')
flags.DEFINE_integer('nb_classes',5,'Number of classification classes')
flags.DEFINE_integer('source_samples',10,'Nb of test set examples to attack')

#from google.colab import drive
#drive.mount('/content/drive')

#from google.colab import drive
#drive.mount('/content/drive')
# from google.colab import files
# uploaded = files.upload()
# import io

names=['duration','protocol','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack_type','other']

dft = pd.read_csv('KDDTest+.txt',names=names,header=None)
df = pd.read_csv('KDDTrain+.txt',names=names,header=None)

dft.shape

df.shape

full=pd.concat([df,dft])

assert full.shape[0]==df.shape[0]+dft.shape[0]

full['label']=full['attack_type']

#DoSattacks
full.loc[full.label=='neptune','label']='dos'
full.loc[full.label=='back','label']='dos'
full.loc[full.label=='land','label']='dos'
full.loc[full.label=='pod','label']='dos'
full.loc[full.label=='smurf','label']='dos'
full.loc[full.label=='teardrop','label']='dos'
full.loc[full.label=='mailbomb','label']='dos'
full.loc[full.label=='processtable','label']='dos'
full.loc[full.label=='udpstorm','label']='dos'
full.loc[full.label=='apache2','label']='dos'
full.loc[full.label=='worm','label']='dos'

#User-to-Root(U2R)
full.loc[full.label=='buffer_overflow','label']='u2r'
full.loc[full.label=='loadmodule','label']='u2r'
full.loc[full.label=='perl','label']='u2r'
full.loc[full.label=='rootkit','label']='u2r'
full.loc[full.label=='sqlattack','label']='u2r'
full.loc[full.label=='xterm','label']='u2r'
full.loc[full.label=='ps','label']='u2r'

#Remote-to-Local(R2L)
full.loc[full.label=='ftp_write','label']='r2l'
full.loc[full.label=='guess_passwd','label']='r2l'
full.loc[full.label=='imap','label']='r2l'
full.loc[full.label=='multihop','label']='r2l'
full.loc[full.label=='phf','label']='r2l'
full.loc[full.label=='spy','label']='r2l'
full.loc[full.label=='warezclient','label']='r2l'
full.loc[full.label=='warezmaster','label']='r2l'
full.loc[full.label=='xlock','label']='r2l'
full.loc[full.label=='xsnoop','label']='r2l'
full.loc[full.label=='snmpgetattack','label']='r2l'
full.loc[full.label=='httptunnel','label']='r2l'
full.loc[full.label=='snmpguess','label']='r2l'
full.loc[full.label=='sendmail','label']='r2l'
full.loc[full.label=='named','label']='r2l'

#Probeattacls
full.loc[full.label=='satan','label']='probe'
full.loc[full.label=='ipsweep','label']='probe'
full.loc[full.label=='nmap','label']='probe'
full.loc[full.label=='portsweep','label']='probe'
full.loc[full.label=='saint','label']='probe'
full.loc[full.label=='mscan','label']='probe'
full=full.drop(['other','attack_type'],axis=1)

print("Uniquelabels",full.label.unique())

full2=pd.get_dummies(full,drop_first=False)

features=list(full2.columns[:-5])

y_train=np.array(full2[0:df.shape[0]][['label_normal','label_dos','label_probe','label_r2l','label_u2r']])

X_train=full2[0:df.shape[0]][features]

y_test=np.array(full2[df.shape[0]:][['label_normal','label_dos','label_probe','label_r2l','label_u2r']])

X_test=full2[df.shape[0]:][features]

scaler=MinMaxScaler().fit(X_train)

X_train_scaled=np.array(scaler.transform(X_train))

X_test_scaled=np.array(scaler.transform(X_test))

#GeneratelabelencodingforLogisticregression
labels=full.label.unique()
le=LabelEncoder()
le.fit(labels)
y_full=le.transform(full.label)
y_train_l=y_full[0:df.shape[0]]
y_test_l=y_full[df.shape[0]:]

print("Training data set shape",X_train_scaled.shape,y_train.shape)
print("Testdata set shape",X_test_scaled.shape,y_test.shape)
print("Label encodery shape",y_train_l.shape,y_test_l.shape)

def mlp_model():
    model =Sequential()
    model.add(Dense(256, activation='relu',input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(FLAGS.nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

#Evaluation of Model
def evaluate():
    
    eval_params={'batch_size':FLAGS.batch_size}
    accuracy=model_eval(sess,x,y,predictions,X_test_scaled,y_test,args=eval_params)
    print('Testaccuracyonlegitimatetestexamples:'+str(accuracy))

#jsma = SaliencyMapMethod(cleverhans.model.Model(model), sess=sess)

#passed this to bypass the  Unrecognized flag error:)
tf.app.flags.DEFINE_string('f', '', 'kernel')

x=tf.placeholder(tf.float32,shape=(None,X_train_scaled.shape[1]))
y=tf.placeholder(tf.float32,shape=(None,FLAGS.nb_classes))

tf.set_random_seed(42)
model=mlp_model()
sess=tf.Session()
predictions=model(x)
init=tf.global_variables_initializer()
sess.run(init)

train_params={'nb_epochs':FLAGS.nb_epochs,'batch_size':FLAGS.batch_size,'learning_rate':FLAGS.learning_rate,'verbose':0}
model_train(sess,x,y,predictions,X_train_scaled,y_train,evaluate=evaluate,args=train_params)

source_samples=X_test_scaled.shape[0]

results=np.zeros((FLAGS.nb_classes,source_samples),dtype='i')
perturbations=np.zeros((FLAGS.nb_classes,source_samples),dtype='f')
grads=jacobian_graph(predictions,x,FLAGS.nb_classes)

X_adv=np.zeros((source_samples,X_test_scaled.shape[1]))

#pip install statlib

jsma = SaliencyMapMethod(cleverhans.model.Model(model), sess=sess)

print(X_adv.shape)

eval_params={'batch_size':FLAGS.batch_size}

jsma_params = {'theta': 1., 'gamma': 0.1, 'clip_min': 0., 'clip_max': 1.,'y_target': None}

figure = None
# Keep track of success (adversarial example classified in target)
results = np.zeros((FLAGS.nb_classes, source_samples), dtype='i')

  # Rate of perturbed features for each test set example and target class
perturbations = np.zeros((FLAGS.nb_classes, source_samples), dtype='f')



accuracy=model_eval(sess,x,y,predictions,X_test_scaled,y_test,args=eval_params)
print('Test accuracy on normal examples:'+str(accuracy))

for sample_ind in range(0,source_samples):
    current_class =int(np.argmax(y_test[sample_ind]))
    for target in [0]: 
        adv_x,res,percent_perturb=jsma.generate_np(sess,x)

        X_adv[sample_ind]=adv_x
        results[target,sample_ind]=res
        perturbations[target,sample_ind]=percent_perturb

#Decision Tree
dt=OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
dt.fit(X_train_scaled,y_train)
y_pred=dt.predict(X_test_scaled)

fpr_dt,tpr_dt,_=roc_curve(y_test[:,0],y_pred[:,0])
roc_auc_dt=auc(fpr_dt,tpr_dt)
print("Accuracyscore:",accuracy_score(y_test,y_pred))
print("F1score:",f1_score(y_test,y_pred,average='micro'))

y_pred_adv=dt.predict(X_adv)
fpr_dt_adv,tpr_dt_adv,_=roc_curve(y_test[:,0],y_pred_adv[:,0])
roc_auc_dt_adv=auc(fpr_dt_adv,tpr_dt_adv)
print("Accuracyscoreadversarial:",accuracy_score(y_test,y_pred_adv))
print("F1scoreadversarial:",f1_score(y_test,y_pred_adv,average='micro'))
print("AUCscoreadversarial:",roc_auc_dt_adv)

plt.figure()
lw=2
plt.plot(fpr_dt,tpr_dt,color='darkorange',
lw=lw,label='ROCcurve(area=%0.2f)'%roc_auc_dt)
plt.plot(fpr_dt_adv,tpr_dt_adv,color='green',lw=lw,label='ROCcurveadv.(area=%0.2f)'%roc_auc_dt_adv)
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('FalsePositiveRate')
plt.ylabel('TruePositiveRate')
plt.title('ROCDecisionTree(class=Normal)')
plt.legend(loc="lower right")
plt.savefig('ROC_DT.png')

#Random Forest
rf=OneVsRestClassifier(RandomForestClassifier(n_estimators=200,random_state=42))
rf.fit(X_train_scaled,y_train)
y_pred=rf.predict(X_test_scaled)
fpr_rf,tpr_rf,_=roc_curve(y_test[:,0],y_pred[:,0])
roc_auc_rf=auc(fpr_rf,tpr_rf)
print("Accuracyscore:",accuracy_score(y_test,y_pred))
print("F1score:",f1_score(y_test,y_pred,average='micro'))
print("AUCscore:",roc_auc_rf)

y_pred_adv=rf.predict(X_adv)
fpr_rf_adv,tpr_rf_adv,_=roc_curve(y_test[:,0],y_pred_adv[:,0])
roc_auc_rf_adv=auc(fpr_rf_adv,tpr_rf_adv)
print("Accuracyscoreadversarial:",accuracy_score(y_test,y_pred_adv))
print("F1scoreadversarial:",f1_score(y_test,y_pred_adv,average='micro'))
print("AUCscoreadversarial:",roc_auc_rf_adv)

plt.figure()
lw=2
plt.plot(fpr_rf,tpr_rf,color='darkorange',
lw=lw,label='ROCcurve(area=%0.2f)'%roc_auc_rf)
plt.plot(fpr_rf_adv,tpr_rf_adv,color='green',
lw=lw,label='ROCcurveadv.(area=%0.2f)'%roc_auc_rf_adv)
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('FalsePositiveRate')
plt.ylabel('TruePositiveRate')
plt.title('ROCRandomForest(class=Normal)')
plt.legend(loc="lower right")
plt.savefig('ROC_RF.png')

#SVM
sv=OneVsRestClassifier(LinearSVC(C=1.,random_state=42,loss='hinge'))
sv.fit(X_train_scaled,y_train)

y_pred=sv.predict(X_test_scaled)
fpr_sv,tpr_sv,_=roc_curve(y_test[:,0],y_pred[:,0])
roc_auc_sv=auc(fpr_sv,tpr_sv)
print("Accuracyscore:",accuracy_score(y_test,y_pred))

print("F1score:",f1_score(y_test,y_pred,average='micro'))
print("AUCscore:",roc_auc_sv)

y_pred_adv=sv.predict(X_adv)
fpr_sv_adv,tpr_sv_adv,_=roc_curve(y_test[:,0],y_pred_adv[:,0])
roc_auc_sv_adv=auc(fpr_sv_adv,tpr_sv_adv)
print("Accuracyscoreadversarial",accuracy_score(y_test,y_pred_adv))
print("F1scoreadversarial:",f1_score(y_test,y_pred_adv,average='micro'))
print("AUCscoreadversarial:",roc_auc_sv_adv)

plt.figure()
lw=2
plt.plot(fpr_sv,tpr_sv,color='darkorange',
lw=lw,label='ROCcurve(area=%0.2f)'%roc_auc_sv)
plt.plot(fpr_sv_adv,tpr_sv_adv,color='green',
lw=lw,label='ROCcurveadv.(area=%0.2f)'%roc_auc_sv_adv)
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('FalsePositiveRate')
plt.ylabel('TruePositiveRate')
plt.title('ROCSVM(class=Normal)')
plt.legend(loc="lower right")
plt.savefig('ROC_SVM.png')


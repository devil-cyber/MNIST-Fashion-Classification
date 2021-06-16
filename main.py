import sys
from mxnet import gluon,autograd,np,npx
from d2l import mxnet as d2l
import streamlit as st

npx.set_np()
num_inputs = 784
num_outputs = 10
train=gluon.data.vision.FashionMNIST(train=True)
test=gluon.data.vision.FashionMNIST(train=False)
w = np.random.normal(0,0.01,(num_inputs,num_outputs))
b = np.zeros(num_outputs)
w.attach_grad()
b.attach_grad()

class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
def load_data(batch_size,resize=None):
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0,dataset.transforms.Resize(resize))
    trans=dataset.transforms.Compose(trans)
    train=dataset.FashionMNIST(train=True).transform_first(trans)
    test = dataset.FashionMNIST(train=False).transform_first(trans)
    train_iter = gluon.data.DataLoader(train,batch_size,shuffle=True)
    test_iter = gluon.data.DataLoader(test,batch_size,shuffle=True)
    return train_iter,test_iter


def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs,num_rows,num_cols,title=None,scale=1.5):
	cols = st.beta_columns(6) 
	cols[0].image(imgs[0].asnumpy(),width=100,caption=title[0])
	cols[1].image(imgs[1].asnumpy(),width=100,caption=title[1])
	cols[2].image(imgs[2].asnumpy(),width=100,caption=title[2])
	cols[3].image(imgs[3].asnumpy(),width=100,caption=title[3])
	cols[4].image(imgs[4].asnumpy(),width=100,caption=title[4])
	cols[5].image(imgs[5].asnumpy(),width=100,caption=title[5])   
def softmax(X):
    X_exp = np.exp(X)
    num_sum = X_exp.sum(1,keepdims=True)
    return X_exp/num_sum



def model(X):
    return softmax(np.dot(X.reshape((-1,w.shape[0])),w)+b)



def cross_entropy(y_hat,y):
    return -np.log(y_hat[range(len(y_hat)),y])


def  accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype)==y
    return float(cmp.astype(y.dtype).sum())



def evalute_accuracy(model,data_iter):
    metric=Accumulator(2)
    for X,y in data_iter:
        metric.add(accuracy(model(X),y),y.size)
    return metric[0]/metric[1]

def updater(batch_size,lr):
    return d2l.sgd([w,b],lr,batch_size)


def train_epoch(model,train_iter,loss,updater,lr): #@save
    metric = Accumulator(3)
    if isinstance(updater,gluon.Trainer):
        updater = updater.step
    for X,y in train_iter:
        with autograd.record():
            y_hat=model(X)
            l=loss(y_hat,y)
        l.backward()
        updater(X.shape[0],lr)
        metric.add(l.sum(),accuracy(y_hat,y),y.size)
        return metric[0]/metric[2],metric[1]/metric[2]



import streamlit as st
from main import *
from mxnet import npx
from d2l import mxnet as d2l
import time
npx.set_np()



# Hypaerparameter
batch_size=256
train_iter,test_iter = load_data(batch_size)

num_epochs = 1



def run(model,train_iter,loss,num_epochs,updater,lr):
	
	for epoch in range(num_epochs):
        
		train_metric=train_epoch(model,train_iter,loss,updater,lr)
		test_acc=evalute_accuracy(model,test_iter)
	train_loss,train_acc=train_metric
	st.success(f'test_acc:[{test_acc:.2f}, train_acc:{train_acc:.2f}, loss:{train_loss:.2f}]')
    


def predict(model,test_iter,n=3):
    for X,y in test_iter:
        break
    trues=d2l.get_fashion_mnist_labels(y)
    preds=d2l.get_fashion_mnist_labels(model(X).argmax(axis=1))
    titles = ['true_label:\n'+true + '\n' + "predicted:\n"+pred for true,pred in zip(trues,preds)]
    show_images(X[0:n].reshape((n,28,28)),1,n,titles[:n])
if __name__ == '__main__':
	start_train=False
	start_predict = False

	st.title('Fashion MNIST Classification Using Softmax Regression')
	form =st.form(key='my-form')
	with form:
		learn=st.slider(label='select learning_rate',min_value=0.01,max_value=0.05)
		batch=st.slider(label='select batch_size',min_value=32,max_value=256,key=5)
		epoch=st.slider(label='select number of epoch',min_value=5,max_value=50)
		submit = st.form_submit_button('Start Training')
		if submit:
			num_epochs = epoch
			lr = learn
			batch_size = batch
			start_train=True
	if start_train:
		with st.spinner('Training Started....'):
			run(model,train_iter,cross_entropy,num_epochs,updater,lr)
			start_predict=True
	if start_predict:
		predict(model,test_iter,n=6)
	st.write(
        f"<a style='text-align:center' href='https://github.com/devil-cyber/MNIST-Fashion-Classification'>Github</a>",
        unsafe_allow_html=True,)

 	# run(model,train_iter,cross_entropy,num_epochs,updater)
 	# 
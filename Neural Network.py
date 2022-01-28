import numpy as np
import tensorflow as tf
import cv2 as cv
data=tf.keras.datasets.mnist.load_data("mnist.pz")

(x_train_data, y_train_data), (x_test_data, y_test_data)=data
print(y_train_data.shape)


def data_processing(x_data,m):
  x=x_data.reshape(m,-1).T
  x=x/255
  return x


def def_layer(x_train,no_of_unique_output=10):
    input_layer=x_train_data[0].shape[0]*x_train_data[0].shape[1]
    print(input_layer)
    first_hidden_layer=300#no of neurons in the first hidden layer
    second_hidden_layer=200#no of neurons in the second hidden layer
    no_of_output_unit=no_of_unique_output
    return input_layer,first_hidden_layer,second_hidden_layer,no_of_output_unit

def vec_generator(y):
  label=np.zeros(10)
  label[y]=1
  print(label)
  return label


def sigmoid(z):
  s=1/(1+np.exp(-z))
  return s



def training_labels(y_train):
  m=y_train.shape[0]
  labels=[[0,0,0,0,0,0,0,0,0,0]]
  for i in range(m):
    labels=np.append(labels,[vec_generator(y_train[i])],axis=0)
  labels=np.delete(labels,(0),axis=0)
  print('\n \n')
  labels=labels.T
  return labels


def inatialize_parameters(input_layer,first_hidden_layer,second_hidden_layer,no_of_output_layer):
  W1=np.random.randn(first_hidden_layer,input_layer)*0.01
  b1=np.zeros((first_hidden_layer,1))
  W2=np.random.randn(second_hidden_layer,first_hidden_layer)*0.01
  b2=np.zeros((second_hidden_layer,1))
  W3=np.random.randn(no_of_output_layer,second_hidden_layer)*0.01
  b3=np.zeros((no_of_output_layer,1))
  return {"W1":W1,
          "b1":b1,
          "W2":W2,
          "b2":b2,
          "W3":W3,
          "b3":b3,
          }


def fordwardPropagation(x_train,parameters):
  W1=parameters["W1"]
  b1=parameters["b1"]
  W2=parameters["W2"]
  b2=parameters["b2"]
  W3=parameters["W3"]
  b3=parameters["b3"]
  z1=np.dot(W1,x_train)+b1
  a1=np.tanh(z1)
  z2=np.dot(W2,a1)+b2
  a2=np.tanh(z2)
  z3=np.dot(W3,a2)+b3
  a3=sigmoid(z3)
  value={
      "z1":z1,
      "a1":a1,
      "z2":z2,
      "a2":a2,
      "z3":z3,
      "a3":a3

  }
  return value


def tanh(x):
  value=np.tanh(x)
  return value


def backPropagation(fordward_cache,parameters,x_train,training_labels,m):
  z1=fordward_cache['z1']
  a1=fordward_cache['a1']
  z2=fordward_cache['z2']
  a2=fordward_cache['a2']
  z3=fordward_cache['z3']
  a3=fordward_cache['a3']
  W1=parameters["W1"]
  b1=parameters["b1"]
  W2=parameters["W2"]
  b2=parameters["b2"]
  W3=parameters["W3"]
  b3=parameters["b3"]
  dz3=a3-training_labels
  dw3=(1/m)*np.dot(dz3,a2.T)
  db3=(1/m)*np.sum(dz3,axis=1,keepdims=True)
  dz2=(1/m)*np.dot(W3.T,dz3)*(1-np.power(a2,2))#Since the derivate of sigmoid function is (s(x)(1-s(x))) and s(x)=a2
  dw2=(1/m)*np.dot(dz2,a1.T)
  db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)
  dz1=(1/m)*np.dot(W2.T,dz2)*(1-np.power(a1,2))
  dw1=(1/m)*np.dot(dz1,x_train.T)
  db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)
  back_propagation={
      "dw3":dw3,
      "db3":db3,
      "dw2":dw2,
      "db2":db2,
      "dw1":dw1,
      "db1":db1
  }
  return back_propagation


def update_parameters(parameters,learning_rate,back_Propagation):
  W1=parameters["W1"]
  b1=parameters["b1"]
  W2=parameters["W2"]
  b2=parameters["b2"]
  W3=parameters["W3"]
  b3=parameters["b3"]
  dw3=back_Propagation["dw3"]
  db3=back_Propagation["db3"]
  dw2=back_Propagation["dw2"]
  db2=back_Propagation["db2"]
  dw1=back_Propagation["dw1"]
  db1=back_Propagation["db1"]
  W1=W1-learning_rate*dw1
  W2=W2-learning_rate*dw2
  W3=W3-learning_rate*dw3
  b1=b1-learning_rate*db1
  b2=b2-learning_rate*db2
  b3=b3-learning_rate*db3
  return {"W1":W1,
          "b1":b1,
          "W2":W2,
          "b2":b2,
          "W3":W3,
          "b3":b3
          }


def computeCost(fordward_cache,y_train,m):
  a3=fordward_cache['a3']
  cost=(1/m)*(np.sum(-np.multiply(y_train,np.log(a3))-np.multiply((1-y_train),np.log(1-a3))))
  return cost


def model(x_train_data,y_train_data,m,learning_rate):
  x_train=data_processing(x_train_data,m)
  print(x_train.shape)
  y_train=training_labels(y_train_data)
  print(y_train.shape)
  input_layer,first_hidden_layer,second_hidden_layer,no_of_output_layer=def_layer(x_train,no_of_unique_output=10)
  parameters= inatialize_parameters(input_layer,first_hidden_layer,second_hidden_layer,no_of_output_layer)
  for i in range(5000):
    fordward_cache=fordwardPropagation(x_train,parameters)
    cost=computeCost(fordward_cache,y_train,m)
    back_propagation=backPropagation(fordward_cache,parameters,x_train,y_train,m)
    parameters=update_parameters(parameters,learning_rate,back_propagation)
    if i%1000==0:
      print(cost)
  return parameters



parameters=model(x_train_data,y_train_data,x_train_data.shape[0],learning_rate=0.10)


def returningCorrectprediction(predictions):
  max=np.argmax(predictions,axis=0)
  return max

def softmax(predictions):
  exp=np.exp(predictions)
  sum=np.sum(exp,axis=0,keepdims=True)
  soft=exp/sum
  return soft


def predict(x_test_data,y_test_data,parameters):
  x_predict=data_processing(x_test_data,x_test_data.shape[0])
  prediction=fordwardPropagation(x_predict,parameters)
  prediction=prediction['a3']
  print(prediction.shape)
  soft_max=softmax(prediction)
  count=0
  prediction=returningCorrectprediction(soft_max)
  print(prediction)
  print(y_test_data)
  for i in range(0,y_test_data.shape[0]):
    if prediction[i]==y_test_data[i]:
      count=count+1
  accuracy=(count/y_test_data.shape[0])*100
  return accuracy
print("The test accuaracy is "+str(predict(x_test_data,y_test_data,parameters)))
print("The train accuaracy is "+str(predict(x_train_data,y_train_data,parameters)))
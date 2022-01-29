# In the first pase of this of this project, basic neural network was implemented without standarderziation of the input data
## Basic architecture of the neural network is described below
- The first layer is a vector of dimension [784,1] and represent the input from 28*28 iamge 
- The second layer consists of 20 neurons
- The third layer consists of 15 neurons
- The final layer is the output layer with 10 neurons each representing output from 0 to 9
- In each hidden layer and output layer we apply sigmoid function for non-linearity 


#These are the value of loss function of every 1000 iterations using two layer neural network with sigmoid function in each layer
- 6.900024858879152,
- 3.2489403211808363,
- 3.248667464043293,
- 3.2483945210062797,
- 3.2481213619609286,
- 3.2478478569201386,
- 3.2475738759388784,
- 3.247299289035112,
- 3.2470239661113074,
- 3.2467477768764703,


#####We can see that the model is performing very poorly
#####In next phase I will tweak some model parameters and see how good my model is performing

# Phase 2
 
 ##Let's change the number of neurons in hidden layer and see how our model performs
- Changing the no of neurons in each hidden layer to 100 and applying tanh function the loss obtained is as follows
  - 6.930356336410742
  - 3.404633329445414
  - 3.264835574609521
  - 3.2482665371063995
  - 3.2448712120717964
  - 3.243340089667133
  - 3.242101384867591
  - 3.2409111868677
  - 3.239729830763367
  - 3.2385507080301306


  - There is a slight improvement but it is not performing better 
  -The test data accuracy was calculated at 50.22

The train data accuracy was calculated at 48.84 which is highly unusal

The basic conclusion is that the nn is not fitting the training data well so we will increase the network size and increase our training itteration to get a better result

# Phase 3

Tuning hyperparameters to perform better result:
- Changing the hyper-parameters of first layer to 300 and second layer to 200 and running for 5000 iterations the loss function for each 500 iterations are given below 
   - 6.931752200360075
   - 3.217330746399642
   - 3.18574521254387
   - 3.154782748213021
   - 3.124354634623096
   - 3.0944324724254484
   - 3.064990650341306
   - 3.036006142098107
   - 3.007458320233514
   - 2.979328780271083


 The test accuaracy is 65.86999999999999


 The train accuaracy is 64.38166666666667

 We can see that our model is performing much much better than initial model so lets us increase the number of itteration to get better result
# Phase 4
- Running the model for 10000 itterations instead of 5000 the follweing result were obtained
    - 6.930226273573128
    - 3.1863312288646224
    - 3.125736615352614
    - 3.0671393307736134
    - 3.0103473940689494
    - 2.9552065728811443
    - 2.901595354101548
    - 2.849420331102336
    - 2.798611909204409
    - 2.7491202786336224 
 - The test accuaracy is 70.73
 - The train accuaracy is 69.17333333333333
 ### The model is performing slightly better but the computational time is getting longer

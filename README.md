java c
GR5242 HW01 Problem 1: Basics 
Instructions: This problem is an individual assignment -- you are to complete this problem on your own, without conferring with your classmates. You should submit a completed and published notebook to Courseworks; no other files will be accepted.
Description: The goal of this problem is to get your familiar with neural network training from end to end.
Our main tool is torch , especially torch.nn and torch.optim , that helps us with model building and automatic differentiation / backpropagation.
There are 4 questions in this notebook, including 3 coding quesitons and 1 text question. Each coding question expects 1~3 lines of codes, and the text question expects just 1 sentence of explanation.In [ ]:# PyTorch imports:## torch is the base package, nn gives nice classes for Neural Networks,# F contains our ReLU function, optim gives our SG method,# DataLoader allows us to do batches efficiently,# and torchvision is for downloading MNIST data directly from PyTorchimport torchfrom torch import nnfrom torch.nn import functional as Fimport torch.optim as optimfrom torch.utils.data import DataLoaderimport torchvision.transforms as transformsimport torchvision.datasets as datasets# Helper librariesimport numpy as npimport matplotlib.pyplot as pltprint(torch.__version__)
Dataset 
We will working on mnist dataset, which contain images of written digits of 0-9 and corresponding labels.
We have it set up to download the data directly from the torch library.In [ ]:# First, we will define a way of transforming the dataset automatically# upon downloading from pytorch# first convert an image to a tensor and then scale its values to be between -1transform. = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])# Next, we fetch the datamnist_train = datasets.MNIST(root='./data', train=True,download=True, transform=transform)mnist_test = datasets.MNIST(root='./data', train=False,download=True, transform=transform)# and define our DataLoaderstrain_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)test_loader = DataLoader(mnist_test, batch_size=32, shuffle=True)
Each image is represented as a 28x28 matrix of pixel values, and each label is the corresponding digit.
Let's show an image of a random one! Try running the below cell a few times to see different examples and how the DataLoaders will be shuffling batches.
Note: Why is this random, when there is no random code in the next cell? The randomness comes from shuffle=True in the train_loader !In [ ]:inputs, classes = next(iter(train_loader))plt.imshow(inputs[23].squeeze())plt.title('Training label: '+str(classes[23].item()))plt.show()
Let's now show 25 of them in black and white:In [ ]:plt.figure(figsize=(10,10))for i in range(25):plt.subplot(5,5,i+1)plt.xticks([])plt.yticks([])plt.grid(False)plt.imshow(inputs[i].squeeze(), cmap=plt.cm.binary)plt.xlabel(classes[i].item())plt.show()
By printing out the shapes, we see there are 60,000 training data and 10,000 test data. Each image is represented as a 28x28 matrix of pixel values, and each label is the corresponding digit.In [ ]:# For training datatrain_data_example, train_label_example = mnist_train[0]print("Shape of a single training image:", train_data_example.shape)# For test datatest_data_example, test_label_example = mnist_test[0]print("Shape of a single test image:", test_data_example.shape)# The total number of images in each datasetprint("Total number of training images:", len(mnist_train))print("Total number of test images:", len(mnist_test))Recap of classification task
In a classification task with K classes, suppose the predicted logits for an image are s1, ..., sK. The predicted probabilities are then

The CrossEntropy (CE) loss is defined as

where ti = 1 if the image belongs to the th class or otherwise ti = 0.
Model 
Now, we will build a model to predict the logits of images for the classificaiton task.
Question 1: Building the Model 
In the following, we will write a class for a basic one-hidden-layer, ReLU, feedforward network. There are a few components to a model in Pytorch, and we will break them down step by step.
First, we need to define the class. As with any class definition, we start with an __init__ method. Since Pytorch provides us with many useful features within the torch.nn.Module class, we will use inheritence to pass these down to our Net class. This involves putting nn.Module inside the parenthesis in the class definition, and a super().__init__() call in the __init__() method.
Within the initialization, we then define two layers: one hidden layer with 128 neurons,代 写GR5242 HW01 Problem 1: BasicsR
代做程序编程语言 and one output layer with 10 class logits. The hidden layer should take an input of size 28 x 28 and give an output of size 128 , while the output layer takes input of size 128 and gives output of size 10 . It is suggested to use the nn.Linear() object to accomplish this, which applies a transformation z = xWT + b.
Next, we define a special method called forward(), which defines how data propagate through the model. This method will be called either by model.forward(x) or by model(x) , and is where Pytorch looks for the information for its automatic derivative computation capabilities.
In the forward method, we first will reshape our image img using img.view() .
Then, we will apply the hidden layer (the one we defined) and the ReLU function F.relu .
Finally, we apply the output layer and return our output. Importantly, do not apply SoftMax to the output just yet. We will handle that part laterIn [ ]:class Net(nn.Module):def __init__(self):super(Net, self).__init__()### YOUR CODE HERE #### define hidden layer and output layer below:######################def forward(self, img):x = img.view(-1, 28*28) # reshape the image to be a single row# pass x through both layers, with ReLU in between### YOUR CODE HERE #########################return xmodel = Net()
Question 2: Defining the Loss and Optimizer 
When training a torch model, typically you need to specify the following two items:
optimizer: specifies a way to apply gradient descent update of model parameters. We will use the optim.Adam optimizer with a learning rate of 0.001 in this example.
loss_fn: the objective function to minimize over. In classification task, the cross-entropy loss is used.
Please fill in the optimizer with an appropriate learning rate lr , and choose an appropriate number of epochs (number of passes through the data) in the following code.
Note: remember that the neural network outputs the logits instead of the class probabilities (why? answer the question below), and make sure to specify this in the loss function .In [ ]:loss_fn = nn.CrossEntropyLoss()### YOUR CODE HERE #########################
Question 3: The neural network specified above does not output class probabilities, because the last layer of the neural network is a linear layer which outputs value ranging from (-∞, ∞). Your choice of loss function above should take care of that, but what mathematical function maps these logit values to class probabilities? 
#
YOUR ANSWER HERE
#
Training 
Now let's train the model for your chosen number of epochs. By the end of the training, you should expect an accuracy above 0.98.
In each step, we need to:
1.) grab x and y from the batch (note that each batch is a tuple of x and y )
2.) zero the optimizer's gradients
3.) make a prediction y_pred
4.) call the loss_fn between y and y_pred
5.) backpropagate
6.) make the approprite step calculated by the optimizerIn [ ]:epochs = 10for epoch in range(epochs):losses = []accuracies = []for batch in train_loader:correct, total = 0, 0x_batch, y_batch = batchoptimizer.zero_grad()### YOUR CODE HERE #########################for index, output in enumerate(y_logit):y_pred = torch.argmax(output)if y_pred == y_batch[index]:correct += 1total += 1### YOUR CODE HERE #########################loss.backward()optimizer.step()losses.append(loss.item())accuracies.append(correct/total)avg_loss = np.mean(np.array(losses))avg_accuracy = np.mean(np.array(accuracies))print('epoch ' + str(epoch+1) + ' average loss: ', avg_loss,'-- average accuracy: ', avg_accuracy)Test Evaluation
Finally, we evaluate our model on the test set. You could expect the test accuracy to be slightly lower than the training accuracy.In [ ]:with torch.no_grad():correct = 0total = 0for batch in test_loader:x_batch, y_batch = batchy_logit = model(x_batch)for index, output in enumerate(y_logit):y_pred = torch.argmax(output)if y_pred == y_batch[index]:correct += 1total += 1print('testing accuracy:', correct/total)Make Prediction
Question 4: fill in the following code block to estimate class probabilities and make predictions on test images. The results should be stored in class_probabilities and predicted_labels . Compare to the true labels, stored in true_labels by computing the accuracy. It should be the same as above.
(Hint: you can use much of the same structure from the cell above. You can use F.softmax to calculate probabilities from the logits, and store the results however you please.)In [ ]:### YOUR CODE HERE ###########################print('accuracy verification: ', sum(true_labels==predicted_labels)/len(true_la





         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com

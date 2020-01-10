library(dslabs)
library(caret)
mnist <- read_mnist()
data <- mnist$train$images
dataindex <- mnist$train$labels
sample_4 <- matrix(as.numeric(data[4,]), nrow = 28, byrow = TRUE)
image(sample_4, col = grey.colors(255))
sample_700 <- matrix(as.numeric(data[700,]), nrow = 28, byrow = TRUE)
image(sample_700, col = grey.colors(255))
# Rotate the matrix by reversing elements in each column
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(sample_700), col = grey.colors(255))
# Transform target variable "label" from integer to factor, in order to perform classification
is.factor(dataindex)
dataindex <- as.factor(dataindex)
summary(dataindex)
proportion <- prop.table(table(dataindex)) * 100
cbind(count=table(dataindex), proportion=proportion)

#git init
#git remote add origin "https://github.com/Wenlong-Y/MNIST"
library(dslabs)
library(caret)
mnist <- read_mnist()
data <- mnist$train$images
dataindex <- mnist$train$labels
sample_4 <- matrix(as.numeric(data[4,]), nrow = 28, byrow = TRUE)
image(sample_4, col = grey.colors(255))
sample_700 <- matrix(as.numeric(data[700,]), nrow = 28, byrow = TRUE)
image(sample_700, col = grey.colors(255))
# Rotate the matrix by reversing elements in each column
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(sample_700), col = grey.colors(255))
# Transform target variable "label" from integer to factor, in order to perform classification
is.factor(dataindex)
dataindex <- as.factor(dataindex)
summary(dataindex)
proportion <- prop.table(table(dataindex)) * 100
cbind(count=table(dataindex), proportion=proportion)

#git init
#git remote add origin "https://github.com/Wenlong-Y/MNIST"

par(mar=c(2,2,2,2))
for(i in 1:9) {
  hist(c(as.matrix(data[as.integer(dataindex)==i, central_block])),
       main=sprintf("Histogram for digit %d", i),
       xlab="Pixel value")
}

library(caret)
# set.seed(42, sample.kind = "Rounding")
# train_perc = 0.75

library(nnet)

model_lr <- multinom(dataindex ~ ., data=data.frame(data), MaxNWts=10000, decay=5e-3, maxit=100)
prediction_lr <- predict(model_lr, data.frame(mnist$test$images),type="class")
prediction_lr[1:5]
mnist$test$labels[1:5]
cm_lr = table(mnist$test$labels,prediction_lr)
cm_lr
accuracy_lr = mean( prediction_lr == mnist$test$labels)
accuracy_lr

#too many weights
#mnisttrain=data.frame(dataindex,data)
#train_glm <- train(dataindex ~., method="multinom",data=mnisttrain)

#single layer neural network
model_nn <- nnet(dataindex ~ ., data=data, size=50, maxit=300, MaxNWts=100000, decay=1e-4)
prediction_nn <- predict(model_nn, data.frame(mnist$test$images), type = "class")
cm_nn <- table(mnist$test$label, prediction_nn)
cm_nn
accuracy_nn = mean(prediction_nn == mnist$test$label)
accuracy_nn



#multilayer neural network, need new library to do that.

data <- read.csv ("train.csv")
#set.seed(42, sample.kind = "Rounding")
set.seed(42)
train_perc = 0.75
train_index <- createDataPartition(data$label, p=train_perc, list=FALSE)
data_train <- data[train_index,]
data_test <- data[-train_index,]

library(mxnet)
data_train <- data.matrix(data_train)
data_train.x <- data_train[,-1]
data_train.x <- t(data_train.x/255)
data_train.y <- data_train[,1]

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
devices <- mx.cpu()
mx.set.seed(42)
model_dnn <- mx.model.FeedForward.create(softmax, X=data_train.x, y=data_train.y, ctx=devices, num.round=30, array.batch.size=100,learning.rate=0.01, momentum=0.9, eval.metric=mx.metric.accuracy,initializer=mx.init.uniform(0.1),epoch.end.callback=mx.callback.log.train.metric(1000))

data_test.x <- data_test[,-1]
data_test.x <- t(data_test.x/255)

prob_dnn <- predict(model_dnn, data_test.x)
prediction_dnn <- max.col(t(prob_dnn)) - 1
cm_dnn = table(data_test$label, prediction_dnn)
cm_dnn
accuracy_dnn = mean(prediction_dnn == data_test$label)
accuracy_dnn


#for the original mnist data, the folowing command can be used
#model_dnn <- mx.model.FeedForward.create(softmax, X=t(mnist$train$images/255), y=mnist$train$labels, ctx=devices, num.round=30, array.batch.size=100,learning.rate=0.01, momentum=0.9, eval.metric=mx.metric.accuracy,initializer=mx.init.uniform(0.1),epoch.end.callback=mx.callback.log.train.metric(1000))
#prob_dnn <- predict(model_dnn, t(mnist$test$images/255))
#prediction_dnn <- max.col(t(prob_dnn)) - 1
#cm_dnn = table(mnist$test$labels, prediction_dnn)
#cm_dnn
#accuracy_dnn = mean(prediction_dnn == mnist$test$labels)
#accuracy_dnn

#CNN
# first convolution
data <- mx.symbol.Variable("data")
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5),num_filter=20)
act1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=act1, pool_type="max",kernel=c(2,2), stride=c(2,2))

# second convolution
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5),num_filter=50)
act2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=act2, pool_type="max",kernel=c(2,2), stride=c(2,2))

flatten <- mx.symbol.Flatten(data=pool2)

# first fully connected layer
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
act3 <- mx.symbol.Activation(data=fc1, act_type="relu")

# second fully connected layer
fc2 <- mx.symbol.FullyConnected(data=act3, num_hidden=10)

# softmax output
softmax <- mx.symbol.SoftmaxOutput(data=fc2, name="sm")

#data manipulation
train.array <- data_train.x
dim(train.array) <- c(28, 28, 1, ncol(data_train.x))
mx.set.seed(42)
model_cnn <- mx.model.FeedForward.create(softmax, X=train.array,y=data_train.y, ctx=devices, num.round=30,array.batch.size=100, learning.rate=0.05,momentum=0.9, wd=0.00001,eval.metric=mx.metric.accuracy,epoch.end.callback=mx.callback.log.train.metric(100))


test.array <- data_test.x
dim(test.array) <- c(28, 28, 1, ncol(data_test.x))
prob_cnn <- predict(model_cnn, test.array)
prediction_cnn <- max.col(t(prob_cnn)) - 1
cm_cnn = table(data_test$label, prediction_cnn)
cm_cnn
accuracy_cnn = mean(prediction_cnn == data_test$label)
accuracy_cnn



# Visualize the model

graph.viz(model_cnn$symbol)



# Plot learning curve

data_test.y <- data_test[,1]
logger <- mx.metric.logger$new()
model_cnn <- mx.model.FeedForward.create(softmax, X=train.array, y=data_train.y,eval.data=list(data=test.array, label=data_test.y), ctx=devices, num.round=30, array.batch.size=100,learning.rate=0.05, momentum=0.9, wd=0.00001, eval.metric=mx.metric.accuracy, epoch.end.callback = mx.callback.log.train.metric(1, logger))

plot(logger$train,type="l",col="red", ann=FALSE)
lines(logger$eval,type="l", col="blue")
title(main="Learning curve")
title(xlab="Iterations")
title(ylab="Accuary")
legend(20, 0.5, c("training","testing"), cex=0.8, 
       col=c("red","blue"), pch=21:22, lty=1:2);



# Visualize convolutional layers
par(mfrow=c(1,2))  #set screen distribution
test_1 <- matrix(as.numeric(data_test[1,-1]), nrow = 28, byrow = TRUE)
image(rotate(test_1), col = grey.colors(255))
test_2 <- matrix(as.numeric(data_test[2,-1]), nrow = 28, byrow = TRUE)
image(rotate(test_2), col = grey.colors(255))


layerss_for_viz <- mx.symbol.Group(c(conv1, act1, pool1, conv2, act2, pool2, fc1, fc2))
executor <- mx.simple.bind(symbol=layerss_for_viz, data=dim(test.array), ctx=mx.cpu())

mx.exec.update.arg.arrays(executor, model_cnn$arg.params, match.name=TRUE)
mx.exec.update.aux.arrays(executor, model_cnn$aux.params, match.name=TRUE)

mx.exec.update.arg.arrays(executor, list(data=mx.nd.array(test.array)), match.name=TRUE)
mx.exec.forward(executor, is.train=FALSE)



names(executor$ref.outputs)

# Plot the activation of the testing samples
par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array(executor$ref.outputs$activation0_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n', col=grey.colors(255)
  )
}

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array(executor$ref.outputs$activation0_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n', col=grey.colors(255)
  )
}


par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array(executor$ref.outputs$pooling0_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n', col=grey.colors(255)
  )
}

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array(executor$ref.outputs$pooling0_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n', col=grey.colors(255)
  )
}



par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array(executor$ref.outputs$convolution1_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n', col=grey.colors(255)
  )
}

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array(executor$ref.outputs$convolution1_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n', col=grey.colors(255)
  )
}



validation_perc = 0.4
validation_index <- createDataPartition(data_test.y, p=validation_perc, list=FALSE)

validation.array <- test.array[, , , validation_index]
dim(validation.array) <- c(28, 28, 1, length(validation.array[1,1,]))
data_validation.y <- data_test.y[validation_index]
final_test.array <- test.array[, , , -validation_index]
dim(final_test.array) <- c(28, 28, 1, length(final_test.array[1,1,]))
data_final_test.y <- data_test.y[-validation_index]


mx.callback.early.stop <- function(eval.metric) {
  function(iteration, nbatch, env, verbose) {
    if (!is.null(env$metric)) {
      if (!is.null(eval.metric)) {
        result <- env$metric$get(env$eval.metric)
        if (result$value >= eval.metric) {
          return(FALSE)
        }
      }
    }
    return(TRUE)
  }
}



model_cnn_earlystop <- mx.model.FeedForward.create(softmax, X=train.array, y=data_train.y,
                                                   eval.data=list(data=validation.array, label=data_validation.y),
                                                   ctx=devices, num.round=30, array.batch.size=100,
                                                   learning.rate=0.05, momentum=0.9, wd=0.00001,
                                                   eval.metric=mx.metric.accuracy,
                                                   epoch.end.callback = mx.callback.early.stop(0.985))



prob_cnn <- predict(model_cnn_earlystop, final_test.array)
prediction_cnn <- max.col(t(prob_cnn)) - 1
cm_cnn = table(data_final_test.y, prediction_cnn)
cm_cnn
accuracy_cnn = mean(prediction_cnn == data_final_test.y)
accuracy_cnn






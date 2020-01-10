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
set.seed(42, sample.kind = "Rounding")
train_perc = 0.75
train_index <- createDataPartition(data$label, p=train_perc, list=FALSE)
data_train <- data[train_index,]
data_test <- data[-train_index,]

library(mxnet)
data_train <- data.matrix(data_train)
data_train.x <- data_train[,-1]
data_train.x <- data_train.x/255
data_train.y <- data_train[,1]

data1 <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data1, name="fc1", num_hidden=128)
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
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5),num_filter=20)
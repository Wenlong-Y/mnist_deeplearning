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
library(keras)
library(tensorflow)


install_keras()
install_tensorflow()

library(devtools)
devtools::install_github("xrobin/mnist")
library(mnist)
data("mnist")




#naming
c(c(train_images,train_labels),c(test_images,test_labels))%<-%mnist

#rearrange and convert from 3-d to 2-d
#train_images<-array_reshape(train_images,c(60000,28*28))
#train_images<-train_images/255#normalizing (0-1)
#test_images<-array_reshape(test_images,c(60000,28*28))
#test_images<-test_images/255



#reformating to 1s and 0s fro the answer (showing as the probabilities)
#we do this to keep it in the same format the model gives its results
#so we can compare
train_labels<-to_categorical(train_labels)
test_labels<-to_categorical(test_labels)


#model
library(dplyr)


network<-keras_model_sequential()%>%
  layer_dense(units=512,activation = "relu",input_shape = c(28*28))%>%
  layer_dense(units = 10,activation = "softmax")


summary(network)



network%>%compile(
  optimizer="rmsprop",
  loss="categorical_crossentropy",
  metrics=c("accuracy")
  )


#train

history<-network%>%
  fit(train_images,train_labels,epochs=5,batch_size=128)


#test

metrics<-network%>%evaluate(test_images,test_labels)
metrics


#predict

network%>%predict_classes(test_images[1:10,])



predictions<-network%>%predict_classes(test_images)
actual<-mnist$test$y
sum(predictions!=actual)


confusionMatrix(as.factor(predictions),as.factor(actual))



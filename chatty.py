#Import all dependencies
import tensorflow
import tflearn
import random
import numpy
import nltk
import json

#Define the SnowballStemmer for word stemming
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


with open("conversations.json", "r") as myfile:
    data = json.load(myfile)

#Lists for storing the unique words from the conversations file
all_words = []
all_labels = []
first_set = []
second_set = []

#Tokenization, stemming, and sorting processes
for conversation in data['conversations']:
    for each_user_input in conversation['user_input']:
        tokenized_inputs = nltk.word_tokenize(each_user_input)
        all_words.extend(tokenized_inputs)
        first_set.append(tokenized_inputs)
        second_set.append(conversation["label"])

    if conversation['label'] not in all_labels:
        all_labels.append(conversation['label'])

for each_word in all_words:
	all_words = stemmer.stem(each_word.lower())

#Start the training process
training_data = []
output_data = []

empty_list = [0 for i in range(len(all_labels))]

for each_topic, each_doc in enumerate(first_set):
    bag_of_words = []

    for new_words in each_doc:
        tokenized_inputs = stemmer.stem(new_words.lower())

    for new_words in all_words:
        if new_words in tokenized_inputs:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)

#Generate output
    chat_output = list(empty_list)
    chat_output[all_labels.index(second_set[each_topic])] = 1
    
    training_data.append(bag_of_words)
    output_data.append(chat_output)

training = numpy.array(training_data)
output = numpy.array(output_data)

#Build the model structure
neural_net = tflearn.input_data(shape=[None, len(training[0])])
neural_net = tflearn.fully_connected(neural_net, 12)
neural_net = tflearn.fully_connected(neural_net, 12)
neural_net = tflearn.fully_connected(neural_net, len(output[0]), activation="softmax")
neural_net = tflearn.regression(neural_net)

model = tflearn.DNN(neural_net)
model.fit(training, output, n_epoch=2000, batch_size=10, show_metric=True)
model.save("chatbotmodel.tflearn")

#Calculate the predictions
def bow(sentence, all_words):
    bag_of_words = [0 for i in range(len(all_words))]

    tokenized = nltk.word_tokenize(sentence)
    
    for word in tokenized:
        tokenized = stemmer.stem(word.lower()) 

    for sent in tokenized:
        for i, current_word in enumerate(all_words):
            if current_word == sent:
                bag_of_words[i] = 1
            
    return numpy.array(bag_of_words)

#Set up the chat
def activate_chatty():
    print("Hi there! I'm Chatty, nice to meet you!(Please type leave_chat to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "leave_chat":
            break

        chatty_results = model.predict([bow(inp, all_words)])
        chatty_results_index = numpy.argmax(chatty_results)
        labels = all_labels[chatty_results_index]

        for lb in data["conversations"]:
            if lb['label'] == labels:
                chatty_responses = lb['chatbot_output']

        print(random.choice(chatty_responses))

#Start talking!
activate_chatty()

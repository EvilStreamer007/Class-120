import nltk
import json
import pickle
import numpy as np
import random 

ignore_words = ["?","!",",",".","'s","'m"]

import tensorflow 
from data_preprocessing import get_stem_words
model = tensorflow.keras.models.load_model("./chatbot_model.h5")

intents = json.loads(open("./intents.json").read())
words = pickle.load(open("./words.pkl","rb"))
classes = pickle.load(open("./classes.pkl","rb"))

def preprocess_userinput(user_input):
    input_word_token1 = nltk.word_tokenize(user_input)
    input_word_token2 = get_stem_words(input_word_token1, ignore_words)
    input_word_token2 = sorted(list(set(input_word_token2)))
    bag = []
    bag_of_words = []
    for word in words:
        if word in input_word_token2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def botclass_prediction(user_input):
    input = preprocess_userinput(user_input)
    prediction = model.predict(input)
    predictedclasslabel = np.argmax(prediction[0])
    return predictedclasslabel 


def botresponse(user_input):
    predictedclasslabel = botclass_prediction(user_input)
    predicted_class = classes[predictedclasslabel]
    for intent in intents["intents"]:
        if intent["tag"] == predicted_class:
            botresponse = random.choice(intent["responses"])
            return botresponse

print("Hi, I am chatbot. How can I help you?")
while True:
    user_input = input("Type your message here: ")
    print("User Input:",user_input)
    response = botresponse(user_input)
    print("Bot Response: ", response)

import os
import telebot
import pandas as pd
from infer import get_prediction


BOT_TOKEN = os.environ['BOT_TOKEN']
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Hello boy")

@bot.message_handler(commands=['sensitive_content'])
def get_input(message):
    #send question
    text = "What is your passage?"
    sent_msg = bot.send_message(message.chat.id, text)
    #message, callback, *args  
    bot.register_next_step_handler(sent_msg, main)

def main(message):
    #get passage
    input = message.text
    outputs = get_prediction(input)
    print("MY WARNING's MAYBE WRONG, BE CAUTIOUS!!!")
    dict = {'text': [], 'label': [], 'probability': []}
    for (input, output, prob) in outputs:
        dict['text'].append(input)
        dict['label'].append(output)
        dict['probability'].append(prob.cpu())
    print(pd.DataFrame.from_dict(dict))



bot.infinity_polling()
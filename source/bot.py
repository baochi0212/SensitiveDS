import os
import telebot
import pandas as pd
from telegram.constants import ParseMode
from prettytable import PrettyTable
from infer import get_prediction


BOT_TOKEN = os.environ['BOT_TOKEN']
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['hello'])
def send_welcome(message):
    bot.reply_to(message, "Hello kiddo!")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Let's working man")

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
    table = PrettyTable(['text', 'label', 'probability'])
    for (input, output, prob) in outputs:
        table.add_row([input.strip()[:20] + '...', output, prob])
    bot.send_message(message.chat.id, "MY WARNING's MAYBE WRONG, BE CAUTIOUS!!!")
    bot.reply_to(message, f'<pre>{table}</pre>', parse_mode=ParseMode.HTML)



bot.infinity_polling()
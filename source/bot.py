import os
import telebot
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
    output, signal = get_prediction(input)
    print("SIGNAL", signal)
    output = ','.join(output)
    bot.reply_to(message, f"Content tag:   {output}")



bot.infinity_polling()
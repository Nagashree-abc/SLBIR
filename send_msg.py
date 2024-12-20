import telepot


def sendtoTelegram( msg ):

    token = '7202390512:AAGKUnfs5z3OQCQl4P_xtfEYlkPy6waRCio' # telegram token
    receiver_id = 5776612228 # https://api.telegram.org/bot<TOKEN>/getUpdatescond 


    bot = telepot.Bot(token)

    bot.sendMessage(receiver_id, msg) # send a activation message to telegram receiver id

# bot.sendPhoto(receiver_id, photo=open('test_img.png', 'rb')) # send message to telegram




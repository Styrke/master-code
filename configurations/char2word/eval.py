from configurations.char2word import char2word


class Model(char2word.Model):
    # overwrite config
    name = 'c2wtest/char2word'
    iterations = 32000*5

    char_enc_units = 400
    word_enc_units = 400
    dec_units = 400

from configs.char2char import char2char


class Model(char2char.Model):
    # overwrite config
    name = 'char2char/seqlen-50'

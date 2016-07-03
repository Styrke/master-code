from configs import default


class Model(default.Model):
    # overwrite config
    name = 'wmt/tok-250'
    batch_size = 128
    seq_len = 250
    valid_freq = 1500
    train_x_files = ['data/train/europarl-v7.de-en.en.tok',
                     'data/train/commoncrawl.de-en.en.tok',
                     'data/train/news-commentary-v10.de-en.en.tok']
    train_t_files = ['data/train/europarl-v7.de-en.de.tok',
                     'data/train/commoncrawl.de-en.de.tok',
                     'data/train/news-commentary-v10.de-en.de.tok']
    valid_x_files = ['data/valid/newstest2014.deen.en.tok']
    valid_t_files = ['data/valid/newstest2014.deen.de.tok']

    # settings that are local to the model
    char_encoder_units = 400  # number of units in character-level encoder
    word_encoder_units = 400  # num nuits in word-level encoders (both forwards and back)
    attn_units = 300  # num units used for attention in the decoder.

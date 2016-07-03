from configs import default


class Model(default.Model):
    # overwrite config
    name = 'seqlen/size-50'
    train_x_files = ['data/train/europarl-v7.de-en.en.tok',
                     'data/train/commoncrawl.de-en.en.tok',
                     'data/train/news-commentary-v10.de-en.en.tok']
    train_x_files = ['data/train/europarl-v7.de-en.de.tok',
                     'data/train/commoncrawl.de-en.de.tok',
                     'data/train/news-commentary-v10.de-en.de.tok']
    valid_x_files = ['data/valid/newstest2014.deen.en.tok']
    valid_t_files = ['data/valid/newstest2014.deen.de.tok']

    # settings that are local to the model
    alphabet_src = Alphabet('data/alphabet/dict_wmt_tok.de-en.en', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_wmt_tok.de-en.de', eos='*', sos='')
    char_encoder_units = 400  # number of units in character-level encoder
    word_encoder_units = 400  # num nuits in word-level encoders (both forwards and back)
    attn_units = 300  # num units used for attention in the decoder.

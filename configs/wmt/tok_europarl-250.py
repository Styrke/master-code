from configs import default


class Model(default.Model):
    # overwrite config
    name = 'wmt/tok_europarl-250'
    batch_size = 128
    seq_len = 250
    valid_freq = 1500
    train_x_files = ['data/train/europarl-v7.de-en.en.tok']
    train_x_files = ['data/train/europarl-v7.de-en.de.tok']
    valid_x_files = ['data/valid/newstest2014.deen.en.tok']
    valid_t_files = ['data/valid/newstest2014.deen.de.tok']

    # settings that are local to the model
    alphabet_src = Alphabet('data/alphabet/dict_wmt_tok.de-en.en', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_wmt_tok.de-en.de', eos='*', sos='')
    char_encoder_units = 400  # number of units in character-level encoder
    word_encoder_units = 400  # num nuits in word-level encoders (both forwards and back)
    attn_units = 300  # num units used for attention in the decoder.

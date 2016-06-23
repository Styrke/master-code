from configs.attn_units import attention_units


class Model(attention_units.Model):
    # overwrite config
    name = 'attn_units/s-200'
    attn_units = 200

from wenet.codec.soundstorm import SoundStorm
from wenet.transformer.encoder import ConformerEncoder


def init_model(configs, encoder: ConformerEncoder):

    assert 'model' in configs
    model = SoundStorm(conformer=encoder, **configs['model_conf'])
    return model

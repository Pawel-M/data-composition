import argparse

import torch
from transformers import MarianMTModel
from transformers import MarianTokenizer

from config_utils import load_config, load_configs
from data.loading import load_dataset
from evaluating.translate import translate_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    tokenizer_path = config.model_path if config.tokenizer_path is None else config.tokenizer_path

    tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
    model = MarianMTModel.from_pretrained(config.model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = torch.compile(model)

    print('tokenizer', tokenizer)
    print('model', model)

    if tokenizer.sep_token is None:
        print('Tokenizer does not have sep_token set!!! The empty separator will be used instead.')

    dataset = load_dataset(dataset_name=config.dataset, config=config)

    results_file_stem = f'{config.dataset}' if config.results_suffix is None \
        else f'{config.dataset}.{config.results_suffix}'
    translations, infos = translate_dataset(dataset,
                                            results_dir=config.results_dir,
                                            results_file_stem=results_file_stem,
                                            model=model,
                                            tokenizer=tokenizer,
                                            src_lang=config.src_lang,
                                            tgt_lang=config.tgt_lang,
                                            src_ctx_size=config.src_ctx_size,
                                            tgt_ctx_size=config.tgt_ctx_size,
                                            max_length=config.max_length,
                                            num_beams=config.beam_size,
                                            batch_size=config.batch_size,
                                            device=device,
                                            full_sequence_decoding=config.full_sequence_decoding,)

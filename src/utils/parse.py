import yaml
from typing import Optional, Any
from src.utils.constants import CONFIG_PATHS, DATASET_DGES, IMPLEMENTED_MODELS, DATASET_PATHS, DATASET_METADATA_NAMES, DATASET_PROMPTS, DATASET_TOKENIZERS
import os.path
from src.modules.base_generative_model import GenerationContext, dge_name

def read_yaml_config(kwargs: dict, name_in_kwargs: str, default_path: str, verbose: bool = True) -> dict:
    """ Reads a yaml config file whose name is kwargs[name_in_kwargs] or, if the key is not found, default_path
    
    Args:
        kwargs (dict): dictionary (e.g. of command line arguments)
        name_in_kwargs (str): key to the path of the config file in kwargs
        default_path (str): default path to the config file
        verbose (bool): print or not
        
    Returns:
        dict: config dict from a yaml file
    """
    if name_in_kwargs in kwargs:
        config_path = kwargs[name_in_kwargs]
        if verbose:
            print('[X] A path to a yaml', name_in_kwargs, 'was provided:', config_path)
    else:
        config_path = default_path
        if verbose:
            print('[X] A path to a yaml', name_in_kwargs, 'can be provided using key', '--' + name_in_kwargs + '.', 'Using the default path:', config_path)
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    return config

def dict_overwrite(default_dict: dict, update_dict: dict, default_dict_name: Optional[str] = None, verbose: bool = True) -> dict:
    """ Overwrites values in d1 with values in d2 if the keys are the same.
    
    Args:
        default_dict (dict): dictionary to be updated
        update_dict (dict): dictionary with new values
        default_dict_name (Union[str, None]): name of the default_dict
        verbose (bool): whether to print the keys that are being overwritten
        
    Returns:
        dict: updated default_dict
    """
    default_dict_name = default_dict_name if default_dict_name is not None else 'dictionary'
    for key, value in update_dict.items():
        if key in default_dict:
            old_value = default_dict[key]
            default_dict[key] = value
            if verbose:
                print('[X] Overwriting', key, old_value, 'with', value, 'in', default_dict_name)
    return default_dict

def unused_args(kwargs: dict, configs: dict[str, dict[str, Any]], verbose: bool = True) -> list[str]:
    """ Prints the command line arguments that were not used

    Args:
        kwargs (dict): command line arguments
        configs (dict[str, dict[str, Any]]): all needed config dictionaries
        verbose (bool): print or not
        
    Returns:
        List[str]: unused command line arguments
    """
    config_keys = []
    unused = []
    found_unused = False
    for _, config in configs.items():
        config_keys += config.keys()
    for key in kwargs.keys():
        if key not in config_keys and key[8:] not in configs['trainer_config']:
            unused.append(key)
            found_unused = True
    if verbose and len(unused) > 0:
        print('[X] The following command line arguments were not used:', unused)
    if not found_unused and verbose:
        print('[X] Found no unused command line arguments')
    return unused

def parse(**kwargs) -> dict[str, dict[str, Any]]:
    """ Parses command line arguments when called with fire and returns config dictionaries
    
    Args:
        kwargs (dict): command line arguments
        
    Returns:
        Dict[str, Dict[str, Any]]: config dictionaries
    """
    
    print('\n------ Parsing info ------')
    
    if 'model' in kwargs:
        model_name = kwargs['model']
        if model_name not in IMPLEMENTED_MODELS:
            raise ValueError('The model name provided is not valid. It must be one of the following:', IMPLEMENTED_MODELS)
        else:
            print('[X] A model name was provided:', model_name)
    else:
        raise ValueError('[X] A model name must be provided in the command line arguments using --model')
    
    if 'data' in kwargs:
        data = kwargs['data']
        data_path = DATASET_PATHS[data] if data in DATASET_PATHS else data
        if data_path[-5:] != '.h5ad' or not os.path.isfile(data_path):
            raise ValueError('The data file provided is not a h5ad or does not exist:', data_path)
        print('[X] A path to a data file was provided:', data_path)
    else:
        raise ValueError('[X] A dataset must be provided in the command line arguments using --data')
    
    prompts = DATASET_PROMPTS.get(data, [{}])
    print('[X] Generation prompts for validation:', prompts)
    dge = dge_name(GenerationContext(metadata_context = prompts[-1]), GenerationContext(metadata_context = prompts[-2])) if len(prompts) > 1 else None
    dge_path = DATASET_DGES.get(data, {}).get(dge, None)
    print('[X] Real DGE path:', dge_path)
    tokenizer_path = DATASET_TOKENIZERS.get(data, None)
    
    if 'seed' in kwargs:
        seed = kwargs['seed']
        print('[X] A random seed was provided:', seed)
    else:
        seed = None
        print('[X] A random seed can be provided using key --seed. Using no seed.')
        
    if 'checkpoint' in kwargs:
        checkpoint = kwargs['checkpoint']
        print('[X] A checkpoint to resume training from was provided:', checkpoint)
    else:
        checkpoint = None
        print('[X] A checkpoint to resume training can be provided using key --checkpoint. Training from scratch.')
        
    if 'metadata_names' in kwargs:
        metadata_names = kwargs['metadata_names']
        if isinstance(metadata_names, list) or isinstance(metadata_names, tuple):
            pass
        elif isinstance(metadata_names, str):
            if metadata_names in ['none', 'None', 'NONE']:
                metadata_names = []
            else:
                metadata_names = metadata_names.split(',')
        else:
            raise ValueError("Don't know how to parse given --metadata_names:", metadata_names)
        print('[X] Metadata names were provided:', metadata_names)
    else:
        metadata_names = DATASET_METADATA_NAMES.get(data, [])
        print('[X] Metadata names can be provided using --metadata_names and separated by commas. Using default names:', metadata_names)
    kwargs['metadata_names'] = metadata_names
        
    configs = read_yaml_config(kwargs, 'config', CONFIG_PATHS[model_name])
    configs['preprocess_config']['model_name'] = model_name
    configs['preprocess_config']['prompts'] = prompts
    configs['experiment_config'] = {'tokenizer_path' : tokenizer_path, 'model' : model_name, 'data' : data_path, 'seed' : seed, 'metadata_names' : metadata_names, 'checkpoint' : checkpoint, 'prompts': prompts, 'dge_path': dge_path}
    configs['trainer_config'] = {key[8:] : value for key, value in kwargs.items() if key.startswith('trainer_')}
    print('[X] Argument names for the lightning trainer have to start with trainer_. The following have been provided:', configs['trainer_config'])
    
    for config_name, config in configs.items():
        if config_name not in ['experiment_config', 'trainer_config']:
            configs[config_name] = dict_overwrite(config, kwargs, config_name)
    
    if model_name != 'diffusion':
        configs['preprocess_config']['round'] = configs['preprocess_config']['normalize'] is not None
        configs['preprocess_config']['log'] = False
    else:
        configs['preprocess_config']['round'] = False

    unused_args(kwargs, configs)
    print('------ Parsing done ------\n')
    return configs
"""Global variables and default values."""

### Datasets

GASPERINI_PATH = 'data/datasets/GasperiniShendure2019_atscale_tiny.h5ad'
GASPERINI_METADATA_NAMES = ['organism', 'tissue_type', 'celltype', 'cell_line', 'disease']
GASPERINI_PROMPTS = [{}]

IBD_TINY_PATH = 'data/datasets/cellarity_ibd_dataset_level_cleaned_tiny.h5ad'
IBD_PATH = 'data/datasets/cellarity_ibd_dataset_level_cleaned.h5ad'
IBD_NORM_PATH = 'data/datasets/cellarity_ibd_dataset_level_cleaned_norm.h5ad'

IBD_METADATA_NAMES = ['disease_colon', 'tissue_unified', 'sample']
IBD_PROMPTS = [{}, {'tissue_unified' : 'mucosa of ascending colon', 'disease_colon' : 'healthy'}, {'tissue_unified' : 'mucosa of ascending colon', 'disease_colon' : 'Crohn disease'}]
IBD_PROMPTS2 = [{}, {'tissue_unified' : 'ileum', 'disease_colon' : 'Crohn disease'}, {'tissue_unified' : 'colon', 'disease_colon' : 'ulcerative colitis'}]
IBD_PROMPTS3 = [{}, {'tissue_unified' : 'colon', 'disease_colon' : 'healthy'}, {'tissue_unified' : 'ileal mucosa', 'disease_colon' : 'colonic disorder'}]
IBD_PROMPTS4 = [{}, {'tissue_unified' : 'sigmoid colon', 'disease_colon' : 'healthy'}, {'tissue_unified' : 'sigmoid colon', 'disease_colon' : 'ulcerative colitis'}]
IBD_PROMPTS5 = [{}, {'tissue_unified' : 'colon', 'disease_colon' : 'healthy'}, {'tissue_unified' : 'colon', 'disease_colon' : 'ulcerative colitis'}]
IBD_PROMPTS6 = [{}, {'tissue_unified' : 'rectum', 'disease_colon' : 'healthy'}, {'tissue_unified' : 'rectum', 'disease_colon' : 'ulcerative colitis'}]



IBD_TOKENIZER_PATH = 'data/tokenizers/ibd_tokenizer.pkl'

TOYDATA1_PATH = 'data/datasets/toy_dataset1.h5ad'
TOYDATA2_PATH = 'data/datasets/toy_dataset2.h5ad'
TOYDATA_METADATA_NAMES = ['disease', 'organ']
TOYDATA_PROMPTS = [{}, {'disease' : 'healthy', 'organ' : 'heart'}, {'disease' : 'cancer', 'organ' : 'liver'}]

DATASET_PATHS = {'gasperini': GASPERINI_PATH,
                 'ibd_tiny': IBD_TINY_PATH, 
                 'ibd_norm': IBD_NORM_PATH,
                 'ibd': IBD_PATH, 
                 'toy1': TOYDATA1_PATH, 
                 'toy2': TOYDATA2_PATH}

DATASET_METADATA_NAMES = {'gasperini': GASPERINI_METADATA_NAMES, 
                          'ibd': IBD_METADATA_NAMES, 
                          'ibd_tiny': IBD_METADATA_NAMES, 
                          'ibd_norm': IBD_METADATA_NAMES,
                          'toy1': TOYDATA_METADATA_NAMES, 
                          'toy2': TOYDATA_METADATA_NAMES}

DATASET_PROMPTS = {'ibd' : IBD_PROMPTS,
                   'ibd_tiny' : IBD_PROMPTS,
                   'ibd_norm' : IBD_PROMPTS,
                   'toy1' : TOYDATA_PROMPTS,
                   'toy2' : TOYDATA_PROMPTS,
                   'gasperini' : GASPERINI_PROMPTS} 

DATASET_TOKENIZERS = {'ibd': IBD_TOKENIZER_PATH, 'ibd_norm' : IBD_TOKENIZER_PATH, 'ibd_tiny' : IBD_TOKENIZER_PATH}

DATASET_DGES = {'ibd_norm': 
                    {'colon_healthy_vs_colon_ulcerative_colitis' : 
                        'data/dges/ibd_pb_norm_dge_colon_healthy_vs_colon_ulcerative_colitis_2000hvg.csv',
                    'mucosa_of_ascending_colon_Crohn_disease_vs_mucosa_of_ascending_colon_healthy' :
                        'data/dges/ibd_pb_norm_dge_mucosa_of_ascending_colon_Crohn_disease_vs_mucosa_of_ascending_colon_healthy_2000hvg.csv'
                    },
                'ibd' :
                    {'colon_healthy_vs_colon_ulcerative_colitis' : 
                        'data/dges/ibd_pb_dge_colon_healthy_vs_colon_ulcerative_colitis_2000hvg.csv',
                    'mucosa_of_ascending_colon_Crohn_disease_vs_mucosa_of_ascending_colon_healthy' :
                        'data/dges/ibd_pb_dge_mucosa_of_ascending_colon_Crohn_disease_vs_mucosa_of_ascending_colon_healthy_2000hvg.csv'
                    }
                }

### Special tokens

END_TOKEN = '<END>'
UNKOWN_TOKEN = '<UNKOWN>'
MASK_TOKEN = '<MASK>'
START_TOKEN = '<START>'
PAD_TOKEN = '<PAD>'
GENE_START_TOKEN = '<GENE_START>'
GENE_END_TOKEN = '<GENE_END>'

SPECIAL_TOKENS = {'mask_token' : MASK_TOKEN, 'pad_token' : PAD_TOKEN, 'start_token' : START_TOKEN, 
                  'unknown_token' : UNKOWN_TOKEN, 'end_token' : END_TOKEN, 
                  'gene_start_token' : GENE_START_TOKEN, 'gene_end_token' : GENE_END_TOKEN}

SPECIAL_TOKENS_IDS = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
PAD_IDX = SPECIAL_TOKENS_IDS['pad_token']

### LLMs

BERT_MODEL_NAME = 'distilbert-base-uncased'
BERT_EMBEDDING_DIM = 768
BERT_GENE_EMBEDDINGS_PATH = 'data/embeddings/bert_gene_embeddings.pt'
BERT_SPECIAL_TOKENS_EMBEDDINGS_PATH = 'data/embeddings/bert_special_tokens_embeddings.pt'
BERT_IBD_METADATA_EMBEDDINGS_PATH = 'data/embeddings/bert_ibd_metadata_embeddings.pt'

### Models

IMPLEMENTED_MODELS = ['decoder', 'encoder', 'clip', 'encoder_decoder', 'diffusion']
DECODER_CONFIG_PATH = 'src/models/decoder/decoder_config.yaml'
ENCODER_CONFIG_PATH = 'src/models/encoder/encoder_config.yaml'
CLIP_CONFIG_PATH = 'src/models/clip/clip_config.yaml'
ENCODER_DECODER_CONFIG_PATH = 'src/models/encoder_decoder/encoder_decoder_config.yaml'
DIFFUSION_CONFIG_PATH = 'src/models/diffusion/diffusion_config.yaml'

CONFIG_PATHS = {'decoder': DECODER_CONFIG_PATH, 
                'encoder': ENCODER_CONFIG_PATH, 
                'clip': CLIP_CONFIG_PATH, 
                'encoder_decoder': ENCODER_DECODER_CONFIG_PATH, 
                'diffusion': DIFFUSION_CONFIG_PATH}


### Genes

ESM2_PATH = 'data/embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt'
ESM2_EMBEDDING_DIM = 5120
GENE_LIST_PATH = 'data/tokenizers/gene_list.txt'
GENE_TOKENIZER_PATH = 'data/tokenizers/gene_tokenizer.pkl'
NUM_GENES = 58676

import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # missing-task
            'mfmb_net': self.__MFMB_NET,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        
        if commonArgs['data_missing']:
            dataArgs = dataArgs['aligned_missing'] if (commonArgs['need_data_aligned'] and 'aligned_missing' in dataArgs) else dataArgs['unaligned_missing']
        else: 
            dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
            
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '/root/autodl-tmp/MFMB_Net/MMSA'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': None,
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'aligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    # 'missing_rate': (0.0, 0.0, 0.0),
                    'missing_seed': (111, 1111, 11111),
                },
                'unaligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': None,
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    # 'missing_rate': (0.1, 0.1, 0.1),
                    'missing_seed': (111, 1111, 11111),
                }
            },
            'sims':{
                'unaligned': {
                    #'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                },
                'unaligned_missing': {
                    #'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    'seq_lens': None,
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                    'missing_seed': (111, 1111, 11111),
                }
            },
            'mosei': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': None,
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'aligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    # 'missing_rate': (0.0, 0.0, 0.0),
                    'missing_seed': (111, 1111, 11111),
                },
                'unaligned_missing': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': None,
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    # 'missing_rate': (0.1, 0.1, 0.1),
                    'missing_seed': (111, 1111, 11111),
                }
            }
        }
        return tmp

    def __MFMB_NET(self):
        tmp = {
            'commonParas':{
                'data_missing': True,
                'deal_missing': True,
                'need_data_aligned': False,
                'alignmentModule': 'crossmodal_attn',
                'generatorModule': 'linear',
                'fusionModule': 'c_gate',
                'recloss_type': 'combine',
                'without_generator': False,

                'early_stop': 6,
                'use_bert': True,
                # use finetune for bert
                'use_bert_finetune': True,
                # use attention mask for Transformer
                'attn_mask': True, 
                'update_epochs': 4,
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # TODO ALIGNMENT Params.
                    # dropout
                    'text_dropout': 0.2, # textual Embedding Dropout
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 3,
                    # Transformer dropours
                    'attn_dropout': 0.2, # crossmodal attention block dropout
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.0,
                    'relu_dropout': 0.2,
                    'embed_dropout': 0.2,
                    'res_dropout': 0.2,
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (30, 6), 
                    # number of layers(Blocks) in the Crossmodal Networks
                    'nlevels': 3,
                    # TODO FUSION Params
                    'fusion_t_in': 90,
                    'fusion_a_in': 90,
                    'fusion_v_in': 90,
                    'fusion_t_hid': 36,
                    'fusion_a_hid': 20,
                    'fusion_v_hid': 48,
                    'fusion_gru_layers': 3, # USED FOR GRU / GATE FUSION.
                    'use_linear': True, # USED FOR GRU FUSION .
                    'fusion_drop': 0.2, #  USED FOR GRU / GATE FUSION.(before clf)
                    'cls_hidden_dim': 128,
                    'cls_dropout': 0.0,
                    
                    # gradient clip value (default: 0.8)
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8, 
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 24,
                    'learning_rate_bert': 1e-05,
                    'learning_rate_other': 0.002,
                    # when to decay learning rate (default: 20)
                    'patience': 5,
                    'weight_decay_bert': 0.0001,
                    'weight_decay_other': 0.001,
                    'weight_gen_loss': (5, 2, 20),
                    # 'weight_sim_loss': 5,
                },
                'sims':{
                    # TODO ALIGNMENT Params.
                    # dropout
                    'text_dropout': 0.2, # textual Embedding Dropout
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1, 
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 3,
                    # Transformer dropours
                    'attn_dropout': 0.2, # crossmodal attention block dropout
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.0,
                    'relu_dropout': 0.0,
                    'embed_dropout': 0.1,
                    'res_dropout': 0.2,
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (30, 6), 
                    # number of layers(Blocks) in the Crossmodal Networks
                    'nlevels': 2,
                    # TODO GENERATOR Params
                    'trans_hid_t': 40,
                    'trans_hid_t_drop': 0.0,
                    'trans_hid_a': 80,
                    'trans_hid_a_drop': 0.1,
                    'trans_hid_v': 48,
                    'trans_hid_v_drop': 0.3,
                    # 'generator_in': (40, 40, 40),
                    # TODO FUSION Params
                    'fusion_t_in': 90,
                    'fusion_a_in': 90,
                    'fusion_v_in': 90,
                    'fusion_t_hid': 36,
                    'fusion_a_hid': 20,
                    'fusion_v_hid': 48,
                    'fusion_gru_layers': 3, # USED FOR GRU / GATE FUSION.
                    'use_linear': True, # USED FOR GRU FUSION .
                    'fusion_drop': 0.2, #  USED FOR GRU / GATE FUSION.(before clf)
                    'cls_hidden_dim': 128,
                    'cls_dropout': 0.1,
                    
                    # gradient clip value (default: 0.8)
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8, 
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 16,
                    'learning_rate_bert': 1e-05,
                    'learning_rate_other': 0.002,
                    # when to decay learning rate (default: 20)
                    'patience': 10,
                    'weight_decay_bert': 0.0001,
                    'weight_decay_other': 0.001,
                    'weight_gen_loss': (1, 0.01, 0.0001),
                    # 'weight_sim_loss': 5,
                },
                'mosei': {
                    # TODO ALIGNMENT Params.
                    # dropout
                    'text_dropout': 0.2,  # textual Embedding Dropout
                    # temporal convolution kernel size
                    'conv1d_kernel_size_l': 1,
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 3,
                    # Transformer dropours
                    'attn_dropout': 0.2,  # crossmodal attention block dropout
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.0,
                    'relu_dropout': 0.2,
                    'embed_dropout': 0.2,
                    'res_dropout': 0.2,
                    #  transformers hidden unit size(d) &&  transformers hidden unit size(d)
                    'dst_feature_dim_nheads': (30, 6),
                    # number of layers(Blocks) in the Crossmodal Networks
                    'nlevels': 3,
                    # TODO FUSION Params
                    'fusion_t_in': 90,
                    'fusion_a_in': 90,
                    'fusion_v_in': 90,
                    'fusion_t_hid': 36,
                    'fusion_a_hid': 20,
                    'fusion_v_hid': 48,
                    'fusion_gru_layers': 3,  # USED FOR GRU / GATE FUSION.
                    'use_linear': True,  # USED FOR GRU FUSION .
                    'fusion_drop': 0.2,  # USED FOR GRU / GATE FUSION.(before clf)
                    'cls_hidden_dim': 128,
                    'cls_dropout': 0.0,

                    # gradient clip value (default: 0.8)
                    # when grad_clip == -1.0, means not use that
                    'grad_clip': 0.8,
                    # the batch_size of each epoch is updata_epochs * batch_size
                    'batch_size': 24,
                    'learning_rate_bert': 1e-05,
                    'learning_rate_other': 0.002,
                    # when to decay learning rate (default: 20)
                    'patience': 5,
                    'weight_decay_bert': 0.0001,
                    'weight_decay_other': 0.001,
                    'weight_gen_loss': (5, 2, 20),
                    # 'weight_sim_loss': 5,
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args

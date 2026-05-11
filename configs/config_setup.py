import datetime
from dassl.config import get_cfg_default


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print()
    print("************")
    print("** Config **")
    print("************")
    print(cfg)
    print()


def extend_cfg(cfg):
    cfg.OPTIM_PROMPT_LEARNER = cfg.OPTIM.clone()
    cfg.OPTIM_CLASSIFICATION_HEAD = cfg.OPTIM.clone()
    cfg.OPTIM_CLASSIFICATION_MODEL = cfg.OPTIM.clone()


def reset_cfg(cfg, args):
    cfg.SEED = args.seed

    cfg.GPU = args.gpu

    now = datetime.datetime.now()
    save_root = args.output_dir + "/" + str(now)
    cfg.OUTPUT_DIR = save_root

    cfg.BACKBONE = args.backbone

    cfg.EMBEDDING_DIM = args.embedding_dim

    cfg.PRETRAINED_CLIP_PATH = args.pretrained_clip_path

    cfg.PRETRAINED_RESNET50_PATH = args.pretrained_resnet50_path

    cfg.LOAD_RESNET50_PARAMETERS = args.load_resnet50_parameters

    cfg.LOAD_CLASSIFICATIONHEAD_PARAMETERS = args.load_classificationhead_parameters

    cfg.INPUT_SIZE = args.input_size

    cfg.N_CLS = args.n_cls

    cfg.STYLE_PROMPT_NUM = args.style_prompt_num

    cfg.N_CTX = args.n_ctx

    cfg.CTX_INIT = args.ctx_init

    cfg.CLASS_TOKEN_POSITION = args.class_token_position

    cfg.DATA_ROOT = args.data_root

    cfg.TRAIN_SET = args.train_set

    cfg.IMAGE_TRANSFORM = args.image_transform

    cfg.BATCH_SIZE = args.batch_size

    cfg.TRAIN_PROMPT_EPOCH = args.train_prompt_epoch

    cfg.TRAIN_MODEL_EPOCH = args.train_model_epoch

    cfg.PRINT_FREQ = args.print_freq

    cfg.CHECKPOINT_FREQ = args.checkpoint_freq

    cfg.DO_VAL = args.do_val

    cfg.SAVE_MODEL = args.save_model

    cfg.STYLE_PROMPT_SAVE_NAME = args.style_prompt_save_name

    cfg.STYLE_CLIP_SAVE_NAME = args.style_clip_save_name

    cfg.CLASSIFICATION_MODEL_SAVE_NAME = args.classification_model_save_name

    cfg.TEMPERATURE = args.temperature

    cfg.W_SD = args.w_sd

    cfg.W_ME = args.w_me

    cfg.W_TCE = args.w_tce

    cfg.W_ICE = args.w_ice

    cfg.W_CCE = args.w_cce

    cfg.W_FD = args.w_fd

    cfg.W_LD = args.w_ld


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 2. From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    print_args(args, cfg)

    return cfg

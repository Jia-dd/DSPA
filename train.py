import torch
import os
import random
import json
import time
import argparse
import torchvision
import warnings
import numpy as np
from clip import clip
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset
from dassl.utils import count_num_param
from model.style_clip import StyleCLIP
from model.modified_resnet50 import resnet50
from configs.config_setup import setup_cfg
from trainer.train_prompt import style_generation
from trainer.train_model import train_classification_model

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def load_clip_to_cpu(backbone_name, model_path):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, model_path)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def main(args):
    cfg = setup_cfg(args)

    print("****************************** Begin running ******************************")

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_seed(cfg.SEED)

    device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # transform
    if cfg.IMAGE_TRANSFORM:
        train_data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        val_data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    else:
        train_data_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
        ])
        val_data_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
        ])
    print("train_data_transform:")
    print(train_data_transform)
    print("val_data_transform:")
    print(val_data_transform)

    # number of workers
    numworker = min([os.cpu_count(), cfg.BATCH_SIZE if cfg.BATCH_SIZE > 1 else 0, 8])
    print('Using {} dataloader workers every process.'.format(numworker))

    # dataset
    image_path = os.path.join(cfg.DATA_ROOT, cfg.TRAIN_SET)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    print("image_path: {}".format(image_path))

    # Train data loading
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=train_data_transform)
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.BATCH_SIZE, shuffle=True,
                                               num_workers=numworker,
                                               persistent_workers=True)
    print("using {} images for training.".format(train_num))

    # Validate data loading
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=val_data_transform)
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.BATCH_SIZE, shuffle=False,
                                             num_workers=numworker,
                                             persistent_workers=True)
    print("using {} images for val.".format(val_num))

    # class of dataset
    class_list = train_dataset.class_to_idx
    class_dict = dict((val, key) for key, val in class_list.items())
    # write dict into json file
    json_str = json.dumps(class_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    classnames = [
        'baseball diamond',
        'basketball court',
        'bridge',
        'ground track field',
        'harbor',
        'plane',
        'ship',
        'small vehicle',
        'storage tank',
        'tennis court'
    ]

    print(f"Loading pretrained CLIP (backbone: {cfg.BACKBONE})")
    backbone_name = cfg.BACKBONE
    model_path = cfg.PRETRAINED_CLIP_PATH
    pretrained_clip = load_clip_to_cpu(backbone_name, model_path)

    print(f"Loading pretrained ResNet50")
    pretrained_resnet50 = torchvision.models.resnet50(pretrained=True)
    torch.save(pretrained_resnet50.state_dict(), cfg.PRETRAINED_RESNET50_PATH)

    print("Building style_clip model...")
    style_clip = StyleCLIP(cfg, classnames, pretrained_clip)
    style_clip = style_clip.float()

    print("Building classification model...")
    classification_model = resnet50(cfg.EMBEDDING_DIM, num_classes=cfg.N_CLS)

    print("Turning off gradients in both the image and the text encoder...")
    for name, param in style_clip.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name or "classification_head" in name:
            param.requires_grad_(True)

    print("# Total params: {:,} (prompt_learner: {:,}, classification_head: {:,})".format(
        count_num_param(style_clip.prompt_learner) + count_num_param(style_clip.classification_head),
        count_num_param(style_clip.prompt_learner), count_num_param(style_clip.classification_head)))

    # Double check
    enabled = set()
    for name, param in style_clip.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"parameters to be updated: {sorted(enabled)}\n")

    style_clip.to(device)
    classification_model.to(device)

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    start_time = time.time()
    print("********** Generating style prompts **********")

    style_generation(cfg, style_clip)

    print("\n********** Training classification model **********")

    if cfg.LOAD_RESNET50_PARAMETERS:
        print(f"Loading parameters of pretrained ResNet50... ")
        weights_path = cfg.PRETRAINED_RESNET50_PATH
        classification_model.load_state_dict(
            torch.load(weights_path, map_location=device),
            strict=False
        )

    if cfg.LOAD_CLASSIFICATIONHEAD_PARAMETERS:
        print(f"Loading parameters of ClassificationHead of style_clip... ")
        classification_model.fc_category_adjust.weight.data = style_clip.classification_head.fc.weight.data.clone()
        classification_model.fc_category_adjust.bias.data = style_clip.classification_head.fc.bias.data.clone()

    train_classification_model(cfg, device, style_clip, classification_model, train_loader, val_loader)

    print("\nFinish training.")
    end_time = time.time()

    # print training time
    total_time = int(end_time - start_time)
    total_min = total_time // 60
    s = total_time % 60
    h = total_min // 60
    m = total_min % 60
    print('\nTotal training time: {}h {}min {}s.'.format(h, m, s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="path to config file")

    parser.add_argument("--backbone", type=str, default="ViT-B/32", help="name of CNN backbone")
    parser.add_argument("--embedding_dim", type=int, default=512, help="embedding dim")
    parser.add_argument("--pretrained_clip_path", type=str, default="./pretrained_clip",
                        help="pretrained clip model path")

    parser.add_argument("--pretrained_resnet50_path", type=str, default="./pretrained_resnet50/resnet50.pt",
                        help="pretrained resnet50 path")
    parser.add_argument("--load_resnet50_parameters", type=bool, default=True,
                        help="load parameters of pretrained ResNet50 or not")

    parser.add_argument("--load_classificationhead_parameters", type=bool, default=True,
                        help="load parameters of ClassificationHead of style_clip or not")

    parser.add_argument("--input_size", type=int, default=224, help="input image size")
    parser.add_argument("--n_cls", type=int, help="number of class")

    parser.add_argument("--style_prompt_num", type=int, default=10, help="style num")
    parser.add_argument("--n_ctx", type=int, default=1, help="number of text context vectors")
    parser.add_argument("--ctx_init", type=str, default="A S style of a", help="initialization words")
    parser.add_argument("--class_token_position", type=str, default="end", help="middle or end or front")

    parser.add_argument("--data_root", type=str, help="data root")
    parser.add_argument("--train_set", type=str, help="train set")
    parser.add_argument("--image_transform", type=bool, default=False, help="default without image transform")

    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--train_prompt_epoch", type=int, help="max epoch of prompt training")
    parser.add_argument("--train_model_epoch", type=int, help="max epoch of model training")
    parser.add_argument("--print_freq", type=int, help="print frequency")
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint frequency")
    parser.add_argument("--do_val", type=bool, default=False, help="do val or not")
    parser.add_argument("--save_model", type=bool, default=True, help="save model or not")

    parser.add_argument("--style_prompt_save_name", type=str, default="style_prompt.pt",
                        help="style prompt save name")
    parser.add_argument("--style_clip_save_name", type=str, default="style_clip.pt",
                        help="style clip save name")
    parser.add_argument("--classification_model_save_name", type=str, default="classification_model.pt",
                        help="classification model save name")

    parser.add_argument("--temperature", type=float, default=3.0, help="distillation temperature")

    parser.add_argument("--w_sd", type=float, default=0.5, help="weight of L_StyleDiversity")
    parser.add_argument("--w_me", type=float, default=0.25, help="weight of L_MaxEntropy")
    parser.add_argument("--w_tce", type=float, default=0.25, help="weight of L_TextCrossEntropy")

    parser.add_argument("--w_ice", type=float, default=1, help="weight of L_ImageCrossEntropy")
    parser.add_argument("--w_cce", type=float, default=4, help="weight of L_ContrastCrossEntropy")
    parser.add_argument("--w_fd", type=float, default=1, help="weight of L_FeatureDistill")
    parser.add_argument("--w_ld", type=float, default=1, help="weight of L_LogitDistill")

    args = parser.parse_args()

    main(args)

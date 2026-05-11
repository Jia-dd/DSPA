import torch
import os
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dassl.utils import load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler


def train_classification_model(cfg, device, style_clip, classification_model, train_loader, val_loader):
    optim_classification_model = build_optimizer(classification_model, cfg.OPTIM_CLASSIFICATION_MODEL)
    sched_classification_model = build_lr_scheduler(optim_classification_model, cfg.OPTIM_CLASSIFICATION_MODEL)

    best_epoch = 0
    best_result = 0
    loss_list = []

    for epoch in range(cfg.TRAIN_MODEL_EPOCH):
        style_clip.eval()
        classification_model.train()

        len_train_loader = len(train_loader)
        num_batches = len_train_loader
        train_loader_iter = iter(train_loader)

        L_model_avg = 0.0
        for batch_idx in range(num_batches):
            batch = next(train_loader_iter)
            L_model = get_model_loss(cfg, device, style_clip, classification_model, batch)

            optim_classification_model.zero_grad()
            L_model.backward()
            optim_classification_model.step()
            if (batch_idx + 1) == num_batches:
                sched_classification_model.step()

            L_model_avg += L_model

            meet_print_freq = (batch_idx + 1) % cfg.PRINT_FREQ == 0
            if meet_print_freq:
                info = []
                info += [f"epoch:[{epoch + 1}/{cfg.TRAIN_MODEL_EPOCH}]"]
                info += [f"batch:[{batch_idx + 1}/{num_batches}]"]
                info += [f"loss_model:{L_model:.3f}"]
                info += [f"lr:{get_current_lr(optim_classification_model):.4e}"]
                print("\t".join(info))

        L_model_avg = L_model_avg / num_batches
        loss_list.append(L_model_avg.item())

        last_epoch = (epoch + 1) == cfg.TRAIN_MODEL_EPOCH
        do_val = cfg.DO_VAL
        meet_checkpoint_freq = ((epoch + 1) % cfg.CHECKPOINT_FREQ == 0 if cfg.CHECKPOINT_FREQ > 0 else False)

        if do_val:
            curr_result = val(device, classification_model, val_loader)
            is_best = curr_result > best_result
            if is_best:
                best_result = curr_result
                best_epoch = epoch
                if cfg.SAVE_MODEL:
                    save_name = cfg.CLASSIFICATION_MODEL_SAVE_NAME + "-best"
                    save_model(epoch, cfg.OUTPUT_DIR, classification_model, model_name=save_name)
            print('******* best acc: {:.1f}%, best epoch: {} *******'.format(best_result, best_epoch + 1))

        classification_model.train()

        if cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            info = []
            info += [f"checkpoint epoch [{epoch + 1}/{cfg.TRAIN_MODEL_EPOCH}]"]
            print("\t".join(info))
            save_name = cfg.CLASSIFICATION_MODEL_SAVE_NAME + "-epoch" + str(epoch + 1)
            save_model(epoch, cfg.OUTPUT_DIR, classification_model, model_name=save_name)

    plt.plot(np.arange(1, cfg.TRAIN_MODEL_EPOCH + 1), loss_list, label='L_model', marker='o')
    plt.title('L_model per epoch')
    plt.xlabel('epoch')
    plt.ylabel('L_model')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'L_model.png'))
    plt.close()


def get_model_loss(cfg, device, style_clip, classification_model, batch):
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    image_logits = classification_model(images)
    L_ImageCrossEntropy = F.cross_entropy(image_logits, labels)

    prompt_features_list = []
    for i in range(cfg.STYLE_PROMPT_NUM):
        prompt_i, _ = style_clip.prompt_learner(i)
        prompt_i = prompt_i.to(device)
        tokenized_prompts = style_clip.prompt_learner.tokenized_prompts.to(device)
        prompt_features_i = style_clip.text_encoder(prompt_i, tokenized_prompts)
        prompt_features_i_norm = prompt_features_i / prompt_features_i.norm(dim=-1, keepdim=True)
        prompt_features_list.append(prompt_features_i_norm)
    stacked_prompt_features = torch.stack(prompt_features_list, dim=0)
    center_prompt_features = torch.mean(stacked_prompt_features, dim=0)
    classification_features = classification_model.get_feature(images)
    classification_features_norm = classification_features / classification_features.norm(dim=-1, keepdim=True)
    contrast_logits = classification_features_norm @ center_prompt_features.T
    L_ContrastCrossEntropy = F.cross_entropy(contrast_logits, labels)

    clip_image_features = style_clip.image_encoder(images)
    L_FeatureDistill = feature_distillation_loss(classification_features, clip_image_features)

    L_LogitDistill = F.kl_div(F.log_softmax(image_logits / cfg.TEMPERATURE, dim=1),
                              F.softmax(contrast_logits / cfg.TEMPERATURE, dim=1),
                              reduction='batchmean') * cfg.TEMPERATURE * cfg.TEMPERATURE

    L_model = cfg.W_ICE * L_ImageCrossEntropy + cfg.W_CCE * L_ContrastCrossEntropy + cfg.W_FD * L_FeatureDistill + cfg.W_LD * L_LogitDistill

    return L_model


@torch.no_grad()
def val(device, classification_model, val_loader):
    classification_model.eval()

    class_correct = [0.] * 10
    class_total = [0.] * 10
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            outputs = classification_model(val_images)
            predict = torch.max(outputs, dim=1)[1]

            result = (predict == val_labels).squeeze()

            for i, label in enumerate(val_labels):
                class_correct[label] += result[i].item()
                class_total[label] += 1
                correct += result[i].item()
                total += 1

    accuracy = 100 * correct / total

    return accuracy


def get_current_lr(optim):
    return optim.param_groups[0]["lr"]


def feature_distillation_loss(student_feature, teacher_feature):
    return F.mse_loss(student_feature, teacher_feature)


def save_model(epoch, directory, model, optim=None, sched=None, is_best=False, model_name=""):
    model_dict = model.state_dict()

    optim_dict = None
    if optim is not None:
        optim_dict = optim.state_dict()

    sched_dict = None
    if sched is not None:
        sched_dict = sched.state_dict()

    save_checkpoint(
        {
            "state_dict": model_dict,
            "epoch": epoch + 1,
            "optimizer": optim_dict,
            "scheduler": sched_dict,
        },
        directory,
        is_best=is_best,
        model_name=model_name,
    )


def load_model(directory, model, model_file_name):
    if not directory:
        print("Note that load_model() is skipped as no pretrained model is given")
        return

    model_path = os.path.join(directory, model_file_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError('Model not found at "{}"'.format(model_path))

    checkpoint = load_checkpoint(model_path)
    state_dict = checkpoint["state_dict"]
    epoch = checkpoint["epoch"]

    # Ignore fixed token vectors
    if "token_prefix" in state_dict:
        del state_dict["token_prefix"]
    if "token_suffix" in state_dict:
        del state_dict["token_suffix"]

    print('Loading weights from "{}" (epoch = {})'.format(model_path, epoch))
    # set strict=False
    model.load_state_dict(state_dict, strict=False)

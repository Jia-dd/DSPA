import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dassl.utils import save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler


def style_generation(cfg, style_clip):
    print(f'Generating {cfg.STYLE_PROMPT_NUM} style...')

    optim_prompt_learner = build_optimizer(style_clip.prompt_learner, cfg.OPTIM_PROMPT_LEARNER)
    sched_prompt_learner = build_lr_scheduler(optim_prompt_learner, cfg.OPTIM_PROMPT_LEARNER)

    optim_classification_head = build_optimizer(style_clip.classification_head, cfg.OPTIM_CLASSIFICATION_HEAD)
    sched_classification_head = build_lr_scheduler(optim_classification_head, cfg.OPTIM_CLASSIFICATION_HEAD)

    loss_list = []

    for prompt_epoch in range(cfg.TRAIN_PROMPT_EPOCH):
        L_prompt = get_prompt_loss(cfg, style_clip)

        optim_prompt_learner.zero_grad()
        optim_classification_head.zero_grad()
        L_prompt.backward()
        optim_prompt_learner.step()
        optim_classification_head.step()
        sched_prompt_learner.step()
        sched_classification_head.step()

        loss_list.append(L_prompt.item())

        info = []
        info += [f"epoch:[{prompt_epoch + 1}/{cfg.TRAIN_PROMPT_EPOCH}]"]
        info += [f"loss_prompt:{L_prompt.item():.3f}"]
        info += [f"prompt_learner_lr:{get_current_lr(optim_prompt_learner):.4e}"]
        info += [f"classification_head_lr:{get_current_lr(optim_classification_head):.4e}"]
        print("\t".join(info))

    reset_learning_rate(optim_prompt_learner, cfg.OPTIM_PROMPT_LEARNER.LR)
    reset_learning_rate(optim_classification_head, cfg.OPTIM_CLASSIFICATION_HEAD.LR)

    sched_prompt_learner.last_epoch = 0
    sched_classification_head.last_epoch = 0

    style_prompt_path = os.path.join(cfg.OUTPUT_DIR, cfg.STYLE_PROMPT_SAVE_NAME)
    prompt_data = save_style_prompt(cfg, style_clip, style_prompt_path)
    save_model(cfg.TRAIN_PROMPT_EPOCH - 1, cfg.OUTPUT_DIR, style_clip, model_name=cfg.STYLE_CLIP_SAVE_NAME)

    plt.plot(np.arange(1, cfg.TRAIN_PROMPT_EPOCH + 1), loss_list, label='L_prompt', marker='o')
    plt.title('L_prompt per epoch')
    plt.xlabel('epoch')
    plt.ylabel('L_prompt')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'L_prompt.png'))
    plt.close()

    return prompt_data


def save_style_prompt(cfg, style_clip, save_path):
    print("Turning off gradients in prompt learner and classification head...")
    for name, param in style_clip.named_parameters():
        if "prompt_learner" in name or "classification_head" in name:
            param.requires_grad_(False)

    # all prompts with classnames
    prompts = style_clip.prompt_learner()

    prompts_list = []
    tokenized_prompts_list = []
    labels_list = []

    labels = torch.arange(cfg.N_CLS)

    for prompt in prompts:
        prompts_list.append(prompt)
        tokenized_prompts_list.append(style_clip.prompt_learner.tokenized_prompts)
        labels_list.append(labels)

    all_prompts = torch.cat(prompts_list, dim=0)
    all_tokenized_prompts = torch.cat(tokenized_prompts_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    prompt_data = {
        'prompts': all_prompts,
        'tokenized_prompts': all_tokenized_prompts,
        'labels': all_labels
    }

    torch.save(prompt_data, save_path)

    return prompt_data


def get_prompt_loss(cfg, style_clip):
    loss_StyleDiversity_sum = 0.0
    loss_MaxEntropy_sum = 0.0
    loss_TextCrossEntropy_sum = 0.0

    for i in range(cfg.STYLE_PROMPT_NUM):
        # i-th style prompt without classnames, i.e., P^style_i
        prompt_style_i = style_clip.prompt_learner(i, True)
        ctx_feature_i = style_clip.text_encoder(prompt_style_i, style_clip.prompt_learner.tokenized_ctx_init)
        ctx_feature_i_norm = ctx_feature_i / ctx_feature_i.norm(dim=-1, keepdim=True)

        if i == 0:
            loss_Style_i = 0.0
        else:
            loss_Style_i = 0.0
            for j in range(i):
                # j-th style prompt without classnames, i.e., P^style_j
                prompt_style_j = style_clip.prompt_learner(j, True)
                ctx_feature_j = style_clip.text_encoder(prompt_style_j, style_clip.prompt_learner.tokenized_ctx_init)
                ctx_feature_j_norm = ctx_feature_j / ctx_feature_j.norm(dim=-1, keepdim=True)
                feature_distance = ctx_feature_i_norm @ ctx_feature_j_norm.t()
                loss_Style_i += torch.abs(feature_distance)
            loss_Style_i = loss_Style_i / i
        loss_StyleDiversity_sum += loss_Style_i

        style_logit_i = style_clip.classification_head(ctx_feature_i_norm)
        style_prob_i = F.softmax(style_logit_i, dim=-1)
        loss_MaxEntropy_i = -torch.sum(style_prob_i * torch.log(style_prob_i + 1e-10), dim=-1)
        loss_MaxEntropy_sum += -loss_MaxEntropy_i

        prompt_i, classnames_embedding = style_clip.prompt_learner(i)
        tokenized_prompts = style_clip.prompt_learner.tokenized_prompts.to(prompt_i.device)
        prompt_features_i = style_clip.text_encoder(prompt_i, tokenized_prompts)
        prompt_features_i_norm = prompt_features_i / prompt_features_i.norm(dim=-1, keepdim=True)
        labels = torch.arange(cfg.N_CLS).to(prompt_i.device)
        prompt_logit_i = style_clip.classification_head(prompt_features_i_norm)
        loss_TextCrossEntropy_i = F.cross_entropy(prompt_logit_i, labels)
        loss_TextCrossEntropy_sum += loss_TextCrossEntropy_i

    L_StyleDiversity = loss_StyleDiversity_sum / cfg.STYLE_PROMPT_NUM
    L_MaxEntropy = loss_MaxEntropy_sum / cfg.STYLE_PROMPT_NUM
    L_TextCrossEntropy = loss_TextCrossEntropy_sum / cfg.STYLE_PROMPT_NUM

    L_prompt = cfg.W_SD * L_StyleDiversity + cfg.W_ME * L_MaxEntropy + cfg.W_TCE * L_TextCrossEntropy

    return L_prompt


def get_current_lr(optim):
    return optim.param_groups[0]["lr"]


def reset_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


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

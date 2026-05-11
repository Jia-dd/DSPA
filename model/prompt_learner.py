import torch
import torch.nn as nn
from clip import clip


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, pretrained_clip):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX
        ctx_init = cfg.CTX_INIT
        dtype = pretrained_clip.dtype
        ctx_dim = pretrained_clip.ln_final.weight.shape[0]
        clip_imsize = pretrained_clip.visual.input_resolution
        cfg_imsize = cfg.INPUT_SIZE
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        self.ctx_vectors = []
        for _ in range(cfg.STYLE_PROMPT_NUM):
            ctx_vector = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vector, std=0.02)
            ctx_vector = nn.Parameter(ctx_vector)
            self.ctx_vectors.append(ctx_vector)
        self.ctx_vectors = nn.ParameterList(self.ctx_vectors)

        ctx_init = ctx_init.replace("_", " ")

        print('Prompt design: Prompt-driven Style Generation')
        print(f'Initial context: "{ctx_init}"')
        print(f"Number of Prompt Styler context words (tokens): {n_ctx}")
        
        classnames = [name.replace("_", " ") for name in classnames]

        prompts = [ctx_init + " " + name + "." for name in classnames]

        tokenized_ctx_init = clip.tokenize(ctx_init)
        tokenized_classnames = torch.cat([clip.tokenize(classname) for classname in classnames])
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            ctx_init_embedding = pretrained_clip.token_embedding(tokenized_ctx_init).type(dtype)
            self.classnames_embedding = pretrained_clip.token_embedding(tokenized_classnames).type(dtype)
            prompts_embedding = pretrained_clip.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix_init", ctx_init_embedding[:, :2, :])
        self.register_buffer("token_suffix_init", ctx_init_embedding[:, 2 + n_ctx:, :])
        self.register_buffer("token_prefix", prompts_embedding[:, :2, :])
        self.register_buffer("token_suffix", prompts_embedding[:, 2 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.tokenized_ctx_init = tokenized_ctx_init
        self.tokenized_classnames = tokenized_classnames
        self.tokenized_prompts = tokenized_prompts

        self.classnames = classnames  
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None, dim=1):
        '''
        dim0 is either batch_size (during training) or n_cls (during testing)
        ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        '''
        if label is not None:
            prefix = prefix[label].unsqueeze(0)
            suffix = suffix[label].unsqueeze(0)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=dim,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=dim,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=dim,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError('The value of class_token_position is wrong!')

        return prompts

    def forward(self, index=None, style=False):
        ctx = self.ctx_vectors

        # without classnames
        if style:
            prefix = self.token_prefix_init
            suffix = self.token_suffix_init
        # with classnames
        else:
            prefix = self.token_prefix
            suffix = self.token_suffix

        # get all style prompts with classnames (N classes)
        if index == None:
            prompts = []
            for ctx_i in ctx:
                if ctx_i.dim() == 2:
                    ctx_i = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)
                prompt = self.construct_prompts(ctx_i, prefix, suffix)
                prompts.append(prompt)
            return prompts
        # get one style prompt
        else:
            # get one style prompt without classnames
            if style:
                prompt = self.construct_prompts(ctx[index].unsqueeze(0), prefix, suffix)
                return prompt
            # get one style prompt with classnames (N classes)
            else:
                prompt = self.construct_prompts(ctx[index].unsqueeze(0).expand(self.n_cls, -1, -1), prefix, suffix)
                return prompt, self.classnames_embedding

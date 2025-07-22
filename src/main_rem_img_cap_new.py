import os
import sys
import clip
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as nnf
from utils import noise_injection
from CaptionsDataset import collate
from torch.utils.data import DataLoader
from CaptionsDataset import CaptionsDataset
# from ClipCapMemoryLora import ClipCaptionModel, ClipCaptionPrefix
from ClipCapMemoryLora import ClipCaptionModel, ClipCaptionPrefix
from transformers import AdamW, get_linear_schedule_with_warmup

def count_trainable_parameters(model):
    """
    计算模型中所有需要梯度更新（可训练）的参数数量

    参数：
        model: torch.nn.Module 类型的模型

    返回：
        可训练参数总数（int）
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    args,                      # parameters used for training
    datasets: CaptionsDataset, # datasets used for training
    model: ClipCaptionModel,   # captioning model used for training
    warmup_steps: int = 5000,  # warming up steps used for traing
    output_dir: str = '.',     # output path of the wights
    output_prefix: str = ''    # file prefix name of saved weights
):
    device = args.device
    batch_size = args.bs
    epochs = args.epochs
    

    # if the path of outputs does not exist, create it according to the output_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loading model
    model = model.to(device)
    # model.load_state_dict(torch.load("checkpoints_img_fea_best_1098_norm_prefix_0/coco/coco_prefix-004.pt"))
    model.train()
    if not args.using_clip_features:
        encoder, _ = clip.load(args.clip_model, device = device)
        encoder.eval()

    # method of optimization
    optimizer = AdamW(model.parameters(), lr = args.lr)
    dataloader = DataLoader(datasets, batch_size = batch_size, shuffle = True, drop_last = True, num_workers=args.num_workers, collate_fn=collate)
    tokenizer = dataloader.dataset.tokenizer
    schedular = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = epochs * len(dataloader))
    scaler = torch.cuda.amp.GradScaler(enabled = args.use_amp)
    for epoch in range(epochs):
        # visualization
        sys.stdout.flush()
        print(f">>> Training epoch {epoch}")
        progress = tqdm(total = len(dataloader), desc = output_prefix)
        train_loss_sum = 0
        # training
        for idx, (captions_clip, captions_gpt_tokens, captions_tokens_for_loss, masks, hard_prompts_length, rt_feat) in enumerate(dataloader):
            model.zero_grad()
            if not args.using_clip_features:
                with torch.no_grad():
                    captions_clip_tokens = captions_clip.to(device)  # caption_clip -> tokens, (b, 77)
                    continuous_prefix = encoder.encode_text(captions_clip_tokens).float()  # (b, clip_hidden_size)
            else:
                continuous_prefix = captions_clip.to(device).float() # caption_clip -> embeddings, (b, clip_hidden_size)

            if args.normalize_prefix:
                continuous_prefix /= continuous_prefix.norm(2, dim = -1, keepdim = True)
            continuous_prefix = noise_injection(continuous_prefix, variance = args.noise_variance, device = args.device)
            captions_gpt_tokens, captions_tokens_for_loss, masks, rt_feat = captions_gpt_tokens.to(device), captions_tokens_for_loss.to(device), masks.to(device), rt_feat.to(device)

            with torch.cuda.amp.autocast(enabled = args.use_amp):                
                if args.using_hard_prompt:
                    outputs = model(continuous_prefix, captions_gpt_tokens, hard_prompts_length, masks, rt_feat)
                    logits = outputs.logits # (batch_size, max_length, vocab_size)
                else:
                    outputs = model(args, continuous_prefix, captions_gpt_tokens, mask = masks)
                    logits = outputs.logits # (batch_size, max_length, vocab_size)
            captions_tokens_for_loss = captions_tokens_for_loss.masked_fill(captions_tokens_for_loss == tokenizer.eos_token_id, 0)
            # [name for name, _ in model.named_parameters()]
            if (args.frozen1 == 1 and args.frozen2 == 1):
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(False)
                    
                    if name.startswith("gpt"):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)   
                                
                    if name.startswith("mapping_network.linear"):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)    

                    if name.startswith("mapping_network.prefix_const"):
                        parameter.requires_grad_(True)

                    if name.startswith("mapping_network.rt_linear"):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)

                    if args.use_moe_lora and name.startswith("mapping_network.crossatt.transformer.layers."):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)

                    if args.use_moe_lora and name.startswith("mapping_network.transformer.layers."):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)
            if (args.frozen1 == 1 and args.frozen2 == 0):
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(False)
                    
                    if name.startswith("gpt"):
                        parameter.requires_grad_(True)   
                                
                    if name.startswith("mapping_network.linear"):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)    

                    if name.startswith("mapping_network.prefix_const"):
                        parameter.requires_grad_(True)

                    if name.startswith("mapping_network.rt_linear"):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)

                    if args.use_moe_lora and name.startswith("mapping_network.crossatt.transformer.layers."):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)

                    if args.use_moe_lora and name.startswith("mapping_network.transformer.layers."):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)

            if (args.frozen1 == 0 and args.frozen2 == 1):
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(False)
                    
                    if name.startswith("gpt"):
                        train_key_word_list = ['lora_A', 'lora_B']
                        for train_key_word in train_key_word_list:
                            if train_key_word in name:
                                parameter.requires_grad_(True)   
                                
                    if name.startswith("mapping_network"):
                        parameter.requires_grad_(True)    

            # print("模型的可训练参数量为：", count_trainable_parameters(model))

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), captions_tokens_for_loss.flatten(), ignore_index = 0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedular.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            train_loss_sum += loss.item()
            log_iters = len(dataloader)//5 if len(dataloader) > 5 else len(dataloader)
            if (idx + 1) % (log_iters) == 0:
                print('epoch {}, iter {}, average train loss: {}'.format(epoch, idx, train_loss_sum / log_iters))
                train_loss_sum = 0
                torch.save(model.state_dict(), os.path.join(output_dir, f"latest.pt"))
                
        progress.close()
        ckpt_path = os.path.join(output_dir, f"00{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f'saving checkpoint to {ckpt_path}')

        # if (epoch+1) % args.save_every == 0 or epoch == epochs - 1:
        #     ckpt_path = os.path.join(output_dir, f"{output_prefix}-00{epoch}.pt")
        #     torch.save(model.state_dict(), ckpt_path)
        #     print(f'saving checkpoint to {ckpt_path}')

        # 2. 调用外部脚本评估
        #    构造命令参数列表，epoch 作为字符串传入

        import subprocess
        if args.prefix == "sydney":
            cmd_flickr = [
                "bash", "scripts/eval_sydney.sh",
                "train_sydney",
                str(0),
                "--entity_filtering --retrieved_info caption_sydney_image_sydney_9.json --K 5",
                "sydney",
                str(epoch),
                str(args.r),
                str(args.lora_alpha),
                str(args.frozen1),
                str(args.frozen2),
            ]
            print(f"Running evaluation script for epoch {epoch} …")
            _ = subprocess.run(cmd_flickr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if args.prefix == "ucm":
            cmd_flickr = [
                "bash", "scripts/eval_ucm.sh",
                "train_ucm",
                str(0),
                "--entity_filtering --retrieved_info caption_ucm_image_ucm_9.json --K 5",
                "ucm",
                str(epoch),
                str(args.r),
                str(args.lora_alpha),
                str(args.frozen1),
                str(args.frozen2),
            ]
            print(f"Running evaluation script for epoch {epoch} …")
            _ = subprocess.run(cmd_flickr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) 
            cmd_flickr = [
                "bash", "scripts/eval_msrvtt.sh",
                "train_ucm",
                str(0),
                "--entity_filtering --retrieved_info caption_msrvtt_image_msrvtt_5_woclass.json --K 9",
                "ucm",
                str(epoch),
                str(args.r),
                str(args.lora_alpha),
                str(args.frozen1),
                str(args.frozen2),
            ]
            print(f"Running evaluation script for epoch {epoch} …")
            _ = subprocess.run(cmd_flickr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) 
            cmd_flickr = [
                "bash", "scripts/eval_coco_new.sh",
                "train_ucm",
                str(0),
                "--entity_filtering --retrieved_info caption_coco_image_coco_9.json --K 5",
                "ucm",
                str(epoch),
                str(args.r),
                str(args.lora_alpha),
                str(args.frozen1),
                str(args.frozen2),
            ]
            print(f"Running evaluation script for epoch {epoch} …")
            _ = subprocess.run(cmd_flickr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)                        
        # cmd_coco = [
        #     "bash", "scripts/eval_coco_new.sh",
        #     "train_flickr30k",
        #     str(0),
        #     "--entity_filtering --retrieved_info caption_coco_image_coco_9.json --K 5",
        #     "coco",
        #     str(epoch),
        #     str(args.r),
        #     str(args.lora_alpha),
        #     str(args.frozen1),
        #     str(args.frozen2),
        # ]
        # print(f"Running evaluation script for epoch {epoch} …")
        
        # _ = subprocess.run(cmd_coco, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # # 3. 打印脚本输出
        # print(cocoinfo.stdout)

        # 4. 脚本执行结束后删除该 checkpoint
        # if os.path.isfile(ckpt_path):
        #     os.remove(ckpt_path)
        #     print(f"removed checkpoint {ckpt_path}")
        # else:
        #     print(f"warning: checkpoint {ckpt_path} not found, cannot remove")        
        

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type = int, default = 80, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 2e-5, help = 'learning rate for training')
    parser.add_argument('--device', default = 'cuda:1', help = 'gpu for training')
    parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs')
    parser.add_argument('--random_mask', action = 'store_true', default = True, help = 'entity masking strategy')
    parser.add_argument('--prob_of_random_mask', type = float, default = 0.4, help = 'masking rate')
    parser.add_argument('--clip_project_length', type = int, default = 10, help = 'clip projecting length')
    parser.add_argument('--continuous_prompt_length', type = int, default = 10, help = 'soft prompts length')
    parser.add_argument('--k', type=int, default=5, help='#retrieved captions')
    parser.add_argument('--max_num_of_entities', type = int, default = 10, help = 'maximum number of detected entities')
    parser.add_argument('--prompt_template_length', type = int, default = 5, help = 'maximum number of hard prompt entities')
    parser.add_argument('--num_layers', type = int, default = 8, help = 'number of layer in Transformer-based projector')
    parser.add_argument('--noise_variance', type = float, default = 0.016, help = 'noise variance')
    parser.add_argument('--clip_model', default = 'ViT-B/32', help = "'RN50', 'RN101', 'RN50x4', 'ViT-B/32'")
    parser.add_argument('--using_clip_features', action = 'store_true', default = True, help = 'whether to use the pre-extracted features')
    parser.add_argument('--is_rn', dest = 'is_rn', action = 'store_true', default = False, help = 'CLIP backbone: True -> ResNet, False -> ViT')
    parser.add_argument('--language_model', default = 'gpt2', help = 'gpt2, facebook/opt-350m')
    parser.add_argument('--using_hard_prompt', action = 'store_true', default = True, help = 'whether to entity-aware hard prompts')
    parser.add_argument('--soft_prompt_first', action = 'store_true', default = True, help = 'True -> soft prompt first, i.e., soft prompt + hard prompt')
    parser.add_argument('--only_hard_prompt', action = 'store_true', default = False, help = 'True -> do not use soft prompts in this case')
    parser.add_argument('--debug', action = 'store_true', default = False, help = 'debug = True means using a smaller dataloader')
    parser.add_argument('--few_shot_ratio', type = float, default = 1.0, help = 'measuring the low-data setting')
    parser.add_argument('--save_every', type = int, default = 1, help = 'save weights every n epochs')
    parser.add_argument('--prefix', default = 'coco_prefix', help = 'prefix name for saved weights')
    parser.add_argument('--path_of_datasets', default = f'/home/liu/.workspace/IFCap/annotations/ucm/ucm_with_entities_cocoid_imgfeature.pickle') # coco_with_entities
    parser.add_argument('--out_dir', default = './checkpoints', help = 'the path of output')
    parser.add_argument('--normalize_prefix', dest = 'normalize_prefix', type = int, default = True, help = 'normalizing prefix')
    parser.add_argument('--name_of_objects_vocabs', default = 'visual_genome_entities')
    parser.add_argument('--path_of_objects_vocabs', default = './annotations/vocabulary/all_objects_attributes_relationships.pickle')
    parser.add_argument('--frozen_gpt', action = 'store_true', default = False, help = 'freezing language models during training')
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--use_amp', action = 'store_true', default = True, help = "whether to use torch.amp to acclerate training")
    parser.add_argument('--disable_random_seed', action = 'store_true', default = False, help = 'set random seed for reproducing')
    parser.add_argument('--random_seed', type = int, default = 30, help = 'set random seed for reproducing')
    # parser.add_argument("--prefix", type=str, default="prefix prefix prefix:")
    parser.add_argument('--rt_path', default='./annotations/coco/coco_train_seed30_var0.04.json')
    parser.add_argument('--use_moe_lora', type=bool, default=True)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--frozen1', type=int, default=1)
    parser.add_argument('--frozen2', type=int, default=1)
    
    args = parser.parse_args()

    print(f'args: {vars(args)}')

    if not args.disable_random_seed:
        set_seed(args.random_seed)

    clip_hidden_size = 640 if args.is_rn else 512
    
    datasets = CaptionsDataset(
        language_model = args.language_model,
        max_num_of_entities = args.max_num_of_entities,
        using_clip_features = args.using_clip_features,
        path_of_datasets = args.path_of_datasets,
        debug = args.debug,
        args = args
    )
    if args.frozen_gpt:
        model = ClipCaptionPrefix(args, args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, args.num_layers, gpt_type = args.language_model, soft_prompt_first = args.soft_prompt_first, only_hard_prompt = args.only_hard_prompt)
    else:
        model = ClipCaptionModel(args, args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, args.num_layers, gpt_type = args.language_model, soft_prompt_first = args.soft_prompt_first, only_hard_prompt = args.only_hard_prompt, k=args.k)
    
    if args.frozen1 or args.frozen2:

        if "ucm" in args.rt_path:
        # ori_model = ClipCaptionModelOri(args, args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, args.num_layers, gpt_type = args.language_model, soft_prompt_first = args.soft_prompt_first, only_hard_prompt = args.only_hard_prompt)
            ori_model_state_dict = torch.load("/home/liu/.workspace/IFCap/checkpoints_r32_alpha64_0_0/train_msrvtt/002.pt")
        elif "sydney" in args.rt_path:
            ori_model_state_dict = torch.load("/home/liu/.workspace/IFCap/checkpoints_r32_alpha64_0_0/train_ucm/0018.pt")

        if args.frozen2:
            for i in range(12):
                ori_model_state_dict[f"'gpt.base_model.model.transformer.h.{i}.attn.c_attn.base_layer.weight'"] = ori_model_state_dict[f"gpt.transformer.h.{i}.attn.c_attn.weight"]
                ori_model_state_dict[f"'gpt.base_model.model.transformer.h.{i}.attn.c_attn.base_layer.bias'"] = ori_model_state_dict[f"gpt.transformer.h.{i}.attn.c_attn.bias"]
                ori_model_state_dict[f"'gpt.base_model.model.transformer.h.{i}.attn.c_proj.base_layer.weight'"] = ori_model_state_dict[f"gpt.transformer.h.{i}.attn.c_proj.weight"]
                ori_model_state_dict[f"'gpt.base_model.model.transformer.h.{i}.attn.c_proj.base_layer.bias'"] = ori_model_state_dict[f"gpt.transformer.h.{i}.attn.c_proj.bias"]
                ori_model_state_dict[f"'gpt.base_model.model.transformer.h.{i}.mlp.c_proj.base_layer.weight'"] = ori_model_state_dict[f"gpt.transformer.h.{i}.mlp.c_proj.weight"]
                ori_model_state_dict[f"'gpt.base_model.model.transformer.h.{i}.mlp.c_proj.base_layer.bias'"] = ori_model_state_dict[f"gpt.transformer.h.{i}.mlp.c_proj.bias"]
                ori_model_state_dict[f"'gpt.base_model.model.transformer.h.{i}.mlp.c_fc.base_layer.weight'"] = ori_model_state_dict[f"gpt.transformer.h.{i}.mlp.c_fc.weight"]
                ori_model_state_dict[f"'gpt.base_model.model.transformer.h.{i}.mlp.c_fc.base_layer.bias'"] = ori_model_state_dict[f"gpt.transformer.h.{i}.mlp.c_fc.bias"]
        
        if args.frozen1 and args.use_moe_lora:    
            for i in range(8):
                ori_model_state_dict[f"mapping_network.transformer.layers.{i}.attn.to_queries.base_layer_.weight"] = ori_model_state_dict[f"mapping_network.transformer.layers.{i}.attn.to_queries.weight"]
                ori_model_state_dict.pop(f"mapping_network.transformer.layers.{i}.attn.to_queries.weight")
                ori_model_state_dict[f"mapping_network.transformer.layers.{i}.attn.to_keys_values.base_layer_.weight"] = ori_model_state_dict[f"mapping_network.transformer.layers.{i}.attn.to_keys_values.weight"]
                ori_model_state_dict.pop(f"mapping_network.transformer.layers.{i}.attn.to_keys_values.weight")    

            
            ori_model_state_dict["mapping_network.crossatt.transformer.layers.0.attn.to_queries.base_layer_.weight"] = ori_model_state_dict["mapping_network.crossatt.transformer.layers.0.attn.to_queries.weight"]
            ori_model_state_dict.pop("mapping_network.crossatt.transformer.layers.0.attn.to_queries.weight")
            ori_model_state_dict["mapping_network.crossatt.transformer.layers.0.attn.to_keys_values.base_layer_.weight"] = ori_model_state_dict["mapping_network.crossatt.transformer.layers.0.attn.to_keys_values.weight"]
            ori_model_state_dict.pop("mapping_network.crossatt.transformer.layers.0.attn.to_keys_values.weight")    
            ori_model_state_dict["mapping_network.linear.base_layer_.weight"] = ori_model_state_dict["mapping_network.linear.weight"]
            ori_model_state_dict.pop("mapping_network.linear.weight")    
            ori_model_state_dict["mapping_network.rt_linear.base_layer_.weight"] = ori_model_state_dict["mapping_network.rt_linear.weight"]
            ori_model_state_dict.pop("mapping_network.rt_linear.weight")    

        # list(model.named_parameters())
        # list(ori_model.named_parameters())
        # 拷贝原始权重（主干）
        # ori_model.load_state_dict( )
        # 加载新模型（空的）
        new_state_dict = model.state_dict()

        matched_state_dict = {k: v for k, v in ori_model_state_dict.items() if k in new_state_dict and v.shape == new_state_dict[k].shape}

        # 加载这些匹配的参数
        new_state_dict.update(matched_state_dict)
        model.load_state_dict(new_state_dict)
    args.out_dir = f"{args.out_dir.split('/')[0]}_r{args.r}_alpha{args.lora_alpha}_{str(args.frozen1)}_{str(args.frozen2)}/{args.out_dir.split('/')[1]}"
    train(args, datasets, model, output_dir = args.out_dir, output_prefix = args.prefix)

if __name__ == '__main__':
    main()

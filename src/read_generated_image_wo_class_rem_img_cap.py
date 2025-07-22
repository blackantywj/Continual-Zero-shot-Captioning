import os
import pickle
import json
import argparse
import random
import clip
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from load_annotations import load_captions
import random
import re
from collections import OrderedDict

def sort_dict_by_group_and_suffix(data: dict, group_order: list[str]) -> OrderedDict:
    """
    :param data: 原字典，键形如 "<prefix>_<number>.png"
    :param group_order: 前缀的优先顺序列表，例如 ["-4wsuPCjDBc_5_15", "-foo_1"]
    :return: 按指定组顺序及数字后缀排序后的 OrderedDict
    """
    # 1. 初始化各组容器
    groups = {grp: [] for grp in group_order}
    others = []  # 不在 group_order 中的前缀

    # 2. 按前缀分组
    for key in data:
        # 前缀 = 最后一个 "_" 之前的所有内容
        prefix = key.rsplit("_", 1)[0]    # 
        if prefix in groups:
            groups[prefix].append(key)
        else:
            others.append(key)

    # 3. 组内按数字后缀排序的辅助函数
    def suffix_num(k):
        # 从 "_<number>.png" 提取数字部分并转 int
        return int(re.search(r"_(\d+)\.png$", k).group(1))  # 

    # 4. 按 group_order 拼接排序后键
    ordered = OrderedDict()
    for grp in group_order:
        # 稳定排序：sorted() 保证相同后缀时保持原 relative order :contentReference[oaicite:1]{index=1}
        for k in sorted(groups[grp], key=suffix_num):
            ordered[k] = data[k]
    # 可选：把不在 group_order 的键放到末尾，也可忽略
    for k in sorted(others, key=suffix_num):
        ordered[k] = data[k]

    return ordered

with open("/home/liu/.workspace/IFCap/annotations/ucm/ucm_train_RN50x64.pickle", 'rb') as f:
    total_data = pickle.load(f)

# with open("/home/cumt/workspace/IFCap/annotations/sydney/train_captions.json", 'r') as file:
#     data3 = json.load(file)
    
# group_order = [k for k, v in data3.items()]
# total_data = sort_dict_by_group_and_suffix(total_data, group_order)
total_data = [v.to("cuda:0") for k, v in sorted(total_data.items(), key=lambda x: (int(x[0].split("_")[0]), int(x[0].split('_')[1][0])))]
# total_data = [v.to("cuda:0") for k, v in total_data.items()]


def get_captions_path(domain):
    
    datasets = {
            'coco' : './annotations/coco/train_captions.json',
            'flickr30k' : './annotations/flickr30k/train_captions_sorted.json',
            'nocaps' : './annotations/nocaps/nocaps_corpus.json',
            'msvd': "/home/cumt/workspace/IFCap/annotations/msvd/train_captions.json",
            'msrvtt': "/home/cumt/workspace/IFCap/annotations/msrvtt/train_captions.json",
            'sydney': "/home/liu/.workspace/IFCap/annotations/sydney/train_captions.json",
            'ucm': "/home/liu/.workspace/IFCap/annotations/ucm/train_captions.json",
            }
    
    return datasets[domain]

def get_image_path(domain):
    
    datasets = {
            'coco' : '/usr/data/data/coco/val2014/',
            'flickr30k' : './annotations/flickr30k/flickr30k-images/',
            'nocaps' : './annotations/nocaps/images/',
            "msvd" : "/home/cumt/workspace/IFCap/msvd/testing",
            "msrvtt" : "/home/cumt/workspace/IFCap/frames_cv",
            'sydney': "/home/cumt/workspace/IFCap/Sydney-captions/images_png",
            'ucm': "/home/cumt/workspace/IFCap/UCM_captions/imgs_png",
            }
    
    return datasets[domain]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_caption_features(clip_feature_path, train_captions, args):
    if os.path.exists(clip_feature_path):
        with open(clip_feature_path, 'rb') as f:
            caption_features = pickle.load(f)
    else:
        features = []
        batch_size = 256
        with torch.no_grad():
            for i in tqdm(range(0, len(train_captions), batch_size)):
                batch_captions = train_captions[i: i + batch_size]
                clip_captions = tokenizer(batch_captions).to(args.device)
                clip_features = clip_model.encode_text(clip_captions)
                features.append(clip_features)

            caption_features = torch.cat(features).to('cpu')
        with open(clip_feature_path, 'wb') as f:
            pickle.dump(caption_features, f)

    return caption_features

def image_like_retrieval_train(train_captions, output_path, caption_features, args):

    retrieved_captions = {}
    
    # noise_features = noise_injection(caption_features, caption_features[5]
    #                                  variance=args.variance,
    #                                  device=args.device).to(torch.float16)

    for i in tqdm(range(len(total_data))):
        image_feature = total_data[i].unsqueeze(0)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        similarity = image_feature @ caption_features.T
        similarity[0][i] = 0
        niber = []
        for _ in range(args.K):
            _, max_id = torch.max(similarity, dim=1)
            niber.append(max_id.item())
            similarity[0][max_id.item()] = 0
        retrieved_captions[train_captions[i]] = [train_captions[k] for k in niber]

    # with open(output_path, 'wb') as f:
    #     pickle.dump(retrieved_captions, f)

    with open(output_path, 'w') as f:
        json.dump(retrieved_captions, f, indent=4)

def retrieve_caption_test(image_path, annotations, train_captions, output_path, caption_features, args):

    bs = 1
    image_ids = list(annotations.keys())
    image_features = []
    with torch.no_grad():
        for idx in tqdm(range(0, len(image_ids), bs)):
            image_input = [preprocess(Image.open(os.path.join(image_path, i)))
                           for i in image_ids[idx:idx + bs]]
            image_features.append(clip_model.encode_image(torch.tensor(np.stack(image_input)).to(args.device)))
        image_features = torch.concat(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        retrieved_captions = {}

        for i in tqdm(range(image_features.shape[0])):
            image_feature = image_features[i].unsqueeze(0).to(args.device)
            similarity = image_feature @ caption_features.T
            niber = []
            for _ in range(args.L):
                _, max_id = torch.max(similarity, dim=1)
                niber.append(max_id.item())
                similarity[0][max_id.item()] = 0

            retrieved_captions[image_ids[i]] = [train_captions[k] for k in niber]

        with open(output_path, 'w') as f:
            json.dump(retrieved_captions, f, indent=4)
          
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 30, help = 'Random seed')
    parser.add_argument('--L', type = int, default = 5, help = 'The number of retrieved captions for Image Like Retrieval')
    parser.add_argument('--K', type = int, default = 5, help = 'The number of retrieved captions for Entity Filtering')
    parser.add_argument('--variance', type = float, default = 0.04, help = 'Variance for noise injection')
    parser.add_argument('--domain_test', default = 'ucm', help = 'Name of test dataset', choices=['coco', 'flickr30k', 'nocaps', "msvd", "msrvtt", "sydney", 'ucm'])
    parser.add_argument('--domain_source', default = 'ucm', help = 'Name of source dataset', choices=['coco', 'flickr30k', "msvd", "msrvtt", "sydney", "ucm"])
    parser.add_argument('--device', default = 'cuda:0', help = 'Cuda device')
    parser.add_argument('--variant', default = 'RN50x64', help = 'CLIP variant')
    parser.add_argument('--test_only', action = 'store_true', help = 'No ILR')

    args = parser.parse_args()

    global clip_model, preprocess, tokenizer

    set_seed(args.seed)
    
    clip_model, preprocess = clip.load(args.variant, device=args.device)
    # model_name = 'ViT-L/14' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
    # clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = clip.tokenize

    # ckpt = torch.load(f"checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-{model_name}.pt", map_location=args.device)
    # message = clip_model.load_state_dict(ckpt)
    
    # print(message)

    clip_model = clip_model.cuda().eval()

    captions_path = get_captions_path(args.domain_source)
    datasets = f'{args.domain_source}_captions'
    clip_feature_path = f'./annotations/{args.domain_source}/text_feature_clip{args.variant}.pickle'
    print('clip_feature_path', clip_feature_path)
    
    train_captions = load_captions(datasets, captions_path)
    caption_features = load_caption_features(clip_feature_path, train_captions, args).to(args.device)
    caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
    
    train_output_path = f'./annotations/{args.domain_test}/{args.domain_test}_train_woclass.json'
    print('train_output_path', train_output_path)
    if not args.test_only and not os.path.exists(train_output_path):
        print('Perform image-like retrieval')
        image_like_retrieval_train(train_captions, train_output_path, caption_features, args)

    if args.domain_test in ["coco", "flickr30k"]:
        image_path = get_image_path(args.domain_test)
        with open(f"./annotations/{args.domain_test}/test_captions.json", 'r') as f:
            annotations = json.load(f)
        # test_center_point(image_path, annotations, classname_caption, test_output_path, caption_features, args)
        test_output_path = f'./annotations/retrieved_sentences/test_caption_{args.domain_source}_image_{args.domain_test}_{args.L}_woclass.json'
        print('test_output_path', test_output_path)    
        if not os.path.exists(test_output_path):
            print('Perform image-to-text retrieval')
            retrieve_caption_test(image_path, annotations, train_captions, test_output_path, caption_features, args)
    else:
        image_path = get_image_path(args.domain_test)
        with open(f"./annotations/{args.domain_test}/test_captions.json", 'r') as f:
            annotations = json.load(f)
        # test_center_point(image_path, annotations, classname_caption, test_output_path, caption_features, args)
        test_output_path = f'./annotations/retrieved_sentences/caption_{args.domain_source}_image_{args.domain_test}_{args.L}_woclass.json'
        print('test_output_path', test_output_path)    
        if not os.path.exists(test_output_path):
            print('Perform image-to-text retrieval')
            retrieve_caption_test(image_path, annotations, train_captions, test_output_path, caption_features, args)        
    
    
if __name__=='__main__':
    main()

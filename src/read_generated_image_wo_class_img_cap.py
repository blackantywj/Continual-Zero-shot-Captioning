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
from utils import noise_injection

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

with open("/home/cumt/workspace/IFCap/annotations/ucm/ucm_train_RN50x64.pickle", 'rb') as f:
    total_data = pickle.load(f)

# with open("/home/cumt/workspace/IFCap/annotations/sydney/train_captions.json", 'r') as file:
#     data3 = json.load(file)
    
# group_order = [k for k, v in data3.items()]
# total_data = sort_dict_by_group_and_suffix(total_data, group_order)
total_data = [v.to("cuda:0") for k, v in sorted(total_data.items(), key=lambda x: (int(x[0].split("_")[0]), int(x[0].split('_')[1][0])))]
# total_data = [v.to("cuda:0") for k, v in total_data.items()]

# global variable
clip_model, preprocess = None, None

def get_captions_path(domain):
    
    datasets = {
            'coco' : './annotations/coco/train_captions.json',
            'flickr30k' : './annotations/flickr30k/train_captions_sorted.json',
            'nocaps' : './annotations/nocaps/nocaps_corpus.json',
            'msvd': "/home/cumt/workspace/IFCap/annotations/msvd/train_captions.json",
            'msrvtt': "/home/cumt/workspace/IFCap/annotations/msrvtt/train_captions.json",
            'sydney': "/home/cumt/workspace/IFCap/annotations/sydney/train_captions.json",
            'ucm': "/home/cumt/workspace/IFCap/annotations/ucm/train_captions.json",
            }
    
    return datasets[domain]

def get_image_path(domain):
    
    datasets = {
            'coco' : '/usr/data/data/coco/val2014/',
            'flickr30k' : '/usr/data/flickr30k/flickr30k-images/',
            'nocaps' : './annotations/nocaps/images/',
            "msvd" : "/home/cumt/workspace/IFCap/msvd/validation",
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
                clip_captions = clip.tokenize(batch_captions, truncate=True).to(args.device)
                clip_features = clip_model.encode_text(clip_captions)
                features.append(clip_features)

            caption_features = torch.cat(features).to('cpu')
        with open(clip_feature_path, 'wb') as f:
            pickle.dump(caption_features, f)

    return caption_features

def image_like_retrieval_train(train_captions, output_path, caption_features, args):

    retrieved_captions = {}
    
    noise_features = noise_injection(caption_features,
                                     variance=args.variance,
                                     device=args.device).to(torch.float16)

    for i in tqdm(range(len(noise_features))):
        image_feature = noise_features[i].unsqueeze(0)
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

def retrieve_caption_test_vid(image_path, annotations, train_captions, output_path, caption_features, args):

    bs = 256
    num_images_per_folder = 5
    image_ids = list(annotations.keys())
    image_features = []
    # with torch.no_grad():
        # for idx in tqdm(range(0, len(image_ids), bs)):
        #     image_input = [preprocess(Image.open(os.path.join(image_path, i)))
        #                    for i in image_ids[idx:idx + bs]]
        #     image_features.append(clip_model.encode_image(torch.tensor(np.stack(image_input)).to(args.device)))
        
        # image_features = torch.concat(image_features)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # cosine_sim_max = 0
        # belogclass = None
        # for classname, caption_features in caption_features_class.items():  
        #     cosine_sim_set = torch.mm(image_features, caption_features.t()).mean()
        #     if cosine_sim_set > cosine_sim_max:
        #         belogclass = classname
    with torch.no_grad():
        for folder in tqdm(image_ids, desc="Processing folders"):
            folder_dir = os.path.join(image_path, folder)
            if not os.path.isdir(folder_dir):
                print(f"Warning: {folder_dir} is not a directory. Skipping.")
                continue

            # 找到所有图片文件
            image_files = [f for f in os.listdir(folder_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(image_files) < num_images_per_folder:
                print(f"Warning: {folder} contains fewer than {num_images_per_folder} images. Skipping.")
                continue

            # 随机选5张
            selected_files = random.sample(image_files, num_images_per_folder)

            # 加载并预处理
            images = []
            for img_file in selected_files:
                img_path = os.path.join(folder_dir, img_file)
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = preprocess(image)
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue


            # 批量编码
            image_input = torch.stack(images).to("cuda:0")  # shape: [5, 3, 224, 224]
            features = clip_model.encode_image(image_input)  # [5, 512] 假设输出512维
            features = features / features.norm(dim=-1, keepdim=True)
            # mean_feature = features.mean(dim=0).unsqueeze(0)  # [512]

            # 加到总列表里
            image_features.append(features.cpu())
        image_features = torch.concat(image_features)
        
        retrieved_captions = {}

        for i in tqdm(range(image_features.shape[0])):
            image_feature = image_features[i].unsqueeze(0).to(args.device)
            similarity = image_feature @ caption_features.T
            niber = []
            for _ in range(args.L):
                _, max_id = torch.max(similarity, dim=1)
                niber.append(max_id.item())
                similarity[0][max_id.item()] = 0
            if image_ids[int(i/5)] in retrieved_captions:
                retrieved_captions[image_ids[int(i/5)]].extend([train_captions[k] for k in niber])
            else:
                retrieved_captions[image_ids[int(i/5)]] = [train_captions[k] for k in niber]
        with open(output_path, 'w') as f:
            json.dump(retrieved_captions, f, indent=4)

def retrieve_caption_test(image_path, annotations, train_captions, output_path, caption_features, args):

    bs = 1
    image_ids = list(annotations.keys())
    image_features = []
    with torch.no_grad():
        for idx in tqdm(range(0, len(image_ids), bs)):
            image_input = [preprocess(Image.open(os.path.join(image_path, i.split('.')[0] + '.jpg')))
                           for i in image_ids[idx:idx + bs]]
            image_features.append(clip_model.encode_image(torch.tensor(np.stack(image_input)).to(args.device)))
        image_features = torch.concat(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # cosine_sim_max = 0
        # belogclass = None
        # for classname, caption_features in caption_features_class.items():  
        #     cosine_sim_set = torch.mm(image_features, caption_features.t()).mean()
        #     if cosine_sim_set > cosine_sim_max:
        #         belogclass = classname

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
    parser.add_argument('--domain_test', default = 'msvd', help = 'Name of test dataset', choices=['coco', 'flickr30k', 'nocaps', "msvd", "msrvtt", "sydney", 'ucm'])
    parser.add_argument('--domain_source', default = 'msvd', help = 'Name of source dataset', choices=['coco', 'flickr30k', "msvd", "msrvtt", "sydney", "ucm"])
    parser.add_argument('--device', default = 'cuda:0', help = 'Cuda device')
    parser.add_argument('--variant', default = 'RN50x64', help = 'CLIP variant')
    parser.add_argument('--test_only', action = 'store_true', help = 'No ILR')

    args = parser.parse_args()

    global clip_model, preprocess

    set_seed(args.seed)
    clip_model, preprocess = clip.load(args.variant, device=args.device)

    captions_path = get_captions_path(args.domain_source)
    datasets = f'{args.domain_source}_captions'
    clip_feature_path = f'./annotations/{args.domain_source}/text_feature_clip{args.variant}_sorted.pickle'
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
        with open(f"./annotations/{args.domain_test}/val_captions.json", 'r') as f:
            annotations = json.load(f)
        # test_center_point(image_path, annotations, classname_caption, test_output_path, caption_features, args)
        test_output_path = f'./annotations/retrieved_sentences/val_caption_{args.domain_source}_image_{args.domain_test}_{args.L}_woclass.json'
        print('test_output_path', test_output_path)    
        if not os.path.exists(test_output_path):
            print('Perform image-to-text retrieval')
            retrieve_caption_test(image_path, annotations, train_captions, test_output_path, caption_features, args)
    elif args.domain_test in ['msvd', 'msrvtt']:
        image_path = get_image_path(args.domain_test)
        with open(f"./annotations/{args.domain_test}/validation_captions.json", 'r') as f:
            annotations = json.load(f)
        # test_center_point(image_path, annotations, classname_caption, test_output_path, caption_features, args)
        test_output_path = f'./annotations/retrieved_sentences/validation_caption_{args.domain_source}_image_{args.domain_test}_{args.L}_woclass.json'
        print('test_output_path', test_output_path)    
        if not os.path.exists(test_output_path):
            print('Perform image-to-text retrieval')
            retrieve_caption_test_vid(image_path, annotations, train_captions, test_output_path, caption_features, args)
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

import os
import json
import clip
import torch
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
from typing import List
from ClipCap import ClipCaptionModel
from transformers import AutoTokenizer
from utils import compose_discrete_prompts, normal, log_normal
from load_annotations import load_entities_text
from search import greedy_search, beam_search, opt_search
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories
from utils import noise_injection
import math

with open("tasktoavgemb.pickle", 'rb') as f:
    tasktoavgemb = pickle.load(f)
    avgemblist = [v.unsqueeze(0) for _, v in tasktoavgemb.items()]
    avgemb = torch.cat(avgemblist, dim = 0).to("cuda:0")

def validation_nocaps(
    args,
    inpath: str,                             # path of annotations file
    entities_text: List[str],                # entities texts of vocabulary
    texts_embeddings: torch.Tensor,          # entities embeddings of vocabulary
    model: ClipCaptionModel,                 # trained language model
    tokenizer: AutoTokenizer,                # tokenizer 
    preprocess: clip = None,                 # processor of the image
    encoder: clip = None,                    # clip backbone
) -> None:
    
    device = args.device
    if args.using_image_features:
        with open(inpath, 'rb') as infile:
            annotations = pickle.load(infile) # [[image_path, image_split, image_features, [caption1, captions2, ...]], ...]
    else:
        with open(inpath, 'r') as infile:
            annotations = json.load(infile) # [{'split': 'near_domain', 'image_id': '4499.jpg', 'caption': [caption1, caption2, ...]}, ...]
    if args.entity_filtering:
        with open(f'annotations/retrieved_entity/{args.retrieved_info}', 'r') as f:
            retrieved_entities = json.load(f)
    with open(f'annotations/retrieved_sentences/{args.retrieved_info}', 'r') as f:
        eval_rt = json.load(f)

    test_rt_id = [k for k, v in eval_rt.items()]
    test_rt_caps = [i for k, v in eval_rt.items() for i in v[:args.k]]

    ## captions to clip features
    bs = 1000
    rt_feats = []
    for idx in range(0, len(test_rt_caps), bs):
        caps = test_rt_caps[idx:idx + bs]
        with torch.no_grad():
            rt_feat_batch = encoder.encode_text(clip.tokenize(caps).to(device))
            rt_feats.append(rt_feat_batch)

    ## connect image id with clip features
    rt_feats = torch.concat(rt_feats)
    eval_rt_dic = {}
    for id, idx in zip(test_rt_id, range(0, len(rt_feats), args.k)):
        eval_rt_dic[id] = rt_feats[idx:idx + args.k]


    indomain = []
    neardomain = []
    outdomain = []
    overall = []
    for idx, annotation in tqdm(enumerate(annotations), total=len(annotations)):
        if args.using_image_features:
            image_id, split, image_features, captions = annotation
            image_features = image_features.float().unsqueeze(dim = 0).to(device)
        else:
            image_id = annotation['image_id']
            split = annotation['split']
            captions = annotation['caption']
            image_path = args.image_folder + split + '/' + image_id
            image = preprocess(Image.open(image_path)).unsqueeze(dim = 0).to(device)
            image_features = encoder.encode_image(image).float()

        image_features /= image_features.norm(2, dim = -1, keepdim = True)

        rt_feats = eval_rt_dic[image_id]
        rt_feats = rt_feats.to(torch.float32)
        continuous_embeddings = model.mapping_network(image_features, rt_feats.unsqueeze(0)).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
        if args.using_hard_prompt:
            key = image_id
            if not args.entity_filtering:
                logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
                detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold) # List[List[]], [[category1, category2, ...], [], ...]
                detected_objects = detected_objects[0] # infering single image -> List[category1, category2, ...]
            else:
                if args.adaptive_ef != "":
                    if args.adaptive_ef == 'log_normal':
                        detected_objects = log_normal(retrieved_entities[key], args.K)
                    if args.adaptive_ef == 'normal':
                        detected_objects = normal(retrieved_entities[key], args.K)
                else:
                    K = args.K
                    detected_objects = list(filter(lambda x: x[0] >= K, retrieved_entities[key]))
                    detected_objects = [l[1] for l in detected_objects]

            discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)
            
            discrete_embeddings = model.word_embed(discrete_tokens)
            if args.only_hard_prompt:
                embeddings = discrete_embeddings
            elif args.soft_prompt_first:
                embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
            else:
                embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
        else:      
            embeddings = continuous_embeddings
        
        if 'gpt' in args.language_model:
            if not args.using_greedy_search:
                sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt) # List[str]
                sentence = sentence[0] # selected top 1
            else:
                sentence = greedy_search(embeddings = embeddings, tokenizer = tokenizer, model = model.gpt)
        else:
            sentence = opt_search(prompts=args.text_prompt, embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt)
            sentence=sentence[0]

        predict = {}
        predict['detected_objects'] = detected_objects
        predict["split"] = split
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        overall.append(predict)
        if split == 'in_domain':
            indomain.append(predict)
        elif split == 'near_domain':
            neardomain.append(predict)
        elif split == 'out_domain':
            outdomain.append(predict)

    with open(os.path.join(args.out_path, f'overall_generated_captions.json'), 'w') as outfile:
        json.dump(overall, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'indomain_generated_captions.json'), 'w') as outfile:
        json.dump(indomain, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'neardomain_generated_captions.json'), 'w') as outfile:
        json.dump(neardomain, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'outdomain_generated_captions.json'), 'w') as outfile:
        json.dump(outdomain, outfile, indent = 4)

def validation_coco_flickr30k(
    args,
    inpath: str,                             # path of annotations file
    entities_text: List[str],                # entities texts of vocabulary
    texts_embeddings: torch.Tensor,          # entities embeddings of vocabulary
    model: ClipCaptionModel,                 # trained language model
    tokenizer: AutoTokenizer,                # tokenizer 
    preprocess: clip = None,                 # processor of the image
    encoder: clip = None,                    # clip backbone
) -> None:

    device = args.device
    if args.using_image_features:
        with open(inpath, 'rb') as infile:
            annotations = pickle.load(infile) # [[image_path, image_features, [caption1, caption2, ...]], ...]
    else:
        with open(inpath, 'r') as infile:
            annotations = json.load(infile)   # {image_path: [caption1, caption2, ...]}
    if args.entity_filtering:
        with open(f'annotations/retrieved_entity/{args.retrieved_info}', 'r') as f:
            retrieved_entities = json.load(f)
    with open(f'annotations/retrieved_sentences/{args.retrieved_info}', 'r') as f:
        eval_rt = json.load(f)

    test_rt_id = [l[0] for l in annotations]
    test_rt_caps = [i for _, caps in eval_rt.items() for i in caps[:args.k]]

    ## captions to clip features
    bs = 1000
    rt_feats = []
    for idx in range(0, len(test_rt_caps), bs):
        caps = test_rt_caps[idx:idx+bs]
        with torch.no_grad():
            rt_feat_batch = encoder.encode_text(clip.tokenize(caps).to(device))
            rt_feats.append(rt_feat_batch)

    ## connect image id with clip features
    rt_feats = torch.concat(rt_feats)
    eval_rt_dic = {}
    for image_id, idx in zip(test_rt_id, range(0, len(rt_feats), args.k)):
        eval_rt_dic[image_id] = rt_feats[idx:idx+args.k]

    similarity_list = []
    predicts = []
    for idx, item in tqdm(enumerate(annotations), total=len(annotations)):
        if args.using_image_features:
            image_id, image_features, captions = item
            image_features = image_features.float().unsqueeze(dim = 0).to(device) # (1, clip_hidden_size)
            image_features /= image_features.norm(2, dim=-1, keepdim=True)
        else:
            image_id = item
            captions = annotations[item]
            image_path = args.image_folder + image_id
            image = preprocess(Image.open(image_path)).unsqueeze(dim = 0).to(device)
            image_features = encoder.encode_image(image).float()
            image_features /= image_features.norm(2, dim=-1, keepdim=True)

        cossimfunc = torch.nn.CosineSimilarity()
        cossim = cossimfunc(avgemb, image_features)
        idx = int(torch.argmax(cossim))
        if idx != 0:
            continue
        # print(image_id)
        rt_feats = eval_rt_dic[image_id]
        rt_feats = rt_feats.to(torch.float32)
        continuous_embeddings = model.mapping_network(image_features, rt_feats.unsqueeze(0)).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
        if args.using_hard_prompt:
            key = image_id
            if not args.entity_filtering:
                logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
                detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold) # List[List[]], [[category1, category2, ...], [], ...]
                detected_objects = detected_objects[0] # infering single image -> List[category1, category2, ...]
            else:
                if args.adaptive_ef != "":
                    if args.adaptive_ef == 'log_normal':
                        detected_objects = log_normal(retrieved_entities[key], args.K)
                    if args.adaptive_ef == 'normal':
                        detected_objects = normal(retrieved_entities[key], args.K)
                else:
                    detected_objects = list(filter(lambda x: x[0] >= args.K, retrieved_entities[key]))
                    detected_objects = [l[1] for l in detected_objects]

            discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)

            discrete_embeddings = model.word_embed(discrete_tokens)

            if args.only_hard_prompt:
                embeddings = discrete_embeddings
            elif args.soft_prompt_first:
                embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
            else:
                embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
        else:
            embeddings = continuous_embeddings


        if 'gpt' in args.language_model:
            if not args.using_greedy_search:
                sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt) # List[str]
                sentence = sentence[0] # selected top 1
            else:
                sentence = greedy_search(embeddings = embeddings, tokenizer = tokenizer, model = model.gpt)
        else:
            sentence = opt_search(prompts=args.text_prompt, embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt)
            sentence = sentence[0]
        text_input = clip.tokenize([sentence]).to(device)
        # compute_clip_score
        with torch.no_grad():
            # 获取图片和文本的特征向量（注意 CLIP 模型默认归一化了特征向量）
            # print(sentence)
            text_feature = encoder.encode_text(text_input)
            
            # 归一化（如果模型未归一化的话，这一步可以保证使用余弦相似度）
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = (image_features @ text_feature.to(torch.float32).T).item()
        
        predict = {}
        predict['detected_objects'] = detected_objects
        predict["split"] = 'valid'
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)
        predict['similarity'] = similarity
        similarity_list.append(similarity)
    print("CLIP SCORE:", sum(similarity_list)/len(similarity_list))
    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent = 4)

# def calculate_clip_scores(dict_list, device=None):
#     """
#     对于每个输入字典，计算图片与预测文本的 CLIP-score（余弦相似度）。
    
#     参数:
#       dict_list: 包含字典的列表，每个字典包含 'image_name' 和 'prediction' 字段
#       device: 设备，默认为 "cuda"（若可用）否则 "cpu"
    
#     返回:
#       scores: 每个输入字典对应的 CLIP-score 列表
#     """
#     # 设置设备
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # 加载模型和预处理方法
#     model, preprocess = clip.load("ViT-B/32", device=device)
    
#     scores = []
#     for entry in dict_list:
#         image_path = entry["image_name"]
#         prediction = entry["prediction"]
        
#         # 加载并预处理图片
#         image = Image.open(image_path).convert("RGB")
#         image_input = preprocess(image).unsqueeze(0).to(device)
        
#         # 将预测文本 tokenize
#         text_input = clip.tokenize([prediction]).to(device)
        
#         with torch.no_grad():
#             # 获取图片和文本的特征向量（注意 CLIP 模型默认归一化了特征向量）
#             image_feature = model.encode_image(image_input)
#             text_feature = model.encode_text(text_input)
            
#             # 归一化（如果模型未归一化的话，这一步可以保证使用余弦相似度）
#             image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
#             text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            
#             # 计算余弦相似度
#             similarity = (image_feature @ text_feature.T).item()
        
#         scores.append(similarity)
    
#     return scores

@torch.no_grad()
def main(args) -> None:
    # initializing
    device = args.device
    clip_name = args.clip_model.replace('/', '') 
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    # loading categories vocabulary for objects
    if args.name_of_entities_text == 'visual_genome_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/all_objects_attributes_relationships.pickle', not args.disable_all_entities)
        if args.prompt_ensemble: # loading ensemble embeddings
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/visual_genome_embedding_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/visual_genome_embedding_{clip_name}.pickle')
    elif args.name_of_entities_text == 'coco_entities':
        entities_text = load_entities_text(args.name_of_entities_text, 'annotations/vocabulary/coco_categories.json', not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, f'annotations/vocabulary/coco_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'../annotations/vocabulary/coco_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'open_image_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/oidv7-class-descriptions-boxable.csv', not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/open_image_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/open_image_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'vinvl_vg_entities':
        entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json', not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vg_embeddings_{clip_name}_with_ensemble.pickle')
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vg_embeddings_{clip_name}.pickle')
    elif args.name_of_entities_text == 'vinvl_vgoi_entities':
        entities_text = load_entities_text(args.name_of_entities_text, 'annotations/vocabulary/vgcocooiobjects_v1_class2ind.json', not args.disable_all_entities)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, f'annotations/vocabulary/vgoi_embeddings_{clip_name}_with_ensemble.pickle', device=args.device, clip_type=args.clip_model)
        else:
            texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vgoi_embeddings_{clip_name}.pickle')
    else:
        print('The entities text should be input correctly!')
        return
    
    # loading model
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(args, args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, gpt_type = args.language_model, k=args.k)
    model.load_state_dict(torch.load(args.weight_path, map_location = device), strict=False)
    model.to(device)
    encoder, preprocess = clip.load(args.clip_model, device=device)
    if not args.using_image_features:
        inpath = args.path_of_val_datasets
    else:
        inpath = args.path_of_val_datasets[:-5] + f'_{clip_name}.pickle' # file with image features
    if args.name_of_datasets == 'nocaps': # nocaps
        if args.using_image_features:
            validation_nocaps(args, inpath, entities_text, texts_embeddings, model, tokenizer, preprocess, encoder)
        else:
            validation_nocaps(args, inpath, entities_text, texts_embeddings, model, tokenizer, preprocess, encoder)
    else: # coco, flickr30k
        if args.using_image_features:
            validation_coco_flickr30k(args, inpath, entities_text, texts_embeddings, model, tokenizer, preprocess, encoder)
        else:
            validation_coco_flickr30k(args, inpath, entities_text, texts_embeddings, model, tokenizer, preprocess, encoder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = 'cuda')
    parser.add_argument('--clip_model', default = 'ViT-B/32')
    parser.add_argument('--language_model', default = 'gpt2')
    parser.add_argument('--continuous_prompt_length', type = int, default = 10)
    parser.add_argument('--clip_project_length', type = int, default = 10)
    parser.add_argument('--temperature', type = float, default = 0.01)
    parser.add_argument('--top_k', type = int, default = 3)
    parser.add_argument('--threshold', type = float, default = 0.2) # 0.4
    parser.add_argument('--using_image_features', action = 'store_true', default = True, help = 'using pre-extracted image features')
    parser.add_argument('--name_of_datasets', default = 'nocaps', choices = ('coco', 'flickr30k', 'nocaps')) # coco
    parser.add_argument('--path_of_val_datasets', default = 'annotations/nocaps/nocaps_corpus.json') # ../annotations/coco/test_captions.json
    parser.add_argument('--disable_all_entities', action = 'store_true', default = False, help = 'whether to use entities with a single word only')
    parser.add_argument('--name_of_entities_text', default = 'vinvl_vgoi_entities', choices = ('visual_genome_entities', 'coco_entities', 'open_image_entities', 'vinvl_vg_entities', 'vinvl_vgoi_entities')) # coco_entities
    parser.add_argument('--prompt_ensemble', action = 'store_true', default = True)
    parser.add_argument('--weight_path', default = 'checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_folder', default = '/data/dataset/coco_2014/val2014/')
    parser.add_argument('--out_path', default = '.')
    parser.add_argument('--using_hard_prompt', action = 'store_true', default = True)
    parser.add_argument('--soft_prompt_first', action = 'store_true', default = True)
    parser.add_argument('--only_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--using_greedy_search', action = 'store_true', default = False, help = 'greedy search or beam search')
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--text_prompt', type = str, default = None)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--entity_filtering', action = 'store_true', default=False)
    parser.add_argument('--K', type = int, default=0)
    parser.add_argument('--retrieved_info', type = str, default='caption_coco_image_coco_9.json')
    parser.add_argument('--adaptive_ef', type = str, default="", choices=['', 'normal', 'log_normal'])

    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))

    main(args)

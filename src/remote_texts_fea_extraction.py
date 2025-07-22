import os
import clip
import pickle
import torch
from tqdm import tqdm
import open_clip

@torch.no_grad()
def main(device: str, clip_type: str, inpath: str, outpath: str):

    device = device
    encoder, _, preprocess = open_clip.create_model_and_transforms(clip_type)
    tokenizer = open_clip.get_tokenizer(clip_type)

    ckpt = torch.load(f"checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-{clip_type}.pt", map_location="cpu")
    message = encoder.load_state_dict(ckpt)
    print(message)

    encoder = encoder.cuda().eval()

    with open(inpath, 'rb') as infile:
        captions_with_entities = pickle.load(infile) # [[[entity1, entity2, ...], caption], ...]

    for idx in tqdm(range(len(captions_with_entities))):
        caption = captions_with_entities[idx][1]
        tokens = tokenizer(caption).to(device)
        embeddings = encoder.encode_text(tokens).squeeze(dim = 0).to('cpu')
        captions_with_entities[idx].append(embeddings)
    
    with open(outpath, 'wb') as outfile:
        pickle.dump(captions_with_entities, outfile)
    
    return captions_with_entities

if __name__ == '__main__':
    
    idx = 1 # change here! 0 -> coco training data, 1 -> flickr30k training data
    device = 'cuda:0'
    clip_type = 'ViT-B/32' # change here for different clip backbone (ViT-B/32, RN50x4)
    clip_name = clip_type.replace('/', '-')

    inpath = [
    "/home/cumt/workspace/IFCap/annotations/sydney/sydney_with_entities.pickle",
    "/home/cumt/workspace/IFCap/annotations/ucm/ucm_with_entities.pickle"
    ]
    outpath = [
    f'./annotations/sydney/sydney_texts_features_{clip_name}.pickle',
    f'./annotations/ucm/ucm_texts_features_{clip_name}.pickle',
    ]

    if os.path.exists(outpath[idx]):
        with open(outpath[idx], 'rb') as infile:
            captions_with_features = pickle.load(infile)
    else:
        captions_with_features = main(device, clip_name, inpath[idx], outpath[idx])

    import random
    print(f'datasets for {inpath[idx]}')
    print(f'The length of datasets: {len(captions_with_features)}')
    idx = random.randint(0, len(captions_with_features) - 1)
    caption_with_features = captions_with_features[idx]
    detected_entities, caption, caption_features = caption_with_features
    print(detected_entities, caption, caption_features.size(), caption_features.dtype)

    encoder, _ = clip.load(clip_type, device)
    with torch.no_grad():
        embeddings = encoder.encode_text(clip.tokenize(caption, truncate = True).to(device)).squeeze(dim = 0).to('cpu')
    print(abs(embeddings - caption_features).mean())
    print(idx)
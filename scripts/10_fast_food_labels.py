#!/usr/bin/env python3
"""Fast image-level labeling for an existing LAION NMF consensus.
Commands: prepare -> score (optional, actual JPGs needed) -> finalize.
Never infer image order from Moondream CSV image_index.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

EXT={'.jpg','.jpeg','.png','.webp','.bmp'}

def key(s):
    if pd.isna(s): return ''
    v=str(s).strip().replace('\\','/').split('/')[-1]
    pp=Path(v)
    return (pp.stem if pp.suffix.lower() in EXT else v).casefold()

def save(df,path):
    path=Path(path).expanduser();path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False);print(f'Saved {path} ({len(df)} rows)')

def cmd_prepare(a):
    order=pd.read_csv(a.order)
    if 'image_key' not in order: raise SystemExit('image_order.csv missing image_key')
    order['image_key']=order.image_key.map(key)
    if order.image_key.duplicated().any(): raise SystemExit('duplicate image keys in image_order')
    labels=pd.read_csv(a.labels)
    if 'image_key' not in labels:
        if 'image_name' not in labels:raise SystemExit('Need image_key or image_name in labels')
        labels['image_key']=labels.image_name.map(key)
    labels.image_key=labels.image_key.map(key)
    if labels.image_key.duplicated().any(): raise SystemExit('Duplicate keys in labels; review before joining')
    keep=[c for c in ['image_key','image_name','label','food_present','food_prominence','review_status'] if c in labels]
    df=order[['image_key']].merge(labels[keep],on='image_key',how='left',validate='one_to_one')
    if 'food_present' not in df:df['food_present']=np.nan
    if 'review_status' not in df:df['review_status']=''
    df['manual_food_present']=''
    df['clip_food_probability']=np.nan
    df['image_path']=''
    df['label_source']=np.where(df.food_present.isin([0,1]),'moondream_keyword','missing')
    print('Existing food labels:',int(df.food_present.isin([0,1]).sum()),'/',len(df))
    print('Unlabeled:',int(df.food_present.isna().sum()))
    save(df,a.output)

def discover(roots):
    found={};duplicates=set()
    for root in roots:
        root=Path(root).expanduser()
        if not root.is_dir(): raise SystemExit(f'Image directory not found: {root}')
        for folder, dirs, names in os.walk(root):
            dirs[:]=[x for x in dirs if x not in {'.git','.ipynb_checkpoints','node_modules','__pycache__'}]
            for filename in names:
                if Path(filename).suffix.lower() not in EXT:continue
                k=key(filename);p=str(Path(folder)/filename)
                if k in found and found[k]!=p:duplicates.add(k)
                else:found[k]=p
    for k in duplicates:found.pop(k,None)
    print(f'Found {len(found)} unambiguous image names; {len(duplicates)} duplicate names skipped')
    return found

def cmd_locate(a):
    df=pd.read_csv(a.manifest,keep_default_na=False)
    paths=discover(a.images)
    df['image_path']=df.image_key.map(paths).fillna('')
    print('Located',int((df.image_path!='').sum()),'of',len(df),'stimulus images')
    save(df,a.output)

def cmd_score(a):
    try:
        import torch,open_clip
        from PIL import Image
    except ImportError as e:raise SystemExit('Install first: python -m pip install open_clip_torch pillow torch') from e
    df=pd.read_csv(a.manifest,keep_default_na=False)
    if not {'image_path','clip_food_probability'} <= set(df):raise SystemExit('Run prepare and locate first')
    torch.set_num_threads(max(1,a.threads))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('CLIP device:',device,'(first run may download model weights)')
    model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained='laion2b_s34b_b79k',device=device)
    model.eval();tokenizer=open_clip.get_tokenizer('ViT-B-32')
    prompts=['a photo containing edible food, a meal, or food ingredients',
             'a photo of a plate of food or a snack',
             'a photo where something to eat is visible',
             'a photo of objects without anything edible',
             'a photo of an environment with no food',
             'a photo that does not contain a meal or food']
    with torch.no_grad():
        txt=model.encode_text(tokenizer(prompts).to(device));txt/=txt.norm(dim=-1,keepdim=True)
    todo=[]
    for i,row in df.iterrows():
        if (not a.all_images) and str(row.get('food_present','')).strip() in ('0','1','0.0','1.0'):continue
        if str(row.get('clip_food_probability','')).strip() not in ('','nan','None'):continue
        p=str(row.image_path)
        if p and Path(p).is_file():todo.append((i,p))
    print('Images to score:',len(todo))
    for start in range(0,len(todo),a.batch_size):
        chunk=todo[start:start+a.batch_size];items=[];idx=[]
        for i,p in chunk:
            try:
                with Image.open(p) as image:items.append(preprocess(image.convert('RGB')));idx.append(i)
            except Exception as e:print('Image failed',p,repr(e))
        if items:
            with torch.no_grad():
                im=model.encode_image(torch.stack(items).to(device));im/=im.norm(dim=-1,keepdim=True)
                sims=(100.0*im@txt.T).float()
                food=torch.logsumexp(sims[:,:3],dim=1)-np.log(3)
                other=torch.logsumexp(sims[:,3:],dim=1)-np.log(3)
                prob=torch.sigmoid(food-other).cpu().numpy()
            for i,v in zip(idx,prob):df.loc[i,'clip_food_probability']=float(v)
        if start%max(a.batch_size*5,1)==0: print('Scored',min(start+len(chunk),len(todo)),'/',len(todo))
    save(df,a.output)

def cmd_finalize(a):
    df=pd.read_csv(a.manifest,keep_default_na=False)
    if not {'image_key','food_present','manual_food_present','clip_food_probability'}<=set(df):
        raise SystemExit('Manifest missing expected fields')
    existing=pd.to_numeric(df.food_present,errors='coerce')
    manual=pd.to_numeric(df.manual_food_present,errors='coerce')
    prob=pd.to_numeric(df.clip_food_probability,errors='coerce')
    for arr,name in [(existing,'food_present'),(manual,'manual_food_present')]:
        bad=arr.notna()&(~arr.isin([0,1]))
        if bad.any():raise SystemExit(f'Invalid 0/1 value in {name}')
    selected=existing.copy()
    source=pd.Series(np.where(existing.notna(),'moondream_keyword','unlabeled'),index=df.index)
    high=prob>=a.food_threshold;low=prob<=a.nonfood_threshold
    if a.use_clip:
        new=existing.isna()&(high|low)
        selected.loc[new]=high.loc[new].astype(int)
        source.loc[new]='clip_provisional'
    selected.loc[manual.notna()]=manual.loc[manual.notna()]
    source.loc[manual.notna()]='manual_review'
    out=df.copy();out['food_present']=selected
    out['label_source']=source
    # Human-approved labels trump old guesses; no automatic relabeling of reviewed entries.
    out['review_status']=np.where(manual.notna(),'reviewed',out.review_status)
    save(out,a.output)
    counts=out.groupby('label_source').size()
    print('Label sources:\n'+counts.to_string())
    print('Total usable:',int(selected.notna().sum()),'food:',int((selected==1).sum()),'nonfood:',int((selected==0).sum()))
    to_review=out[(source!='manual_review')&(selected.eq(1)|((prob>a.nonfood_threshold)&(prob<a.food_threshold)))].copy()
    if 'clip_food_probability' in to_review:to_review=to_review.sort_values('clip_food_probability',ascending=False)
    save(to_review,Path(a.output).with_name('PRIORITY_MANUAL_REVIEW.csv'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--order',required=True);a.add_argument('--labels',required=True);a.add_argument('--output',required=True);a.set_defaults(fn=cmd_prepare)
    a=sub.add_parser('locate');a.add_argument('--manifest',required=True);a.add_argument('--images',nargs='+',required=True,help='Folders containing original stimulus JPG/PNG files');a.add_argument('--output',required=True);a.set_defaults(fn=cmd_locate)
    a=sub.add_parser('score');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--all-images',action='store_true');a.add_argument('--batch-size',type=int,default=16);a.add_argument('--threads',type=int,default=4);a.set_defaults(fn=cmd_score)
    a=sub.add_parser('finalize');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--use-clip',action='store_true',help='Exploratory inclusion of high-confidence automated labels; NOT ground truth');a.add_argument('--food-threshold',type=float,default=0.85);a.add_argument('--nonfood-threshold',type=float,default=0.15);a.set_defaults(fn=cmd_finalize)
    args=p.parse_args();args.fn(args)
if __name__=='__main__':main()

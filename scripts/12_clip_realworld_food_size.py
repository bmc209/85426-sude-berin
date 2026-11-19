#!/usr/bin/env python3
"""Identify visible food item with CLIP and propose its typical REAL-WORLD size.

This is NOT visual prominence or object segmentation. CLIP selects among a curated
vocabulary of food items; a transparent cm-to-ordinal mapping gives a 1-10 score.
Scores are ALWAYS provisional until a human reviews the original photograph.

Input: fast_labels/04_final_food_labels.csv with image_key,food_present and
optionally image_path. Use --images for original stimulus image folders.
Output includes manual_size_1to10, editable in a spreadsheet. Subsequent
analysis uses manual labels by default; --include-provisional is exploratory.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

EXT={'.jpg','.jpeg','.png','.webp','.bmp'}
# Typical largest physical dimension (cm) of ONE named item. Approximate,
# curated illustrative values, not measurements of pictured objects.
# Mixed dishes or servings aren't assigned fixed physical size.
CATALOG = [
    ('single grain of rice', 0.65),('sesame seed',0.35),('peppercorn',0.50),
    ('single lentil',0.55),('single green pea',0.90),('single corn kernel',1.00),
    ('blueberry',1.2),('peanut',1.8),('raspberry',1.7),('almond',2.4),
    ('grape',2.4),('cherry',2.3),('olive',2.5),('cherry tomato',2.7),
    ('strawberry',3.6),('small mushroom',4.5),('small cookie',5),
    ('chicken egg',5.7),('lime',5.8),('kiwi fruit',6.4),('plum',6.3),
    ('lemon',8),('orange',8),('apple',8),('peach',8.5),('pear',9),
    ('pomegranate',10),('avocado',11),('mango',13),('bread roll',11),
    ('hamburger',11),('donut',9),('sandwich',12),('taco',14),
    ('head of broccoli',14),('whole cauliflower',17),('head of cabbage',19),
    ('head of lettuce',19),('whole cantaloupe melon',18),
    ('banana',19),('whole carrot',19),('corn on the cob',20),
    ('whole cucumber',22),('zucchini',19),('hot dog',17),
    ('pineapple',26),('loaf of bread',28),('whole pizza',33),
    ('watermelon',33),('pumpkin',36),('baguette loaf',55),
]
# <= boundaries: grain of rice = 1; watermelon = 10.
SIZE_BOUNDARIES_CM = [0.75,1.5,3,5,7,10,15,22,30]


def cm_to_score(cm):
    x=float(cm)
    if not np.isfinite(x) or x <= 0: raise ValueError('cm must be finite and positive')
    return int(np.searchsorted(SIZE_BOUNDARIES_CM,x,side='left')+1)


def key(s):
    if pd.isna(s): return ''
    name=str(s).strip().replace('\\','/').split('/')[-1]
    return Path(name).stem.casefold()


def find_images(keys,roots):
    found={};duplicates=set()
    for root in roots:
        p=Path(root).expanduser()
        if not p.is_dir():raise SystemExit(f'Image folder does not exist: {p}')
        for directory,dirs,files in os.walk(p):
            dirs[:]=[d for d in dirs if d not in {'.git','.ipynb_checkpoints','node_modules','__pycache__'}]
            for name in files:
                if Path(name).suffix.lower() not in EXT:continue
                k=key(name)
                if k not in keys:continue
                path=str(Path(directory)/name)
                if k in found and found[k]!=path:duplicates.add(k)
                else:found[k]=path
    for k in duplicates:found.pop(k,None)
    if duplicates:print(f'WARNING: ignored {len(duplicates)} ambiguous duplicate image filenames')
    return found


def clean_dataframe(df):
    if 'image_key' not in df.columns:
        raise SystemExit('CSV needs image_key (from 04_final_food_labels.csv).')
    if 'food_present' not in df.columns:
        raise SystemExit('CSV needs food_present. Do not automatically classify unlabeled images as food.')
    df=df.copy()
    df['image_key']=df.image_key.map(key)
    if (df.image_key=='').any() or df.image_key.duplicated().any():
        raise SystemExit('Missing or duplicate normalized image_key in input; fix before scoring')
    y=pd.to_numeric(df.food_present,errors='coerce')
    invalid=y.notna() & ~y.isin([0,1])
    if invalid.any():raise SystemExit('food_present must be 0, 1, or blank')
    df['food_present']=y
    return df


def infer(df, todo, args, laion_images=None):
    try:
        import torch,open_clip
        from PIL import Image
    except ImportError as e:
        raise SystemExit('CLIP dependencies missing: pip install open_clip_torch torch pillow') from e
    torch.set_num_threads(max(1,args.threads))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('CLIP device:',device,'(may download pretrained model on first run)')
    model,_,preproc=open_clip.create_model_and_transforms(
        args.model,pretrained=args.pretrained,device=device)
    model.eval();tokenizer=open_clip.get_tokenizer(args.model)
    prompts=[]
    for name,_ in CATALOG:
        prompts.extend([f'a photograph of {name} as the main food object',
                        f'a close-up photograph of {name}'])
    with torch.inference_mode():
        features=model.encode_text(tokenizer(prompts).to(device))
        features=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        features=features.reshape(len(CATALOG),2,-1).mean(dim=1)
        features=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    for start in range(0,len(todo),args.batch_size):
        part=todo[start:start+args.batch_size]
        idx=[];batch=[]
        for i,p in part:
            try:
                if p.startswith('laion:'):
                    if laion_images is None: raise RuntimeError('Missing LAION images reader')
                    im=laion_images.get(p[len('laion:'):])
                    batch.append(preproc(im.convert('RGB')))
                    idx.append(i)
                else:
                    with Image.open(p) as im:
                        batch.append(preproc(im.convert('RGB')))
                        idx.append(i)
            except Exception as e:print('Skipping unreadable image',p,repr(e))
        if not batch:continue
        with torch.inference_mode():
            em=model.encode_image(torch.stack(batch).to(device))
            em=em/em.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            cosine=(em@features.T).float().cpu().numpy()
        for row,score in zip(idx,cosine):
            ii=np.argsort(score)[::-1]
            first,second=ii[:2]
            name,cm=CATALOG[int(first)]
            df.loc[row,'clip_food_item']=name
            df.loc[row,'clip_second_item']=CATALOG[int(second)][0]
            df.loc[row,'clip_top_cosine']=float(score[first])
            df.loc[row,'clip_top2_margin']=float(score[first]-score[second])
            df.loc[row,'clip_typical_cm']=cm
            df.loc[row,'clip_size_1to10_provisional']=cm_to_score(cm)
        print('CLIP scored:',min(start+args.batch_size,len(todo)),'/',len(todo),flush=True)
    return df


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='fast_labels/04_final_food_labels.csv')
    p.add_argument('--images',nargs='*',default=[],help='Original stimulus image folder(s), searched recursively')
    p.add_argument('--from-laion-stimuli',action='store_true',
                   help='Read images directly from the official LAION-fMRI stimulus HDF5 via laion_fmri.load_stimuli() (DUA access required)')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior',type=Path,help='Previously reviewed output CSV: retain manual size corrections')
    p.add_argument('--model',default='ViT-B-32')
    p.add_argument('--pretrained',default='laion2b_s34b_b79k')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--no-clip',action='store_true',help='Prepare/reconcile CSV without loading CLIP')
    a=p.parse_args()
    if a.batch_size<1:p.error('--batch-size must be >=1')
    df=clean_dataframe(pd.read_csv(a.input))
    columns=['clip_food_item','clip_second_item','clip_top_cosine','clip_top2_margin',
             'clip_typical_cm','clip_size_1to10_provisional','manual_food_item',
             'manual_size_1to10','size_review_status']
    for c in columns:
        if c not in df.columns:df[c]='' if c in {'clip_food_item','clip_second_item','manual_food_item','size_review_status'} else np.nan
    if 'image_path' not in df.columns:df['image_path']=''
    df['image_path']=df.image_path.fillna('')
    if a.prior is not None:
        old=pd.read_csv(a.prior)
        old['image_key']=old.image_key.map(key)
        if old.image_key.duplicated().any():raise SystemExit('Duplicate image keys in --prior CSV')
        old=old.set_index('image_key')
        for col in ['manual_size_1to10','manual_food_item','size_review_status',
                    'clip_food_item','clip_second_item','clip_top_cosine',
                    'clip_top2_margin','clip_typical_cm','clip_size_1to10_provisional',
                    'image_path']:
            if col in old:
                value=df.image_key.map(old[col])
                if col in ['manual_size_1to10','clip_top_cosine','clip_top2_margin',
                           'clip_typical_cm','clip_size_1to10_provisional']:
                    df[col]=pd.to_numeric(value,errors='coerce').combine_first(pd.to_numeric(df[col],errors='coerce'))
                else:
                    value=value.replace('',np.nan)
                    df[col]=value.combine_first(df[col])
    manual=pd.to_numeric(df.manual_size_1to10,errors='coerce')
    bad=manual.notna() & ((manual<1)|(manual>10)|(manual!=np.floor(manual)))
    if bad.any():raise SystemExit('Manual size must be an INTEGER from 1 to 10')
    df['manual_size_1to10']=manual
    food=df.food_present==1
    if a.images:
        found=find_images(set(df.loc[food,'image_key']),a.images)
        ix=df.image_key.map(found).fillna('')
        df.loc[ix!='','image_path']=ix[ix!='']
    paths=df.image_path.fillna('').astype(str)
    available=food & paths.map(lambda x:bool(x) and Path(x).is_file())
    laion_images=None
    laion_names={}
    if a.from_laion_stimuli and not a.no_clip:
        try:
            import laion_fmri
            stim=laion_fmri.load_stimuli()
            laion_images=stim.images
            all_names=laion_images.names()
            for name in all_names:
                k=key(name)
                if k in laion_names and laion_names[k]!=name:
                    laion_names[k]=None  # ambiguous basenames are excluded
                elif k not in laion_names:
                    laion_names[k]=str(name)
        except Exception as e:
            raise SystemExit('Could not open LAION stimulus archive. Run: laion-fmri download-stimuli '
                             '(requires accepting dataset Data Use Agreement). Details: '+str(e)) from e
    todo=[]
    size_existing=pd.to_numeric(df.clip_size_1to10_provisional,errors='coerce')
    for i in df.index[food & size_existing.isna()]:
        if available.loc[i]:todo.append((i,str(paths.loc[i])))
        elif laion_names.get(df.at[i,'image_key']):
            todo.append((i,'laion:'+laion_names[df.at[i,'image_key']]))
    print('Unique images:',len(df),'food:',int(food.sum()),
          'with local JPGs:',int(available.sum()),
          'available in LAION HDF5:',sum(bool(laion_names.get(k)) for k in df.loc[food,'image_key']),
          'to CLIP:',len(todo))
    if todo and not a.no_clip:df=infer(df,todo,a,laion_images=laion_images)
    elif not a.no_clip:
        print('No matching ORIGINAL stimulus images located. CLIP cannot score from filenames alone. '
              'Use --images or download LAION stimulus HDF5 and --from-laion-stimuli.')
    score=pd.to_numeric(df.clip_size_1to10_provisional,errors='coerce')
    # Store proposed size ONLY. Never quietly convert to manually verified status.
    df['size_1to10']=manual.where(manual.notna(),score).where(food,np.nan)
    df['size_source']=np.select([manual.notna() & food,score.notna() & food],
                                ['manual','clip_provisional'],default='unscored')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.output,index=False)
    review=df.loc[food].copy()
    review['needs_review']=review.size_source!='manual'
    review=review.sort_values(['needs_review','clip_top2_margin'],ascending=[False,True])
    review_file=a.output.with_name('FOOD_SIZE_REVIEW.csv')
    review.to_csv(review_file,index=False)
    print(f'Wrote: {a.output}, review queue: {review_file}')
    print('Size source counts among food images:\n'+df.loc[food,'size_source'].value_counts().to_string())
    print('Important: 1 = rice-grain scale; 10 = watermelon scale. NOT size in the photo.')
    print('Mixed meals, multiple food objects and uncertain CLIP item names require manual inspection.')

if __name__=='__main__':main()

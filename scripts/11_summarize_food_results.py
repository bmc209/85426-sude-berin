#!/usr/bin/env python3
"""Produce evidence-grounded markdown report and one-slide text from saved consensus food tests."""
import argparse,json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--method',default='bayesian');p.add_argument('--rank',type=int,default=20);p.add_argument('--output-dir',type=Path,default=Path('analysis/food_selectivity/fast_report'))
a=p.parse_args()
folder=a.input_dir/f'consensus_{a.method}_rank{a.rank}'/'food_evaluation'
s=json.loads((folder/'summary.json').read_text());df=pd.read_csv(folder/'component_food_selectivity.csv')
c=df.iloc[0];significant=(df.p_FWER_20components<.05).sum()
a.output_dir.mkdir(parents=True,exist_ok=True)
summary=f'''# Focused food-selectivity analysis: progress summary

## Research questions

We ask whether hypothesis-neutral decomposition of ventral visual cortex responses yields a component whose response is greater for food-containing than non-food images. We also ask whether food associations remain after accounting for the stimulus family (LAION, NSD, OOD or THINGS+), and whether stronger food prominence predicts greater component response. The prominence question remains untested unless independently collected prominence ratings are available; the Moondream keyword labels only support presence/absence tests.

## Dataset and analysis

We extracted single-trial fMRI responses from **sub-01/ses-01** using the LAION ventral ROI (3,767 voxels; 1,044 image presentations of 769 unique images). The preprocessing script standardized each voxel within session, averaged repeated presentations of each image, and shifted voxel responses to nonnegative values. We fitted a 20-component NMF model with 50 independent Bayesian runs (using the Nimfa implementation, rather than the original authors' exact Bayesian sampler configuration). The consensus procedure retained 931 of 1,000 component profiles. A consensus is intended to reduce dependence on individual random initializations, not to guarantee physiological validity.

Existing Moondream object descriptions supplied by our group were transformed into provisional food/non-food labels. To avoid mixing data and image order, images were joined to the NMF response profiles using their filenames. Images with no food label were excluded, rather than counted as non-food. Because the analysis uses one response per distinct image, repeated presentations do not inflate the statistical sample. We then correlated each component's activation with the binary label, and compared food and non-food response distributions. A maximum-absolute-correlation permutation test corrected for examining all {a.rank} consensus components. We also tested associations within stimulus family to reduce family-related confounding.

## Results

The analysis included **{s['labeled_images']} unique labeled images**: **{s['food_images']} food** and **{s['nonfood_images']} non-food**. The strongest positive association was **consensus component {int(c.component)}**, with food correlation **r={c.food_r:.3f}** and mean difference **{c.food_minus_nonfood_sd:.3f} within-component standard deviations**. Its food-higher AUC was **{c.food_higher_auc:.3f}**. The two-sided permutation p-value, corrected across all {a.rank} components, was **{c.p_FWER_20components:.4f}**. The within-family correlation was **r={c.within_family_r:.3f}**, with corrected **p={c.within_family_p_FWER_20components:.4f}**. **{int(significant)}** components passed corrected p < 0.05. These p-values are exploratory, particularly because categories were provisionally labeled and component identification follows inspection of the data.

Figure 1 (`all_component_food_correlations.png`) displays the association of every consensus component with the provisional food labels. Positive bars represent greater responses on food images. Figure 2 (`best_component_food_scatter.png`) shows image-level response distributions for the strongest candidate, including the amount of overlap between food and non-food examples. The complete numerical results are available in `component_food_selectivity.csv`, and `best_component_images.csv` identifies which images drive the candidate.

## Interpretation and relation to Khosla et al.

Khosla et al. (2022) used a hypothesis-neutral 20-component Bayesian NMF analysis to identify consistent ventral-visual-response components across eight NSD participants, and characterized a response strongly associated with food salience. Our LAION experiment applies the same conceptual strategy to an independent stimulus dataset. It does **not** reproduce the original NSD values, participant count, exact ventral ROI, human salience ratings, or complete consensus algorithm. Consequently, a positive exploratory correlation is not sufficient to conclude that we have recovered the same neural population. In particular, category presence and perceptual prominence should not be conflated.

Important uncertainties are incomplete image labeling, the small number of labeled food exemplars, image-family composition, one participant/session, nonidentical anatomical masks, and possible issues with the limited 12-iteration Bayesian fits. Replication across sessions and participants, using reviewed image labels or five independent 0-9 prominence ratings, will provide stronger evidence. The finding should be described as **a candidate food-associated component** unless independent testing and corrected statistics support a more specific claim.

## Next steps

First, manually inspect automatically labeled food examples and high-response non-food examples. Second, label the previously missing images to expand coverage and rerun the fixed 20-component test with multiple-comparison correction. Third, use actual food-prominence ratings to test the continuous hypothesis. Finally, verify component stability with adequate Bayesian sampling and test reproducibility across participants and sessions.
'''
(a.output_dir/'RESULTS_SUMMARY.md').write_text(summary)
slide=f'''# Food selectivity in LAION-fMRI: progress update

**Question:** Does a data-driven ventral visual component respond more strongly to food images?

**Method:** sub-01/ses-01; 769 unique images × 3,767 ventral voxels; 20-component Bayesian NMF (50 runs) + consensus; provisional Moondream food labels; image-level permutation tests.

**Data used:** {s['labeled_images']} labeled images ({s['food_images']} food; {s['nonfood_images']} non-food).

**Best candidate:** Component {int(c.component)}, food correlation r={c.food_r:.3f}; corrected p={c.p_FWER_20components:.3f}; within-family corrected p={c.within_family_p_FWER_20components:.3f}.

**Interpretation:** {'Corrected association detected; requires independent validation.' if c.p_FWER_20components<.05 else 'No significant food-selective component after correction; exploratory candidate only.'} Further image-label review and additional sessions are needed.

**Suggested figure:** `{folder/'all_component_food_correlations.png'}`
'''
(a.output_dir/'ONE_SLIDE_CONTENT.md').write_text(slide)
print('Wrote',a.output_dir/'RESULTS_SUMMARY.md');print('Wrote',a.output_dir/'ONE_SLIDE_CONTENT.md')
print('Best component:',int(c.component),'r=',float(c.food_r),'corrected p=',float(c.p_FWER_20components))

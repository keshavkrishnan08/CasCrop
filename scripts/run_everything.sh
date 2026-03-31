#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "CasCrop Full Experiment Suite"
echo "Started: $(date)"
echo "=========================================="

# Phase 1: Main ablation (~4 hrs)
echo ""
echo "=== PHASE 1: Main Ablation (5 models × 5 seeds) ==="
python scripts/04_train_all.py \
    --epochs 100 --patience 15 --batch-size 512 \
    --seeds 42 123 456 789 1024 --resume

# Phase 2: Graph perturbation (~30 min)
echo ""
echo "=== PHASE 2: Graph Perturbation ==="
python -c "
import numpy as np
g = np.load('data/graphs/combined_graph.npz')
np.random.seed(42)
np.savez('data/graphs/shuffled.npz',
    edge_index=np.array([g['edge_index'][0], np.random.permutation(g['edge_index'][1])]),
    edge_weight=g['edge_weight'])
print('Shuffled graph created')
"
python scripts/04_train_all.py \
    --models cascrop --seeds 42 123 456 \
    --epochs 100 --patience 15 --batch-size 512 \
    --graph data/graphs/shuffled.npz
cp results/training_results.json results/perturbation_results.json

# Phase 3: Edge type ablation (~1 hr)
echo ""
echo "=== PHASE 3: Edge Ablation ==="
python -c "
import numpy as np
from scipy import sparse

def sp2npz(mat, path, k=20):
    d = mat.toarray(); n = d.shape[0]; r,c,v = [],[],[]
    for i in range(n):
        nz = np.where(d[i]>0)[0]
        if len(nz)==0: continue
        top = nz[np.argsort(d[i,nz])[-k:]]
        for j in top: r.append(i);c.append(j);v.append(d[i,j])
    np.savez(path, edge_index=np.array([r,c]), edge_weight=np.array(v))
    print(f'{path}: {len(v)} edges')

sp2npz(sparse.load_npz('data/graphs/adjacency_geo.npz'), 'data/graphs/geo_only.npz')
comm = (sparse.load_npz('data/graphs/adjacency_commodity_corn.npz') +
        sparse.load_npz('data/graphs/adjacency_commodity_soybeans.npz') +
        sparse.load_npz('data/graphs/adjacency_commodity_wheat.npz')) / 3
sp2npz(comm, 'data/graphs/comm_only.npz')
"

python scripts/04_train_all.py \
    --models cascrop --seeds 42 123 456 \
    --epochs 100 --patience 15 --batch-size 512 \
    --graph data/graphs/geo_only.npz
cp results/training_results.json results/edge_geo_only.json

python scripts/04_train_all.py \
    --models cascrop --seeds 42 123 456 \
    --epochs 100 --patience 15 --batch-size 512 \
    --graph data/graphs/comm_only.npz
cp results/training_results.json results/edge_comm_only.json

# Phase 4: Restore main results + evaluate
echo ""
echo "=== PHASE 4: Restore Main Results ==="
python scripts/04_train_all.py \
    --epochs 100 --patience 15 --batch-size 512 \
    --seeds 42 123 456 789 1024 --resume

# Phase 5: Generate all outputs
echo ""
echo "=== PHASE 5: Evaluation + Figures ==="
python scripts/05_evaluate_and_publish.py

# Phase 6: Disentanglement probe
echo ""
echo "=== PHASE 6: Disentanglement ==="
python -c "
import sys; sys.path.insert(0,'src')
import torch, json, numpy as np
from models.cascrop import CasCrop
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

with open('data/processed/stats.json') as f: stats=json.load(f)
with open('data/processed/feature_groups.json') as f: groups=json.load(f)
with open('data/processed/splits.json') as f: splits=json.load(f)
features = pd.read_parquet('data/processed/features.parquet')

dev='cpu'
mdl=CasCrop(bio_input_dim=len(groups['biophysical']),econ_input_dim=len(groups['economic']),
            hist_dim=len(groups['historical']),latent_dim=64,num_heads=4,dropout=0.0).to(dev)
ckpt=torch.load('checkpoints/cascrop_seed42.pt',map_location=dev,weights_only=False)
mdl.load_state_dict(ckpt['model_state_dict']);mdl.eval()

tf=features.iloc[splits['test']]
def nm(df,cols):
    X=df[cols].values.astype(np.float32)
    for i,c in enumerate(cols):
        if c in stats:X[:,i]=(X[:,i]-stats[c]['mean'])/max(stats[c]['std'],1e-8)
    return torch.from_numpy(np.nan_to_num(X,0.0))
b={'x_bio':nm(tf,groups['biophysical']),'x_econ':nm(tf,groups['economic']),'x_hist':nm(tf,groups['historical']),
   'edge_index':torch.stack([torch.arange(len(tf)),torch.arange(len(tf))]),'edge_attr':None,'price_shocks':torch.zeros(len(tf),1)}
with torch.no_grad():out=mdl(b)
zb,ze=out['z_bio'].numpy(),out['z_econ'].numpy()
el=KMeans(5,random_state=42,n_init=10).fit_predict(ze)
zs=StandardScaler().fit_transform(zb);h=len(zb)//2
acc=LogisticRegression(max_iter=1000,random_state=42).fit(zs[:h],el[:h]).score(zs[h:],el[h:])
print(f'Disentanglement probe: {acc:.3f} (pass if <0.55)')
json.dump({'probe_accuracy':float(acc)},open('results/disentanglement.json','w'))
"

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Results in: results/"
echo "Figures in: paper/figures/"
echo "Tables in:  paper/tables/"

# Final summary
python -c "
import json, pandas as pd
df = pd.DataFrame(json.load(open('results/training_results.json')))
print()
for m in ['local_only','local_econ','geo_gat','symmetric_ecmp','cascrop']:
    d=df[df['model']==m]
    if len(d): print(f'{m:<20} AUC={d[\"test_auc_roc\"].mean():.3f}+/-{d[\"test_auc_roc\"].std():.3f}')
print()
pr=pd.DataFrame(json.load(open('results/perturbation_results.json')))
print(f'Shuffled graph:      AUC={pr[\"test_auc_roc\"].mean():.3f}')
geo=pd.DataFrame(json.load(open('results/edge_geo_only.json')))
print(f'Geo-only edges:      AUC={geo[\"test_auc_roc\"].mean():.3f}')
comm=pd.DataFrame(json.load(open('results/edge_comm_only.json')))
print(f'Commodity-only edges: AUC={comm[\"test_auc_roc\"].mean():.3f}')
dis=json.load(open('results/disentanglement.json'))
print(f'Disentanglement:     probe={dis[\"probe_accuracy\"]:.3f}')
"

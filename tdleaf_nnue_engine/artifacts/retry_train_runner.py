import time
import json
import pathlib
import subprocess
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tdleaf_nnue_engine.eval import Evaluator
from tdleaf_nnue_engine.nnue_features import FEATURE_SIZE
from tdleaf_nnue_engine.nnue_model import NNUEModel
from tdleaf_nnue_engine.search import Searcher
from tdleaf_nnue_engine.selfplay_tdleaf import SelfPlayConfig, generate_tdleaf_dataset

root = pathlib.Path('.')
outdir = root / 'tdleaf_nnue_engine' / 'checkpoints'
outdir.mkdir(parents=True, exist_ok=True)
metrics_dir = root / 'tdleaf_nnue_engine' / 'artifacts'
metrics_dir.mkdir(parents=True, exist_ok=True)

start = time.time()
searcher = Searcher(evaluator=Evaluator())
model = NNUEModel(input_dim=FEATURE_SIZE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

probe_cfg = SelfPlayConfig(games=2, max_plies=28, search_depth=2, seed=7)
x_np, y_np = generate_tdleaf_dataset(searcher, probe_cfg)
if x_np.size:
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()
else:
    x = torch.zeros((1, FEATURE_SIZE))
    y = torch.zeros((1,))
loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

probe_losses = []
model.train()
for _ in range(4):
    ep_losses = []
    for xb, yb in loader:
        pred = model(xb)
        loss = crit(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ep_losses.append(float(loss.item()))
    probe_losses.append(sum(ep_losses) / max(1, len(ep_losses)))

criterion_pass = len(probe_losses) >= 2 and probe_losses[-1] < probe_losses[0]
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'input_dim': FEATURE_SIZE,
        'hidden_dim': model.fc1.out_features,
    },
    outdir / 'nnue_model_probe.pt',
)
probe_elapsed = time.time() - start

extended = {'executed': False, 'minutes': 0.0, 'cycles': 0, 'cycle_losses': []}
if criterion_pass:
    ext_start = time.time()
    target_seconds = 30 * 60
    cycle = 0
    while time.time() - ext_start < target_seconds:
        cycle += 1
        cfg = SelfPlayConfig(games=2, max_plies=28, search_depth=2, seed=100 + cycle)
        x_np, y_np = generate_tdleaf_dataset(searcher, cfg)
        if x_np.size:
            x = torch.from_numpy(x_np).float()
            y = torch.from_numpy(y_np).float()
        else:
            x = torch.zeros((1, FEATURE_SIZE))
            y = torch.zeros((1,))
        loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
        cycle_losses = []
        for _ in range(2):
            for xb, yb in loader:
                pred = model(xb)
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                cycle_losses.append(float(loss.item()))
        extended['cycle_losses'].append(sum(cycle_losses) / max(1, len(cycle_losses)))
        if cycle % 10 == 0:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'input_dim': FEATURE_SIZE,
                    'hidden_dim': model.fc1.out_features,
                },
                outdir / f'nnue_model_cycle_{cycle}.pt',
            )
    extended['executed'] = True
    extended['minutes'] = (time.time() - ext_start) / 60.0
    extended['cycles'] = cycle

torch.save(
    {
        'model_state_dict': model.state_dict(),
        'input_dim': FEATURE_SIZE,
        'hidden_dim': model.fc1.out_features,
    },
    outdir / 'nnue_model.pt',
)

export_cmd = [
    sys.executable,
    '-m',
    'tdleaf_nnue_engine.export',
    '--checkpoint',
    str(outdir / 'nnue_model.pt'),
    '--output',
    str(outdir / 'nnue_runtime.npz'),
]
exp = subprocess.run(export_cmd, capture_output=True, text=True)

smoke_cmd = [
    sys.executable,
    '-m',
    'tdleaf_nnue_engine.main',
    '--depth',
    '1',
    '--weights',
    str(outdir / 'nnue_runtime.npz'),
]
smoke = subprocess.run(smoke_cmd, capture_output=True, text=True, timeout=120)

summary = {
    'commands': {
        'probe_train': 'python tdleaf_nnue_engine/artifacts/retry_train_runner.py',
        'extended_train': 'same script loops ~30 min when criterion passes',
        'export': ' '.join(export_cmd),
        'smoke': ' '.join(smoke_cmd),
    },
    'criterion': 'final_probe_loss < initial_probe_loss',
    'probe_losses': probe_losses,
    'criterion_pass': criterion_pass,
    'probe_elapsed_seconds': probe_elapsed,
    'extended': extended,
    'export_rc': exp.returncode,
    'export_stdout': exp.stdout,
    'export_stderr': exp.stderr,
    'smoke_rc': smoke.returncode,
    'smoke_stdout': smoke.stdout,
    'smoke_stderr': smoke.stderr,
    'artifacts': [str(p) for p in sorted(outdir.glob('*'))],
}

(metrics_dir / 'training_retry_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
(metrics_dir / 'training_retry_summary.txt').write_text(
    '\n'.join(
        [
            f"criterion={summary['criterion']}",
            f"criterion_pass={summary['criterion_pass']}",
            f"probe_losses={summary['probe_losses']}",
            f"probe_elapsed_seconds={summary['probe_elapsed_seconds']:.2f}",
            f"extended_executed={summary['extended']['executed']}",
            f"extended_minutes={summary['extended']['minutes']:.2f}",
            f"extended_cycles={summary['extended']['cycles']}",
            f"export_rc={summary['export_rc']}",
            f"smoke_rc={summary['smoke_rc']}",
            'artifacts:',
            *summary['artifacts'],
        ]
    ),
    encoding='utf-8',
)
print(metrics_dir / 'training_retry_summary.json')

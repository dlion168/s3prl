import s3prl.hub as hub
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_0 = getattr(hub, 'Salmonn7B')()
model_0.to(device)

wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    reps = model_0(wavs)
    print(len(reps['hidden_states']))
    print(reps['hidden_states'][0].shape)
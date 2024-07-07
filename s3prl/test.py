import s3prl.hub as hub
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_0 = getattr(hub, 'speechgpt')()
model_0.to(device)
wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    reps = model_0(wavs)["hidden_states"]
    print(reps[0].shape)
    print(len(reps[0]))
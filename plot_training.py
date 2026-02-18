import pandas as pd
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'log.csv'
df = pd.read_csv(filename)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Score
ax1.plot(df['epoch'], df['score'], alpha=0.3, label='score')
ax1.plot(df['epoch'], df['score'].rolling(50).mean(), label='avg (50)')
ax1.set_ylabel('Score')
ax1.legend()
ax1.set_title('Training Progress')

# Loss
valid = df[df['loss'].notna() & (df['loss'] != 'None')]
valid_loss = valid['loss'].astype(float)
ax2.plot(valid['epoch'], valid_loss, alpha=0.3, label='loss')
ax2.plot(valid['epoch'], valid_loss.rolling(50).mean(), label='avg (50)')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150)
plt.show()

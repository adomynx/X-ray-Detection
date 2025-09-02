import matplotlib.pyplot as plt
import numpy as np

# Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

# Scores from your data
mobilenetv2 = [0.8766025641025641, 0.8614318706697459, 0.9564102564102565, 0.9064398541919806, 0.9489151873767259]
vgg16       = [0.8990384615384616, 0.8724373576309795, 0.982051282051282, 0.9240048250904704, 0.9676309445540214]
resnet50    = [0.375, 0.0, 0.0, 0.0, 0.6924172693403463]
customcnn   = [0.8605769230769231, 0.904, 0.8692307692307693, 0.8862745098039215, 0.93088976550515]

# Collect all
all_scores = [mobilenetv2, vgg16, resnet50, customcnn]
algos = ['MobileNetV2', 'VGG16', 'ResNet50', 'Custom CNN']

x = np.arange(len(metrics))  # X positions
width = 0.18  # Bar width

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

for i, scores in enumerate(all_scores):
    bars = ax.bar(x + i*width - (len(all_scores)-1)/2*width,
                  scores, width, label=algos[i])
    # Add labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Labels and formatting
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison Across 4 Algorithms')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.show()

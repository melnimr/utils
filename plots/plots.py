plt.figure(figsize=(12,12))
for i, (image, label) in enumerate(train_data.take(6)):
    plt.subplot(3, 2, i+1)
    plt.imshow(image)
    plt.title(int(label))

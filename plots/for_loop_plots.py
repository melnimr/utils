fig, axs = plt.subplots(2,4, figsize=(15, 6), facecolor='w', edgecolor='k')
axes = axs.ravel()
for i,filename in enumerate(glob.glob("/Users/neo/wellth-wrk/hackathon/raw_images/IMG_*")):
    tmp_image = cv2.imread(filename)
    gray = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)
    print(np.mean(gray))
    g_hist = gray.flatten()
    axes[i].hist(g_hist,bins=50)

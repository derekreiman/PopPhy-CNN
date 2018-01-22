import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy.interpolate import spline
from parse_result import parse_result

plt.switch_backend('agg')

cirr = pd.read_csv("../data/Cirrhosis/features_svm.txt", sep=",", header=None)
x = np.array(cirr[[0]]).reshape(len(cirr[[0]]))
l1 = np.array(cirr[[1]]).reshape(len(cirr[[0]]))
l2 = np.array(cirr[[2]]).reshape(len(cirr[[0]]))
l3 = np.array(cirr[[3]]).reshape(len(cirr[[0]]))
l4 = np.array(cirr[[4]]).reshape(len(cirr[[0]]))
l5 = np.array(cirr[[5]]).reshape(len(cirr[[0]]))

_, cir_svm_roc_list, _, _, _, _, _, _ = parse_result("Cirrhosis", "svm")
_, cir_rf_roc_list, _, _, _, _, _, _ = parse_result("Cirrhosis", "rf")

svm_auc = np.mean(cir_svm_roc_list)
rf_auc = np.mean(cir_rf_roc_list)

x_smooth = np.linspace(x.min(), x.max(), 200)

print(x)
print(l1)

l1_smooth = spline(x, l1, x_smooth)
l2_smooth = spline(x, l2, x_smooth)
l3_smooth = spline(x, l3, x_smooth)
l4_smooth = spline(x, l4, x_smooth)
l5_smooth = spline(x, l5, x_smooth)

auc_line1 = np.array([svm_auc for i in xrange(len(x_smooth))])
auc_line2 = np.array([rf_auc for i in xrange(len(x_smooth))])

fig = plt.figure(dpi=300, figsize=(9,9), tight_layout={'pad':3})
ax = fig.add_subplot(111)
ax.set_xlim([1,50])
ax.set_xlabel("Number of Features", fontsize=18)
ax.set_ylim([0.5,1])
ax.set_ylabel("ROC AUC", fontsize=18)
ax.set_title("Cirrhosis SVM AUC Using CNN Selected Features", fontsize=18)
ax.plot(x_smooth, l1_smooth, "g", x_smooth, l2_smooth, "r", x_smooth, l3_smooth, "teal", x_smooth, l4_smooth, "gold", x_smooth, l5_smooth, "purple")
ax.plot(x_smooth, auc_line1, color="orange", linestyle="dashed", lw=1)
ax.plot(x_smooth, auc_line2, color="purple", linestyle="dashed", lw=1)

healthy_patch = mpatches.Patch(color='green', label="Healthy Features")
disease_patch = mpatches.Patch(color='red', label="Disease Features")
combo_patch = mpatches.Patch(color='teal', label="Both Sets of Features")
stn_patch = mpatches.Patch(color='gold', label="Using Signal-to-Noise Ratio")
rf_patch = mpatches.Patch(color='purple', label="Using RF Features")

plt.legend(handles=[healthy_patch, disease_patch, combo_patch, stn_patch, rf_patch], bbox_to_anchor=(1, 0), loc='lower right', fontsize=14)
plt.savefig('../images/cirrhosis_svm_features.png')



cirr = pd.read_csv("../data/Obesity/features_svm.txt", sep=",", header=None)
x = np.array(cirr[[0]]).reshape(len(cirr[[0]]))
l1 = np.array(cirr[[1]]).reshape(len(cirr[[0]]))
l2 = np.array(cirr[[2]]).reshape(len(cirr[[0]]))
l3 = np.array(cirr[[3]]).reshape(len(cirr[[0]]))
l4 = np.array(cirr[[4]]).reshape(len(cirr[[0]]))
l5 = np.array(cirr[[5]]).reshape(len(cirr[[0]]))

_, cir_svm_roc_list, _, _, _, _, _, _ = parse_result("Obesity", "svm")
_, cir_rf_roc_list, _, _, _, _, _, _ = parse_result("Obesity", "rf")

svm_auc = np.mean(cir_svm_roc_list)
rf_auc = np.mean(cir_rf_roc_list)

x_smooth = np.linspace(x.min(), x.max(), 200)

print(x)
print(l1)

l1_smooth = spline(x, l1, x_smooth)
l2_smooth = spline(x, l2, x_smooth)
l3_smooth = spline(x, l3, x_smooth)
l4_smooth = spline(x, l4, x_smooth)
l5_smooth = spline(x, l5, x_smooth)

auc_line1 = np.array([svm_auc for i in xrange(len(x_smooth))])
auc_line2 = np.array([rf_auc for i in xrange(len(x_smooth))])

fig = plt.figure(dpi=300, figsize=(9,9), tight_layout={'pad':3})
ax = fig.add_subplot(111)
ax.set_xlim([1,50])
ax.set_xlabel("Number of Features", fontsize=18)
ax.set_ylim([0.3,0.7])
ax.set_ylabel("ROC AUC", fontsize=18)
ax.set_title("Obesity SVM AUC Using CNN Selected Features", fontsize=18)
ax.plot(x_smooth, l1_smooth, "g", x_smooth, l2_smooth, "r", x_smooth, l3_smooth, "teal", x_smooth, l4_smooth, "gold", x_smooth, l5_smooth, "purple")
ax.plot(x_smooth, auc_line1, color="orange", linestyle="dashed", lw=1)
ax.plot(x_smooth, auc_line2, color="purple", linestyle="dashed", lw=1)

healthy_patch = mpatches.Patch(color='green', label="Healthy Features")
disease_patch = mpatches.Patch(color='red', label="Disease Features")
combo_patch = mpatches.Patch(color='teal', label="Both Sets of Features")
stn_patch = mpatches.Patch(color='gold', label="Using Signal-to-Noise Ratio")
rf_patch = mpatches.Patch(color='purple', label="Using RF Features")
plt.legend(handles=[healthy_patch, disease_patch, combo_patch, stn_patch, rf_patch], bbox_to_anchor=(1, 0), loc='lower right', fontsize=14)
plt.savefig('../images/obesity_svm_features.png')






cirr = pd.read_csv("../data/T2D/features_svm.txt", sep=",", header=None)
x = np.array(cirr[[0]]).reshape(len(cirr[[0]]))
l1 = np.array(cirr[[1]]).reshape(len(cirr[[0]]))
l2 = np.array(cirr[[2]]).reshape(len(cirr[[0]]))
l3 = np.array(cirr[[3]]).reshape(len(cirr[[0]]))
l4 = np.array(cirr[[4]]).reshape(len(cirr[[0]]))
l5 = np.array(cirr[[5]]).reshape(len(cirr[[0]]))

_, cir_svm_roc_list, _, _, _, _, _, _ = parse_result("T2D", "svm")
_, cir_rf_roc_list, _, _, _, _, _, _ = parse_result("T2D", "rf")

svm_auc = np.mean(cir_svm_roc_list)
rf_auc = np.mean(cir_rf_roc_list)

x_smooth = np.linspace(x.min(), x.max(), 200)

print(x)
print(l1)

l1_smooth = spline(x, l1, x_smooth)
l2_smooth = spline(x, l2, x_smooth)
l3_smooth = spline(x, l3, x_smooth)
l4_smooth = spline(x, l4, x_smooth)
l5_smooth = spline(x, l5, x_smooth)

auc_line1 = np.array([svm_auc for i in xrange(len(x_smooth))])
auc_line2 = np.array([rf_auc for i in xrange(len(x_smooth))])

fig = plt.figure(dpi=300, figsize=(9,9), tight_layout={'pad':3})
ax = fig.add_subplot(111)
ax.set_xlim([1,50])
ax.set_xlabel("Number of Features", fontsize=18)
ax.set_ylim([0.3,0.8])
ax.set_ylabel("ROC AUC", fontsize=18)
ax.set_title("T2D SVM AUC Using CNN Selected Features", fontsize=18)
ax.plot(x_smooth, l1_smooth, "g", x_smooth, l2_smooth, "r", x_smooth, l3_smooth, "teal", x_smooth, l4_smooth, "gold", x_smooth, l5_smooth, "purple")
ax.plot(x_smooth, auc_line1, color="orange", linestyle="dashed", lw=1)
ax.plot(x_smooth, auc_line2, color="purple", linestyle="dashed", lw=1)

healthy_patch = mpatches.Patch(color='green', label="Healthy Features")
disease_patch = mpatches.Patch(color='red', label="Disease Features")
combo_patch = mpatches.Patch(color='teal', label="Both Sets of Features")
stn_patch = mpatches.Patch(color='gold', label="Using Signal-to-Noise Ratio")
rf_patch = mpatches.Patch(color='purple', label="Using RF Features")
plt.legend(handles=[healthy_patch, disease_patch, combo_patch, stn_patch, rf_patch], bbox_to_anchor=(1, 0), loc='lower right', fontsize=14)
plt.savefig('../images/T2D_svm_features.png')


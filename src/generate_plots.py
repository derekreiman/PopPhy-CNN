import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from parse_result import parse_result, parse_result_metaml
from scipy import interp
from sklearn.metrics import roc_curve, auc

plt.switch_backend('agg')
cir_cnn2_acc_list, cir_cnn2_roc_list, cir_cnn2_prec_list, cir_cnn2_recall_list, cir_cnn2_f1_list, cir_cnn2_tpr_list, cir_cnn2_fpr_list, cir_cnn2_thresh_list = parse_result("Cirrhosis", "cnn_2D")
cir_cnn1_acc_list, cir_cnn1_roc_list, cir_cnn1_prec_list, cir_cnn1_recall_list, cir_cnn1_f1_list, cir_cnn1_tpr_list, cir_cnn1_fpr_list, cir_cnn1_thresh_list = parse_result("Cirrhosis", "cnn_1D")
cir_rf_acc_list, cir_rf_roc_list, cir_rf_prec_list, cir_rf_recall_list, cir_rf_f1_list, cir_rf_tpr_list, cir_rf_fpr_list, cir_rf_thresh_list = parse_result("Cirrhosis", "rf")
cir_svm_acc_list, cir_svm_roc_list, cir_svm_prec_list, cir_svm_recall_list, cir_svm_f1_list, cir_svm_tpr_list, cir_svm_fpr_list, cir_svm_thresh_list = parse_result("Cirrhosis", "svm")
cir_lasso_acc_list, cir_lasso_roc_list, cir_lasso_prec_list, cir_lasso_recall_list, cir_lasso_f1_list, cir_lasso_tpr_list, cir_lasso_fpr_list, cir_lasso_thresh_list = parse_result("Cirrhosis", "lasso")

t2d_cnn2_acc_list, t2d_cnn2_roc_list, t2d_cnn2_prec_list, t2d_cnn2_recall_list, t2d_cnn2_f1_list, t2d_cnn2_tpr_list, t2d_cnn2_fpr_list, t2d_cnn2_thresh_list = parse_result("T2D", "cnn_2D")
t2d_cnn1_acc_list, t2d_cnn1_roc_list, t2d_cnn1_prec_list, t2d_cnn1_recall_list, t2d_cnn1_f1_list, t2d_cnn1_tpr_list, t2d_cnn1_fpr_list, t2d_cnn1_thresh_list = parse_result("T2D", "cnn_1D")
t2d_rf_acc_list, t2d_rf_roc_list, t2d_rf_prec_list, t2d_rf_recall_list, t2d_rf_f1_list, t2d_rf_tpr_list, t2d_rf_fpr_list, t2d_rf_thresh_list = parse_result("T2D", "rf")
t2d_svm_acc_list, t2d_svm_roc_list, t2d_svm_prec_list, t2d_svm_recall_list, t2d_svm_f1_list, t2d_svm_tpr_list, t2d_svm_fpr_list, t2d_svm_thresh_list = parse_result("T2D", "svm")
t2d_lasso_acc_list, t2d_lasso_roc_list, t2d_lasso_prec_list, t2d_lasso_recall_list, t2d_lasso_f1_list, t2d_lasso_tpr_list, t2d_lasso_fpr_list, t2d_lasso_thresh_list = parse_result("T2D", "lasso")

ob_cnn2_acc_list, ob_cnn2_roc_list, ob_cnn2_prec_list, ob_cnn2_recall_list, ob_cnn2_f1_list, ob_cnn2_tpr_list, ob_cnn2_fpr_list, ob_cnn2_thresh_list = parse_result("Obesity", "cnn_2D")
ob_cnn1_acc_list, ob_cnn1_roc_list, ob_cnn1_prec_list, ob_cnn1_recall_list, ob_cnn1_f1_list, ob_cnn1_tpr_list, ob_cnn1_fpr_list, ob_cnn1_thresh_list = parse_result("Obesity", "cnn_1D")
ob_rf_acc_list, ob_rf_roc_list, ob_rf_prec_list, ob_rf_recall_list, ob_rf_f1_list, ob_rf_tpr_list, ob_rf_fpr_list, ob_rf_thresh_list = parse_result("Obesity", "rf")
ob_svm_acc_list, ob_svm_roc_list, ob_svm_prec_list, ob_svm_recall_list, ob_svm_f1_list, ob_svm_tpr_list, ob_svm_fpr_list, ob_svm_thresh_list = parse_result("Obesity", "svm")
ob_lasso_acc_list, ob_lasso_roc_list, ob_lasso_prec_list, ob_lasso_recall_list, ob_lasso_f1_list, ob_lasso_tpr_list, ob_lasso_fpr_list, ob_lasso_thresh_list = parse_result("Obesity", "lasso")

fig = plt.figure(dpi=300, figsize=(14,9), tight_layout=True)
ax = plt.subplot(111)
ax.set_title("Boxplots of Cross-validated ROC AUC Values")
ax.set_ylabel("ROC AUC")
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
colors = ['green', 'blue', 'purple', 'orange', 'red']

bp = plt.boxplot([cir_cnn2_roc_list, cir_cnn1_roc_list, cir_rf_roc_list, cir_svm_roc_list, cir_lasso_roc_list], positions=[1,2,3,4,5], notch=True, widths=0.6, patch_artist=True)
for patch, fliers, col in zip(bp['boxes'], bp['fliers'], colors):
    patch.set_facecolor(col)
    plt.setp(fliers, color=col, marker='+')

bp = plt.boxplot([t2d_cnn2_roc_list, t2d_cnn1_roc_list, t2d_rf_roc_list, t2d_svm_roc_list, t2d_lasso_roc_list], positions=[7,8,9,10,11], notch=True, widths=0.6, patch_artist=True)
for patch, fliers, col in zip(bp['boxes'], bp['fliers'], colors):
    patch.set_facecolor(col)
    plt.setp(fliers, color=col, marker='+')

bp = plt.boxplot([ob_cnn2_roc_list, ob_cnn1_roc_list, ob_rf_roc_list, ob_svm_roc_list, ob_lasso_roc_list], positions=[13,14,15,16,17], notch=True, widths=0.6, patch_artist=True)
for patch, fliers, col in zip(bp['boxes'], bp['fliers'], colors):
    patch.set_facecolor(col)
    plt.setp(fliers, color=col, marker='+')


plt.xlim(0,18)
plt.ylim(0,1)
ax.set_xticklabels(['Cirrhosis', 'T2D', 'Obesity', 'Lasso'])

ax.text(1, 0.05, round(np.mean(cir_cnn2_roc_list),3), horizontalalignment='center', size='large', color=colors[0])
ax.text(2, 0.05, round(np.mean(cir_cnn1_roc_list),3), horizontalalignment='center', size='large', color=colors[1])
ax.text(3, 0.05, round(np.mean(cir_rf_roc_list),3), horizontalalignment='center', size='large', color=colors[2])
ax.text(4, 0.05, round(np.mean(cir_svm_roc_list),3), horizontalalignment='center', size='large', color=colors[3])
ax.text(5, 0.05, round(np.mean(cir_lasso_roc_list),3), horizontalalignment='center', size='large', color=colors[4])
ax.text(7, 0.05, round(np.mean(t2d_cnn2_roc_list),3), horizontalalignment='center', size='large', color=colors[0])
ax.text(8, 0.05, round(np.mean(t2d_cnn1_roc_list),3), horizontalalignment='center', size='large', color=colors[1])
ax.text(9, 0.05, round(np.mean(t2d_rf_roc_list),3), horizontalalignment='center', size='large', color=colors[2])
ax.text(10, 0.05, round(np.mean(t2d_svm_roc_list),3), horizontalalignment='center', size='large', color=colors[3])
ax.text(11, 0.05, round(np.mean(t2d_lasso_roc_list),3), horizontalalignment='center', size='large', color=colors[4])
ax.text(13, 0.05, round(np.mean(ob_cnn2_roc_list),3), horizontalalignment='center', size='large', color=colors[0])
ax.text(14, 0.05, round(np.mean(ob_cnn1_roc_list),3), horizontalalignment='center', size='large', color=colors[1])
ax.text(15, 0.05, round(np.mean(ob_rf_roc_list),3), horizontalalignment='center', size='large', color=colors[2])
ax.text(16, 0.05, round(np.mean(ob_svm_roc_list),3), horizontalalignment='center', size='large', color=colors[3])
ax.text(17, 0.05, round(np.mean(ob_lasso_roc_list),3), horizontalalignment='center', size='large', color=colors[4])

ax.set_xticks([3,9,15])

cnn2_patch = mpatches.Patch(color=colors[0], label='CNN-2D')
cnn1_patch = mpatches.Patch(color=colors[1], label='CNN-1D')
rf_patch = mpatches.Patch(color=colors[2], label='RF')
svm_patch = mpatches.Patch(color=colors[3], label='SVM')
lasso_patch = mpatches.Patch(color=colors[4], label='Lasso')
plt.legend(handles=[cnn2_patch, cnn1_patch, rf_patch, svm_patch, lasso_patch], bbox_to_anchor=(1, 1), loc='upper right')
plt.savefig("../images/boxplots.png", tight_layout=True)


fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
mean_fpr = np.linspace(0, 1, 100)
tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, cir_cnn2_fpr_list[i], cir_cnn2_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(cir_cnn2_roc_list)
std_auc = np.std(cir_cnn2_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[0],
         label=r'Mean CNN-2D ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[0], alpha=0.2)

tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, cir_cnn1_fpr_list[i], cir_cnn1_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(cir_cnn1_roc_list)
std_auc = np.std(cir_cnn1_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[1],
         label=r'Mean CNN-1D ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[1], alpha=0.2)


tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, cir_rf_fpr_list[i], cir_rf_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(cir_rf_roc_list)
std_auc = np.std(cir_rf_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[2],
         label=r'Mean RF ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[2], alpha=0.2)

tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, cir_svm_fpr_list[i], cir_svm_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(cir_svm_roc_list)
std_auc = np.std(cir_svm_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[3],
         label=r'Mean SVM ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[3], alpha=0.2)


tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, cir_lasso_fpr_list[i], cir_lasso_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(cir_lasso_roc_list)
std_auc = np.std(cir_lasso_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[4],
         label=r'Mean Lasso ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[4], alpha=0.2)



plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curves for Cirrhosis')
plt.legend(loc="lower right")

plt.savefig("../images/cirrhosis_ROC.png")




fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)





mean_fpr = np.linspace(0, 1, 100)
tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, t2d_cnn2_fpr_list[i], t2d_cnn2_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(t2d_cnn2_roc_list)
std_auc = np.std(t2d_cnn2_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[0],
         label=r'Mean CNN-2D ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[0], alpha=0.2)

tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, t2d_cnn1_fpr_list[i], t2d_cnn1_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(t2d_cnn1_roc_list)
std_auc = np.std(t2d_cnn1_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[1],
         label=r'Mean CNN-1D ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[1], alpha=0.2)


tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, t2d_rf_fpr_list[i], t2d_rf_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(t2d_rf_roc_list)
std_auc = np.std(t2d_rf_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[2],
         label=r'Mean RF ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[2], alpha=0.2)

tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, t2d_svm_fpr_list[i], t2d_svm_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(t2d_svm_roc_list)
std_auc = np.std(t2d_svm_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[3],
         label=r'Mean SVM ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[3], alpha=0.2)


tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, t2d_lasso_fpr_list[i], t2d_lasso_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(t2d_lasso_roc_list)
std_auc = np.std(t2d_lasso_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[4],
         label=r'Mean Lasso ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[4], alpha=0.2)


plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curves for T2D')
plt.legend(loc="lower right")

plt.savefig("../images/t2d_ROC.png")






fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)





mean_fpr = np.linspace(0, 1, 100)
tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, ob_cnn2_fpr_list[i], ob_cnn2_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(ob_cnn2_roc_list)
std_auc = np.std(ob_cnn2_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[0],
         label=r'Mean CNN-2D ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[0], alpha=0.2)

tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, ob_cnn1_fpr_list[i], ob_cnn1_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(ob_cnn1_roc_list)
std_auc = np.std(ob_cnn1_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[1],
         label=r'Mean CNN-1D ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[1], alpha=0.2)


tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, ob_rf_fpr_list[i], ob_rf_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(ob_rf_roc_list)
std_auc = np.std(ob_rf_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[2],
         label=r'Mean RF ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[2], alpha=0.2)

tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, ob_svm_fpr_list[i], ob_svm_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(ob_svm_roc_list)
std_auc = np.std(ob_svm_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[3],
         label=r'Mean SVM ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[3], alpha=0.2)


tprs = []
for i in range(0,100):
  tprs.append(interp(mean_fpr, ob_lasso_fpr_list[i], ob_lasso_tpr_list[i]))
  tprs [-1][0] = 0.0

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(ob_lasso_roc_list)
std_auc = np.std(ob_lasso_roc_list)
plt.plot(mean_fpr, mean_tpr, color=colors[4],
         label=r'Mean Lasso ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)/10.0
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[4], alpha=0.2)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curves for Obesity')
plt.legend(loc="lower right")

plt.savefig("../images/obesity_ROC.png")


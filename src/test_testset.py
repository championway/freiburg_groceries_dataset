from subprocess import call
import os
from shutil import rmtree

import numpy as np
from settings import CAFFE_ROOT, GPU

def evaluate_results(split):
    # create the confusion matrix and link the misclassified images
    call(["./CaffeNetAnalysis/CaffeNetAnalysisMain",
          "../splits/test.txt",
          os.path.join(os.path.abspath("../images/"), ""),
          "../caffe_data/deploy.prototxt",
          "../results/{0}/snapshots/_iter_10000.caffemodel".format(split),
          "../classid.txt",
          "../results/{0}/".format(split), str(GPU)])

def export_np_mat_with_header(mean_mat, std_dev_mat, file_to_copy_header,
                              export_filename, skip_header=0, skip_footer=0):
    with open(file_to_copy_header, 'r') as f:
        lines = f.readlines()
        header = lines[:skip_header]
        lines = lines[skip_header:len(lines) - skip_footer]
        first_column = [l.split(';')[0] for l in lines]
    with open(export_filename, 'w') as f:
        if header:
            f.writelines(header)  # copy header
        for i in range(len(first_column)):
            if len(mean_mat.shape) == 1:
                mean_stddev = [("{:.3f}".format(mean_mat[i]),
                                "{:.3f}".format(std_dev_mat[i]))]
            else:
                mean_stddev = zip(map("{:.3f}".format, mean_mat[i]),
                                  map("{:.3f}".format, std_dev_mat[i]))
            mean_mat_line = "; ".join([" +- ".join(m) for m in mean_stddev])
            f.write(first_column[i] + "; " + mean_mat_line + "; \n")

def evaluate_mean():
    confusion_mats = []
    accuracy_mats = []
    for i in range(5):
        confusion_mat_path = "../results/{0}/confusion_matrix.csv".format(i)
        accuracy_mat_path = "../results/{0}/accuracy.csv".format(i)
        confusion_mats.append(np.genfromtxt(confusion_mat_path,
                                            usecols=list(range(1, 26)),
                                            delimiter=';', skip_header=1,
                                            skip_footer=1))
        accuracy_mats.append(np.genfromtxt(accuracy_mat_path,
                                           usecols=(1,),
                                           delimiter=';'))
    export_np_mat_with_header(np.mean(confusion_mats, axis=0),
                              np.std(confusion_mats, axis=0),
                              "../results/0/confusion_matrix.csv",
                              "../results/mean_confusion_matrix_testset.csv", 1, 1)
    export_np_mat_with_header(np.mean(accuracy_mats, axis=0),
                              np.std(accuracy_mats, axis=0),
                              "../results/0/accuracy.csv",
                              "../results/mean_accuracy_matrix_testset.csv")


if __name__ == "__main__":
    for i in range(5):  # train the 5 splits
        cwd = os.getcwd()
        storage_dir = "../results/{0}/snapshots/".format(i)
        evaluate_results(i)
    evaluate_mean()

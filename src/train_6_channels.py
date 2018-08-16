from subprocess import call
import os
from shutil import rmtree
import lmdb
import numpy as np

from settings import CAFFE_ROOT, GPU
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append("/home/arg_ws3/caffe/python")
sys.path.append("/home/arg_ws3/caffe/python/caffe")
import caffe
import cv2

def img_to_mdbs(img_path, mdb_path):
    print("start to convert img to mdbs")
    print(img_path)
    print(mdb_path)
    num_img = sum(1 for line in open(img_path,"r"))
    X = np.zeros((num_img, 6, 227, 227), dtype=np.uint8)
    
    #print ("x shape is :",X.shape[1])
    y = np.zeros(num_img, dtype=np.int64)
    #print ("y shape is :",y.shape)
    #print("Image Number: ", num_img)
    index = 0
    with open (img_path, "r") as myfile:
        for line in myfile:
            line_split = line.split(" ")
            img_path = line_split[0]
            y[index] = int(line_split[1])
            #print (jpg, type(jpg))
            img_path = "../images/"+img_path
            #print (img_path)
            img1 = cv2.imread(img_path)
            img1 = cv2.resize(img1, (227, 227))
            img2 = img1
            img_con = np.dstack((img1,img2))
            #The following comment is too slow
            #Use numpy.moveaxis to make it better
            '''for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(img.shape[2]):
                        X[index][k][i][j] = img[i][j][k]'''
            X[index] = np.moveaxis(img_con, 2, 0)
            index = index + 1
            #print(index)
        map_size = X.nbytes * 10
        print ("map_size is:3*32*32*1000*10 --",map_size)
        env = lmdb.open(mdb_path, map_size=map_size)
        with env.begin(write=True) as txn:
            # txn is a Transaction object
            for i in range(num_img):
                datum = caffe.proto.caffe_pb2.Datum()
                #set channels=3
                datum.channels = X.shape[1]
                #set height =32
                datum.height = X.shape[2]
                #set width = 32
                datum.width = X.shape[3]
                datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
                datum.label = int(y[i])
                str_id = '{:08}'.format(i)
                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            
    print("==========Finish==========")

def create_lmdbs(split_num, cwd):
    #train_file_path = os.path.join(cwd, "../splits/read.txt".format(split_num))
    trainfile = os.path.join(cwd, "../splits/train{0}.txt".format(split_num))
    testfile = os.path.join(cwd, "../splits/test{0}.txt".format(split_num))
    train_lmdb_path = os.path.join(cwd, "../images/trainlmdb")
    test_lmdb_path = os.path.join(cwd, "../images/testlmdb")
    img_to_mdbs(trainfile, train_lmdb_path)
    img_to_mdbs(testfile, test_lmdb_path)

def prepare_solver_prototxt(storage_dir):
    solvertemplate_path = "../caffe_data/solvertemplate.prototxt"
    solvername = os.path.basename(solvertemplate_path)
    solverpath = storage_dir + solvername
    call(["cp", solvertemplate_path, storage_dir])
    solverfile = open(solverpath, 'a')
    solverfile.write("snapshot_prefix: " + "\"" + storage_dir + "snapshots/\"")
    solverfile.close()
    return solverpath


def train_split(split_num, cwd, solverpath):
    caffe_path = os.path.join(CAFFE_ROOT, "build/tools/caffe")
    relative_model_path = ("models/bvlc_reference_caffenet/")
    weights_path = os.path.join(CAFFE_ROOT, relative_model_path,
                                "bvlc_reference_caffenet.caffemodel")

    # fine tune from bvlc reference caffe model
    call([caffe_path, "train", "-solver", solverpath,
          "-weights", weights_path, "-gpu", str(GPU)])


def evaluate_results(split):
    # create the confusion matrix and link the misclassified images
    call(["./CaffeNetAnalysis/CaffeNetAnalysisMain",
          "../splits/test{0}.txt".format(split),
          os.path.join(os.path.abspath("../images/"), ""),
          "../caffe_data/deploy.prototxt",
          "../results/{0}/snapshots/solvertemplate_iter_1000.caffemodel".format(split),
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
                              "../results/mean_confusion_matrix.csv", 1, 1)
    export_np_mat_with_header(np.mean(accuracy_mats, axis=0),
                              np.std(accuracy_mats, axis=0),
                              "../results/0/accuracy.csv",
                              "../results/mean_accuracy_matrix.csv")


def check_if_training_files_exist(storage_dir):
    if not os.path.isdir(storage_dir):
        return
    print("It seems there already exist files from a previous"
          "training. Delete all files in folder results? y/n")
    while True:
        inp = str(input())
        if inp.lower() == 'y':
            for i in range(5):
                rmtree("../results/".format(i))
            break
        elif inp.lower() == 'n':
            exit()


if __name__ == "__main__":
    for i in range(5):  # train the 5 splits
        cwd = os.getcwd()
        storage_dir = "../results/{0}/snapshots/".format(i)
        check_if_training_files_exist(storage_dir)
        os.makedirs(storage_dir)
        solverpath = prepare_solver_prototxt("../results/{0}/".format(i))
        create_lmdbs(i, cwd)
        train_split(i, cwd, solverpath)
        evaluate_results(i)
    evaluate_mean()
    '''img1 = np.array([[[1, 2, 3],[0, 0, 0],[0, 0, 0],[0, 0, 0]],
        [[0, 0, 0],[0, 0, 0],[4, 5, 7],[0, 0, 0]]])
    img2 = np.array([[[0, 0, 0],[0, 0, 0],[6, 3, 9],[0, 0, 0]],
        [[0, 0, 0],[4, 2, 4],[0, 0, 0],[0, 0, 0]]])
    caf = np.zeros((2, 6, 2, 4), dtype=np.uint8)

    print (img1.shape)
    print (img2.shape)
    print (caf.shape)
    c = np.dstack((img1,img2))
    d = np.moveaxis(c, 2, 0)
    print (c.shape)
    print (d.shape)'''

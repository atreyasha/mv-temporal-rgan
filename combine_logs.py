#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# get all dependencies
import re
import sys
import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd

################################
# define key functions
################################

def iter_temporal_find(direct):
    directLong = "./pickles/"+direct
    dat_type = re.sub(r".*_(.*)$","\g<1>",direct)
    prefix = re.sub(r"(.*_)(.*)$","\g<1>",direct)
    # get all files present
    logs = glob.glob("./pickles/*"+dat_type)
    logs.remove(directLong)
    # start recursive process
    chron=[directLong]
    found = True
    while found:
        found = False
        for log in logs:
            if prefix in re.sub(r"(.*RGAN_).*","\g<1>",log):
                found = True
                chron.append(log)
                prefix = re.sub(r"(.*_RGAN_)(.*)(_.*)$","\g<2>",log)
                logs.remove(log)
                break
    return chron

def prune_dirs(chron):
    for dr in chron:
        local = glob.glob(dr+"/*csv")
        local_log_file = [fl for fl in local if "log.csv" in fl]
        local_init_file = [fl for fl in local if "init.csv" in fl][0]
        local_init_df = pd.read_csv(local_init_file)
        if "until" in local_init_df.columns:
            continue
        saving_rate = local_init_df.iloc[-1]["saving_rate"]
        intention_epochs = local_init_df["epochs"][0]
        run_epochs = len(glob.glob(dr+"/img/*"))
        if len(local_log_file) == 0:
            # if no log.csv is present, make dummy one
            local_log_df = make_fake_df(run_epochs)
        else:
            # if log.csv is present, remove incomplete epochs
            local_log_file = local_log_file[0]
            local_log_df = pd.read_csv(local_log_file)
            local_log_df = local_log_df[local_log_df.epoch <= run_epochs]
        local_log_df.to_csv(dr+"/log.csv",index=False)
        if run_epochs == intention_epochs:
            # if completed epochs are same as intended epochs, nothing to prune
            continue
        else:
            # if not, find last saved epoch and prune all files to that point
            offset = run_epochs % saving_rate
            last_saved_epochs = run_epochs - offset
            local_log_df = local_log_df[local_log_df.epoch <= last_saved_epochs]
            local_log_df.to_csv(dr+"/log.csv",index=False)
            [os.rename(img,re.sub(".png",".bak",img)) for img in glob.glob(dr+"/img/*") if int(re.sub(r".*epoch([0-9]+)\.png","\g<1>",img)) > last_saved_epochs]

def copy_increment_images(chron,new_direct_long):
    # recursively combine images
    for dr in chron:
        # image copying/combination pipeline
        imgs = glob.glob(dr+"/img/*png")
        src = glob.glob(new_direct_long+"/img/*")
        if len(src) == 0:
            [shutil.copy(img,new_direct_long+"/img/") for img in imgs]
        else:
            max_count = max([int(re.sub(r".*epoch([0-9]+)\.png","\g<1>",img))
                             for img in src])
            [shutil.copyfile(img,new_direct_long+"/img/epoch"+str(int(re.sub(r".*epoch([0-9]+)\.png","\g<1>",img))+max_count)+".png") for img in imgs]

def copy_log_init(chron,new_direct_long):
    for dr in chron:
        local = glob.glob(dr+"/*csv")
        src = glob.glob(new_direct_long+"/*csv")
        if len(src) == 0:
            [shutil.copy(loc,new_direct_long) for loc in local]
            src = glob.glob(new_direct_long+"/*csv")
            # pipeline to merge log.csv
            src_init_file = [fl for fl in src if "init.csv" in fl][0]
            src_init_df = pd.read_csv(src_init_file)
            if "until" not in src_init_df.columns:
                src_log_file = [fl for fl in src if "log.csv" in fl][0]
                src_log_df = pd.read_csv(src_log_file)
                max_epoch = max(src_log_df["epoch"])
                src_init_df["until"] = max_epoch
                src_init_df.to_csv(src_init_file,index=False)
        else:
            # pipeline to merge log.csv
            src_log_file = [fl for fl in src if "log.csv" in fl][0]
            src_log_df = pd.read_csv(src_log_file)
            max_epoch = max(src_log_df["epoch"])
            local_log_file = [fl for fl in local if "log.csv" in fl][0]
            local_log_df = pd.read_csv(local_log_file)
            local_log_df["epoch"] = local_log_df["epoch"] + max_epoch
            # write combined log.csv to file
            src_log_df = pd.concat([src_log_df,local_log_df])
            src_log_df.to_csv(src_log_file,index=False)
            max_epoch = max(src_log_df["epoch"])
            src_init_file = [fl for fl in src if "init.csv" in fl][0]
            src_init_df = pd.read_csv(src_init_file)
            local_init_file = [fl for fl in local if "init.csv" in fl][0]
            local_init_df = pd.read_csv(local_init_file)
            local_init_df["until"] = max_epoch
            pd.concat([src_init_df,local_init_df]).to_csv(src_init_file,index=False)

def make_fake_df(epochs):
    fieldnames = {"epoch":np.arange(1,epochs+1),
                  "batch":np.full(epochs,np.nan),
                  "d_loss":np.full(epochs,np.nan),
                  "d_acc":np.full(epochs,np.nan),
                  "g_loss":np.full(epochs,np.nan),
                  "g_acc":np.full(epochs,np.nan)}
    return pd.DataFrame(fieldnames)

def combineLogs(direct):
    # clean up directory input
    direct = re.sub(r"(\/)?$","",direct)
    direct = re.sub(r"(\.\/)?pickles\/","",direct)
    directLong = "./pickles/" + direct
    if not os.path.isdir(directLong):
        sys.exit(directLong+" does not exist")
    # list related directories chronologically
    chron = iter_temporal_find(direct)
    assert len(chron[-1:]) == 1
    # create new directory to replace old ones
    new_direct = re.sub("(.*)(_R(C)?GAN)(_)(.*)(_.*)$","\g<5>\g<2>\g<6>",chron[-1:][0])
    new_direct_long = "./pickles/"+new_direct
    os.makedirs(new_direct_long,exist_ok=True)
    os.makedirs(new_direct_long+"/img",exist_ok=True)
    # prune existing directories
    prune_dirs(chron)
    # copy and combine images
    copy_increment_images(chron,new_direct_long)
    # copy and combine log's and init's
    copy_log_init(chron,new_direct_long)
    # copy over final weights
    weights = glob.glob(chron[-1:][0]+"/*h5")
    weights.extend(glob.glob(chron[-1:][0]+"/*pickle"))
    [shutil.copy(weight,new_direct_long) for weight in weights]
    # move all processed directories into archive
    [shutil.move(dr,"./pickles/archive/") for dr in chron]

###############################
# main command call
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required name arguments")
    required.add_argument("--log-dir", type=str, required=True,
                        help="base directory within pickles from which to combine recursively forward in time")
    args = parser.parse_args()
    combineLogs(args.log_dir)
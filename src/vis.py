#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# get dependencies
import sys
import re
import os
import PIL
import glob
import imageio
import argparse
import subprocess
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
from pygifsicle import optimize

################################
# define key functions
################################

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def make_plot(direct,number_ticks):
    direct = re.sub(r"(\/)?$","",direct)
    direct = re.sub(r"(\.\/)?pickles\/","",direct)
    directLong = "./pickles/" + direct
    if not os.path.isdir(directLong):
        sys.exit(directLong+" does not exist")
    # make vis directory within log directory
    os.makedirs(directLong+"/vis",exist_ok=True)
    subprocess.call(["Rscript","gg.R","-d",directLong,"-t",str(number_ticks)])

def make_gif(direct,shrink_factor=4,skip_rate=2,
             interval=0.1,until=None,progress_bar=False):
    print("creating training evolution gif")
    # clean up directory input
    direct = re.sub(r"(\/)?$","",direct)
    direct = re.sub(r"(\.\/)?pickles\/","",direct)
    directLong = "./pickles/" + direct
    if not os.path.isdir(directLong):
        sys.exit(directLong+" does not exist")
    # get sorted image list
    sorted_list = sorted_alphanumeric(glob.glob(directLong+"/img/*png"))
    # assume all images are of same size
    size = Image.open(sorted_list[0]).size
    new_size = tuple([int(el/shrink_factor) for el in size])
    if isinstance(until,int):
        sorted_list = sorted_list[:until]
    sorted_list = [Image.open(img).resize(new_size,PIL.Image.ANTIALIAS)
                   for i,img in tqdm(enumerate(sorted_list),total=len(sorted_list))
                   if ((i+1) % skip_rate == 0 or i == 0)]
    kargs = {'duration': interval}
    imageio.mimsave(directLong+"/vis/vis.gif", sorted_list, **kargs)
    optimize(directLong+"/vis/vis.gif",directLong+"/vis/vis.gif")
    if progress_bar:
        print("adding progress bar to gif")
        output = subprocess.call("cat "+directLong+"/vis/vis.gif | gif-progress --bar-color '#000' > "+directLong+"/vis/out.gif",shell=True)
        if output != 0:
            sys.exit("error occurred with gif progress bar, do manual check")

###############################
# main command call
###############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group("required name arguments")
    required.add_argument("--log-dir", type=str, required=True,
                        help="base directory within pickles from which to visualize")
    parser.add_argument("--number-ticks", type=int, default=10,
                        help="number of x-axis ticks to use in main plots")
    parser.add_argument("--create-gif", default=False, action="store_true",
                        help="option to activate gif creation")
    parser.add_argument("--shrink-factor", type=int, default=4,
                        help="shrinking factor for images, applies only when --create-gif is supplied")
    parser.add_argument("--skip-rate", type=int, default=2,
                        help="skip interval when using images to construct gif, applies only when --create-gif is supplied")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="time interval when constructing gifs from images, applies only when --create-gif is supplied")
    parser.add_argument("--until", type=int, default=None,
                        help="set upper epoch limit for gif creation, applies only when --create-gif is supplied")
    parser.add_argument("--progress-bar", default=False, action="store_true",
                        help="option to add progress bar to gifs, applies only when --create-gif is supplied; check readme for additional go package installation instructions")
    args = parser.parse_args()
    # make plot
    make_plot(args.log_dir,args.number_ticks)
    # if necessary, make gif
    if args.create_gif:
        make_gif(args.log_dir,args.shrink_factor,args.skip_rate,
                 args.interval,args.until,args.progress_bar)

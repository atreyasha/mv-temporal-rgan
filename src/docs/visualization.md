## Visualization

Once we have our pruned and combined log directories, we can proceed with plotting some of the training metrics.

```
usage: vis.py [-h] --log-dir LOG_DIR [--number-ticks NUMBER_TICKS]
              [--create-gif] [--shrink-factor SHRINK_FACTOR]
              [--skip-rate SKIP_RATE] [--interval INTERVAL] [--until UNTIL]
              [--progress-bar]

optional arguments:
  -h, --help            show this help message and exit
  --number-ticks NUMBER_TICKS
                        number of x-axis ticks to use in main plots (default:
                        10)
  --create-gif          option to active gif creation (default: False)
  --shrink-factor SHRINK_FACTOR
                        shrinking factor for images, applies only when
                        --create-gif is supplied (default: 4)
  --skip-rate SKIP_RATE
                        skip interval when using images to construct gif,
                        applies only when --create-gif is supplied (default:
                        2)
  --interval INTERVAL   time interval when constructing gifs from images,
                        applies only when --create-gif is supplied (default:
                        0.1)
  --until UNTIL         set upper epoch limit for gif creation, applies only
                        when --create-gif is supplied (default: None)
  --progress-bar        option to add progress bar to gifs, applies only when
                        --create-gif is supplied; check readme for additional
                        go package installation instructions (default: False)

required name arguments:
  --log-dir LOG_DIR     base directory within pickles from which to visualize
                        (default: None)
```

An example of running this script is as follows:

```
$ python3 vis.py --log-dir ./pickles/2019_10_20_19_02_22_RGAN_faces --create-gif --progress-bar
```

This will create resulting loss evolution graphs and a gif with a progress bar showing how constant noise vector image generations evolved with training epochs.

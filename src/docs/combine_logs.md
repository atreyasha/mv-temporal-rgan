## Combination of log directories

Suppose you ran multiple training sessions for a given log directory. As a result of this, you may end up having multiple sequential log directories, such as `2019_10_20_19_02_22_RGAN_faces` and `2019_10_20_19_02_22_RGAN_2019_10_24_13_45_01_faces`. At the end of your training sessions, you can combine these directories into a single directory by using `combine-logs.py`:

```
usage: combine_logs.py [-h] --log-dir LOG_DIR

optional arguments:
  -h, --help         show this help message and exit

required name arguments:
  --log-dir LOG_DIR  base directory within pickles from which to combine
                     recursively forward in time
```

In the above defined example, you could combine both logs by running the following on the `base` or oldest directory:

```
$ python3 combine_logs.py --log-dir ./pickles/2019_10_20_19_02_22_RGAN_faces 
```

This process prunes old directories and combines only the relevant results. The resulting final log directory can then be used for visualization or perhaps even further training. The final combined director will use newest `datetime` string in the form: `(newest_datetime_string)(model)(data)`; which would be `2019_10_24_13_45_01_RGAN_faces` in our previous example. 

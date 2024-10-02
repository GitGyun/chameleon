# DAVIS 2016 evaluation

This is **not** an official script.

Using the [precomputed results](https://davischallenge.org/davis2016/soa_compare.html), the numbers are the same as those on the leaderboard so I think this script is correct. Note that it accepts results in the 0~255 (thresholded at 128) format, not the 0/1 pixel format. 

Example:

```bash
python evaluation_method.py --task semi-supervised --davis_path [path to davis 2017 trainval] --year 2016 --results_path ../mhpvos
```


See also:

https://github.com/davisvideochallenge/davis2017-evaluation

https://github.com/davisvideochallenge/davis2017-evaluation/issues/4
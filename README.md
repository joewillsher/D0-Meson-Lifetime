# Measuring the D0 Meson Lifetime with the LHCb Detector

### About

This work was done for an Imperial College summer term project — work is done in a pair and the [report in this repository was written by Josef Willsher.](D0 Meson Lifetime Report.pdf)

### Abstract

The lifetime of the D0 meson is measured using data from the 2011 run of the LHCb detector. The run collected data with an integrated luminosity 573.019 ± 1.564 fb−1 and centre-of-mass energy 7 TeV. Decays of the Cabbibo-favoured decay mode D0 → K−π+ are observed. The D0 meson is produced at a displaced vertex by the decay of a B meson, allowing better subtraction of background. To remove background cuts are performed on the data, and background subtraction is performed using an un-binned maximum likeli- hood fit; the D0 lifetime is found to be 414.8 ± 0.7stat ± 7.4syst fs, close to the accepted value of 410.1 ± 1.5 fs.

### Flags

- `--full-set`: Cuts.py loads from the large data set
- `--no-plot`: Doesn't run all comparison plot functions in Cuts.py
- `--latex-plot`: Plots all graphs with a half page width and exports a latex compatible pgf format.

### Requires:

- *Python* 3
- Scipy, Numpy, and Matplotlib
- [lazy-property](https://pypi.python.org/pypi/lazy-property)
- [git lfs](https://github.com/git-lfs/git-lfs) for storing data files

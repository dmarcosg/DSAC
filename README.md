# Deep Structured Active Contours (DSAC)

This code allows to train a CNN model to predict a good map of penalizations for the different term of an Active Contour Model (ACM) such that the result gets close to a set of ground truth contours, as presented in [[1]](#marcos2018) (to appear in CVPR 2018). 

A preprint of the paper can be found in https://arxiv.org/pdf/1803.06329.pdf

## Datasets

[Vaihingen buildings](https://drive.google.com/open?id=1nenpWH4BdplSiHdfXs0oYfiA5qL42plB)

[Bing Huts](https://drive.google.com/open?id=1Ta21c3jucWFoe5jwiVXXiAgozvdmnQKP)

## Usage

Download and unzip the datasets. Modify the dataset paths in the main files and run them with Python 3. Requires Tensorflow 1.4.

Please contact me at diego.marcos@wur.nl for questions and feedback. 

<a name="marcos2018"></a>
[1] Marcos, D., Tuia, D., Kellenberger, B., Zhang, L., Bai, M., Liao, R. & Urtasun, R. (2018). Learning deep structured active contours end-to-end. arXiv preprint arXiv:1803.06329.

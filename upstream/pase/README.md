Please follow the installation instruction on [https://github.com/s3prl/pase.git](https://github.com/s3prl/pase.git), which is modified from [https://github.com/santi-pdp/pase.git](https://github.com/santi-pdp/pase.git) to make **QRNN** module compatible with *PyTorch>=1.7.0*, or else the original implementation only supports up to *PyTorch<=1.4.0*. The extracted features from the official released checkpoint [FE_e199.ckpt](https://drive.google.com/file/d/1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW/view) with the original implementation is completely same as the modified one. There is another potential fix to directly adopt the [**QRNN**](https://docs.fast.ai/text.models.qrnn.html) implemented by **fastai**, however we tested it and found the numbers do not match.
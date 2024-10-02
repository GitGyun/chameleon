import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import numpy as np


def dense_crf(img, output_probs):
    """ Conditional Random Field for better segmentation
        Refer to https://github.com/lucasb-eyer/pydensecrf for details. 
    """

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=15)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=img, compat=4)

    Q = d.inference(10)
    Q = np.array(Q).reshape((c, h, w))
    return Q

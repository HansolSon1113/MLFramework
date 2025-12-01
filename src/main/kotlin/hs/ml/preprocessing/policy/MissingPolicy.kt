package hs.ml.preprocessing.policy

import hs.ml.math.Tensor

interface MissingPolicy {
    fun handle(tensor: Tensor): Tensor
}

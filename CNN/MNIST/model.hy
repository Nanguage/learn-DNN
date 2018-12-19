(require [hy.extra.anaphoric [*]])

(import [tensorflow :as tf])
(import [tensorflow.contrib.slim :as slim])


(defn CNN [inputs &optional [is-training True]]
  (setv batch-norm-params
    {
      "is_training" is-training
      "decay" 0.9
      "updates_collections" None
    })
  (with [(slim.arg-scope
           [slim.conv2d slim.fully-connected]
              :normalizer-fn slim.batch-norm
              :normalizer-params batch-norm-params)]
    (-> 
      (tf.reshape inputs [-1 28 28 1])
      (slim.conv2d 32 [5 5] :scope "conv1")
      (slim.max-pool2d [2 2] :scope "pool1")
      (slim.conv2d 64 [5 5] :scope "conv2")
      (slim.max-pool2d [2 2] :scope "pool2")
      (slim.flatten :scope "flatten3")
      (slim.fully-connected 1024 :scope "fc3")
      (slim.dropout :is-training is-training :scope "dropout3")
      (slim.fully-connected 10 :activation-fn None :normalizer-fn None :scope "fco") )))


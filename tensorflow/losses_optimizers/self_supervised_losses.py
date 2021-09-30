import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
import tensorflow_addons as tfa
import numpy as np

######################################################################################
'''Supervised  Contrastive LOSS'''
######################################################################################


def multiclass_npair_loss(z, y):
    '''
    arg: z, hidden feature vectors(B_S[z], n_features)
    y: ground truth of shape (B_S[z])

    '''
    # Cosine similarity matrix
    z = tf.math.l2_normalize(z,  axis=1)
    Similarity = tf.matmul(z, z, transpose_b=True)
    loss = tfa.losses.npairs_loss(y, Similarity)
    return loss

# Supervised Contrastive Learning Paper


def multi_class_npair_loss_temperature(z, y, temperature):
    x_feature = tf.math.l2_normalize(z,  axis=1)
    similarity = tf.divide(
        tf.matmul(x_feature, tf.transpose(x_feature)), temperature)
    return tfa.losses.npairs_loss(y, similarity)


######################################################################################
'''Self-Supervised CONTRASTIVE LOSS'''
######################################################################################

'''N-Pair Loss'''


def multiclass_N_pair_loss(p, z):
    x_i = tf.math.l2_normalize(p, axis=1)
    x_j = tf.math.l2_normalize(z, axis=1)
    similarity = tf.matmul(x_i, x_j, transpose_b=True)
    batch_size = tf.shape(p)[0]
    contrastive_labels = tf.range(batch_size)

    # Simlarilarity treat as logic input for Cross Entropy Loss
    # Why we need the Symmetrized version Here??
    loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, similarity, from_logits=True)
    loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, tf.transpose(similarity), from_logits=True)

    return (loss_1_2+loss_2_1)/2


'''SimCLR Paper Nt-Xent Loss # ASYMETRIC Loss'''
# Nt-Xent ---> N_Pair loss with Temperature scale
# Nt-Xent Loss (Remember in this case dataset two image stacked)


def nt_xent_asymetrize_loss(z,  temperature):
    '''The issue of design this loss two image is in one array
    when we multiply them that will lead two two same things mul together???

    '''
    # Feeding data (ALready stack two version Augmented Image)[2*bs, 128]
    z = tf.math.l2_normalize(z, axis=1)

    similarity_matrix = tf.matmul(
        z, z, transpose_b=True)  # pairwise similarity
    similarity = tf.exp(similarity_matrix / temperature)

    ij_indices = tf.reshape(tf.range(z.shape[0]), shape=[-1, 2])
    ji_indices = tf.reverse(ij_indices, axis=[1])

    #[[0, 1], [1, 0], [2, 3], [3, 2], ...]
    positive_indices = tf.reshape(tf.concat(
        [ij_indices, ji_indices], axis=1), shape=[-1, 2])  # Indice positive pair
    # --> Output N-D array
    numerator = tf.gather_nd(similarity, positive_indices)
    # 2N-1 (sample)
    # mask that discards self-similarity
    negative_mask = 1 - tf.eye(z.shape[0])

    # compute sume across dimensions of Tensor (Axis is important in this case)
    # None sum all element scalar, 0 sum all the row, 1 sum all column -->1D metric
    denominators = tf.reduce_sum(
        tf.multiply(negative_mask, similarity), axis=1)

    losses = -tf.math.log(numerator/denominators)
    return tf.reduce_mean(losses)


'''SimCLR paper Asytemrize_loss V2'''

# Mask to remove the positive example from the rest of Negative Example

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images
    # Ensure distinct pair of image get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i+batch_size] = 0

    return tf.constant(negative_mask)


consie_sim_1d = tf.keras.losses.CosineSimilarity(
    axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(
    axis=2, reduction=tf.keras.losses.Reduction.NONE)


def nt_xent_asymetrize_loss_v2(p, z,  temperature, negative_mask):  # negative_mask
    # L2 Norm

    p_l2 = tf.math.l2_normalize(p, axis=1)
    z_l2 = tf.math.l2_normalize(z, axis=1)
    similarity = tf.matmul(tf.expand_dims(p_l2, 1), tf.expand_dims(z_l2, 2))
    batch_size = tf.shape(p, out_type=tf.dtypes.int32)[0]
    print(batch_size)
    similarity = (tf.reshape(similarity, (batch_size, 1)))/temperature
    negatives = tf.concat([p_l2, z_l2], axis=0)
    #negative_mask = get_negative_mask(batch_size)

    loss = 0
    for positives in [p_l2, z_l2]:
        l_negative = tf.tensordot(tf.expand_dims(
            positives, 1), tf.expand_dims(tf.transpose(negatives), 0), axes=2)
        l_labels = tf.zeros(batch_size, dtype=tf.int32)
        l_neg = tf.boolean_mask(l_negative, negative_mask)
        l_neg = tf.reshape(l_neg, (batch_size, -1))/temperature
        logits = tf.concat([similarity, l_neg], axis=1)
        loss += tf.keras.losses.SparseCategoricalCrossentropy(y_pred=logits, y_true=l_labels,
                                                              from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    loss = loss/(2*batch_size)
    return loss

'''SimCLR Paper Nt-Xent Loss # SYMMETRIZED Loss'''
# Nt-Xent Loss Symmetrized

def nt_xent_symmetrize_keras(p, z, temperature):
    # cosine similarity the dot product of p,z two feature vectors
    x_i = tf.math.l2_normalize(p, axis=1)
    x_j = tf.math.l2_normalize(z, axis=1)
    similarity = (tf.matmul(x_i, x_j, transpose_b=True)/temperature)
    # the similarity from the same pair should be higher than other views
    batch_size = tf.shape(p)[0]  # Number Image within batch
    contrastive_labels = tf.range(batch_size)

    # Simlarilarity treat as logic input for Cross Entropy Loss
    # Why we need the Symmetrized version Here??
    loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, similarity, from_logits=True,)  # reduction=tf.keras.losses.Reduction.SUM
    loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, tf.transpose(similarity), from_logits=True, )
    return (loss_1_2 + loss_2_1) / 2


######################################################################################
'''NONE CONTRASTIVE LOSS'''
####################################################################################

'''BYOL SYMETRIZE LOSS'''
# Symetric LOSS

def byol_symetrize_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)

'''Loss 2 SimSiam Model'''
# Asymetric LOSS


def simsam_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


def simsam_loss_non_stop_Gr(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    #z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

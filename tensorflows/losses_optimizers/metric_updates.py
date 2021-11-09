import tensorflow as tf

# ***************************************************************
# Supervised Finetuning Loss and Accuracy Metric Update
# ***************************************************************


def update_pretrain_metrics_train(contrast_loss, contrast_acc, contrast_entropy, loss, logits_con, labels_con):
    '''
    Args:
        contrast_loss: is the contrastive loss
        contrast_acc: is the accurate predic similar base comparing Negative
        contrast_entropy: is the result Log probability of logits
    Return:
        Update loss (Contrast, contrast_acc, contrast_entropy)
    '''
    contrast_loss.update_state(loss)
    # The label here mean the (We knew the two Augmented pair image is the SAME)
    contrast_acc_val = tf.equal(
        tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
    contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
    contrast_acc.update_state(contrast_acc_val)

    prob_con = tf.nn.softmax(logits_con)
    entropy_con = - \
        tf.reduce_mean(tf.reduce_sum(
            prob_con * tf.math.log(prob_con + 1e-8), -1))
    contrast_entropy.update_state(entropy_con)


def update_pretrain_metrics_eval(contrast_loss_metric,
                                 contrastive_top_1_accuracy_metric,
                                 contrastive_top_5_accuracy_metric,
                                 contrast_loss, logits_con, labels_con):
    '''.

    Args:
    contrast_top1_: predicted of top 1 the similarity (Probability)
    contrast_top5_: Predicted of top 5 Similiartiy (Probabilty)

    '''

    contrast_loss_metric.update_state(contrast_loss)
    contrastive_top_1_accuracy_metric.update_state(
        tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
    contrastive_top_5_accuracy_metric.update_state(labels_con, logits_con)


# ***************************************************************
# Supervised Finetuning Loss and Accuracy Metric Update
# ***************************************************************

def update_finetune_metrics_train(supervised_loss_metric, supervised_acc_metric,
                                  loss, labels, logits):
    supervised_loss_metric.update_state(loss)

    label_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, axis=1))
    label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
    supervised_acc_metric.update_state(label_acc)


def update_finetune_metrics_eval(label_top_1_accuracy_metrics,
                                 label_top_5_accuracy_metrics, outputs, labels):
  label_top_1_accuracy_metrics.update_state(
      tf.argmax(labels, 1), tf.argmax(outputs, axis=1))
  label_top_5_accuracy_metrics.update_state(labels, outputs)

# Helper function convert all metric to Float values


def _float_metric_value(metric):
    """Gets the value of a float-value keras metric."""
    return metric.result().numpy().astype(float)


def log_and_write_metrics_to_summary(all_metrics, global_step):
      for metric in all_metrics:
    metric_value = _float_metric_value(metric)
    logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
    tf.summary.scalar(metric.name, metric_value, step=global_step)

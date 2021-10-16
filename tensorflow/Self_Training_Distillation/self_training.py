'''
'''

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from ..Resnet_architecture_v2 import BuildResnet
from ..datasets import self_distillation_dataset
import argparse
from ..callbacks_list import callback_func

# Setting GP

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[0:4], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()


###############################################################################################
'''Implementation Multiple Improvement of Training Neural Net

1. Training Model with Strong DataAugmentation Methods
    + RandAugment , Hill_Climbing_Aug, Mixup, CutMix, (Consistent Supervision_training)

2. Tranining Model with Strong Regularizers
    + Manifold_Mixup, l2_regularizer, Weight Standardization, Gradient Centralization, LearningRate_Decay

3. Learning to Resize Image also help in Training Optimization 
    + learning to REesize
'''
###############################################################################################

AUTO = tf.data.experimental.AUTOTUNE

IMG_SIZE = 32
IMG_RESIZE = 96
BATCH_SIZE = 100

###############################################################################################
'''CONV_Net ----- Teacher ----- Student Model '''
###############################################################################################
# Backbone Architecture from Keras


def conv_keras_teacher_model():
    raise Exception("Model Not Yet Implement")


def conv_keras_teacher_model():
    raise Exception("Model not Yet Implement")


# Backbone Architecture RESNET-Architecture
def conv_teacher_resnet_custome(model_name, width_scale, num_class):
    resnet = BuildResnet(name=model_name, width_scale=width_scale,
                         num_class=num_class, include_top=True)
    resnet(tf.keras.Input(IMG_SIZE, IMG_SIZE, 3))
    return resnet


def conv_student_resnet_custome(model_name, width_scale, num_class):
    resnet = BuildResnet(name=model_name, width_scale=width_scale,
                         num_class=num_class, include_top=True)
    resnet(tf.keras.Input(IMG_SIZE, IMG_SIZE, 3))
    return resnet

# Backbone WRN Architecture


# Backbone ResNetxt Architecture


# Backbone Regnet Architecture


###############################################################################################
'''CONV_NET_Transformer ----- Teacher ----- Student Model '''
###############################################################################################
# Implementation Compact Convolution an Transformer Model
# https://keras.io/examples/vision/cct/

# Backbone Architecture from Keras


def cct_keras_teacher_model():
    raise NotImplementedError


def cct_keras_teacher_model():
    raise NotImplementedError


###############################################################################################
'''Transformer Architecture [Perciver-- AxialAttention Transformer] ----- Teacher ----- Student Model '''
###############################################################################################

# Backbone Architecture from Keras


def vit_keras_teacher_model():
    raise NotImplementedError


def vit_keras_teacher_model():
    raise NotImplementedError


def perciver_architecture_teacher():
    raise NotImplementedError


def perciver_architecture_student():
    raise NotImplementedError


###############################################################################################
# PART22222222222 ''' Data Augmentation -- Regularizer Distillation Implemment'''
###############################################################################################
dataset = self_distillation_dataset(BATCH_SIZE, IMG_SIZE, different_size=False)


#######################################
'''1.. Implementation Mixup Processing dataset Training'''
#######################################

alpha_mixup = 0.4


def mixup():
    train_ds, test_ds = dataset.mixup_dataset(alpha_mixup)
    return train_ds, test_ds


#######################################
'''2..Implementation RandAug Processing dataset Training'''
#######################################

RandAug = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}


def randaug_single():
    train_ds, test_ds = dataset.RandAugment_single(
        RandAug)
    return train_ds, test_ds


Student_RandAug = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}
Teacher_RandAug = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}


def randaug_double():
    train_ds, test_ds = dataset.RandAugment_two_trasforms(
        Student_RandAug, Teacher_RandAug)
    return train_ds, test_ds


#######################################
'''3...Implementation RandAug_Mixup'''
#######################################

alpha_Randaug_mixup = 0.4

Student_RandAug_mixup = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}
Teacher_RandAug_mixup = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}


def randaug_mixup_double():
    train_ds, test_ds = dataset.Randaug_Mixup_two_transforms(
        alpha_Randaug_mixup, Student_RandAug_mixup, Teacher_RandAug_mixup)
    return train_ds, test_ds


alpha_Randaug_mixup_one = 0.4

RandAug_mixup = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}


def randaug_mixup_single():
    train_ds, test_ds = dataset.Randaug_Mixup_single_transform(
        alpha_Randaug_mixup_one, RandAug_mixup)
    return train_ds, test_ds


#######################################
'''4...Implementation MixupRand'''
#######################################


alpha_Randaug_mixup_double = 0.4

Student_RandAug_mixup = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}
Teacher_RandAug_mixup = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}


def mixup_randaug_double():
    train_ds, test_ds = dataset.Mixup_Randaug_two_transform(
        alpha_Randaug_mixup_double, Student_RandAug_mixup, Teacher_RandAug_mixup)
    return train_ds, test_ds


alpha_mixup_Randaug = 0.4
RandAug = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}


def mixup_randaug_single():
    train_ds, test_ds = dataset.Mixup_Randaug_single_transform(
        alpha_mixup_Randaug, RandAug)

    return train_ds, test_ds


#######################################
'''5...Implementation CutMix '''
#######################################

alpha_cutmix = [0.4]


def cutmix():
    train_ds, test_ds = dataset.cutmix_dataset(alpha_cutmix)
    return train_ds, test_ds


#######################################
'''6...Implementation RandAug CutMix'''
#######################################

alpha_cutmix_Randaug = 0.4

RandAug_cutmix = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}


def rand_cutmix_single():

    train_ds, test_ds = dataset.Cutmix_Randaug_single_transform(
        alpha_cutmix_Randaug,  RandAug_cutmix)

    return train_ds, test_ds


alpha_cutmix_Rancdaug_double = 0.4
RandAug_cutmix_Stu = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}

RandAug_cutmix_Te = {
    "number_transform": 2,  # [0--14]
    "magnitude_transform": 7,  # [0--30]
}


def rand_cutmix_double():
    train_ds, test_ds = dataset.Cutmix_Randaug_two_transform(
        alpha_cutmix_Randaug,  RandAug_cutmix_Stu, RandAug_cutmix_Te)
    return train_ds, test_ds


###############################################################################################
# PART3333 ''' Implement KNOWLEDGE DISTILLATION'''
###############################################################################################
''''
The Implementation will base on Two parts.

P1. Distillation (Teacher and Student model will the same INPUT)

P2. Self supervision (Teacher and Student model will have two different INPUT)

'''


class conv_distillation(tf.keras.Model):
    def __init__(self, student_model, teacher_model):
        super(conv_distillation, self).__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        return metrics

    def compile(self, optimizer, metrics, distillation_loss_fn, temperature=3):

        super(conv_distillation, self).compile(
            optimizer=optimizer, metrics=metrics
        )

        self.temperature = temperature
        # self.student_loss = student_loss_fn
        self.distillation_loss = distillation_loss_fn

    def train_step(self, dataset):
        teacher_ds, student_ds = dataset
        # Forward through teacher network get sudo prediction
        teacher_predict = self.teacher(teacher_ds, training=False)

        with tf.GradientTape() as tape():
            # Forward to student -- listen to what teacher say
            student_predict = self.student(student_ds, training=True)
            # Compute the loss improve Student Model
            distillation_loss = self.distillation_loss(tf.nn.softmax(teacher_predict / self.temperature, axis=1),
                                                       tf.nn.softmax(student_predict / self.temperature, axis=1))

            # Compute the Gradient band backward
            trainable_var_update = self.student.trainable_variables
            gradient = tape.gradient(
                distillation_loss, trainable_var_update)
            # Update the weights
            self.optimizer.apply_gradients(zip(gradient, trainable_var_update))
            # Update loss state
            self.loss_tracker.update_state(distillation_loss)

            return {"Distillation_loss: ", {self.loss_track.result()}}

    def test_step(self, dataset):

        # Unpack data
        x, y = dataset

        # Forward passes
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)

        # Calculate the loss
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(student_predictions / self.temperature, axis=1),
        )

        # Report progress
        self.loss_tracker.update_state(distillation_loss)
        # for Accuracy metrics -- need Y labels
        self.compiled_metrics.update_state(y, student_predictions)
        # append the result in the metrics we use
        # 2 metrics implement
        # keras.metrics.SparseCategoricalAccuracy()
        # metric loss tracker from mean()
        results = {m.name: m.result() for m in self.metrics}
        return results


class conv_consistency_semantic_distillation(tf.keras.Model):
    def __init__(self, student_model, teacher_model):
        super(conv_distillation, self).__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.student_loss_tracker = tf.keras.metrics.Mean(name="Stud_loss")
        self.loss_tracker = tf.keras.metrics.Mean(name="Distill_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="Total_loss")

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        metrics.append(self.student_loss_tracker)
        metrics.append(self.total_loss_tracker)

        return metrics

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, temperature=3):

        super(conv_distillation, self).compile(
            optimizer=optimizer, metrics=metrics
        )

        self.temperature = temperature
        self.student_loss = student_loss_fn
        self.distillation_loss = distillation_loss_fn

    def train_step(self, dataset):
        teacher_ds, student_ds = dataset
        # Forward through teacher network get sudo prediction
        # Setting Threshold [FitMatch AdapMatch Paper Here]
        teacher_predict = self.teacher(teacher_ds, training=False)

        with tf.GradientTape() as tape():
            # Forward to student -- listen to what teacher say
            student_predict = self.student(student_ds, training=True)
            # Compute student Loss From different view dataset Feed to Student MODEL
            # Here also base on Psudo lable from Teacher Model
            student_loss = self.student_loss(student_predict, teacher_predict)
            # Compute the loss improve Student Model
            distillation_loss = self.distillation_loss(tf.nn.softmax(teacher_predict / self.temperature, axis=1),
                                                       tf.nn.softmax(student_predict / self.temperature, axis=1))
            total_loss = (student_loss + distillation_loss)/2

        # Compute the Gradient band backward
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_track.update_state(distillation_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {"Distill_loss: ", {self.loss_track.result()}, "Stud_loss: ", {self.student_loss_tracker.update_state.result()},
                "Total_loss: ", {self.total_loss_tracker.update_state.result()}}

    def test_step(self, dataset):

        # Unpack data
        x, y = dataset

        # Forward passes
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)

        # Calculate the loss
        student_loss = (y, tf.nn.softmax(student_predictions, axis=1))
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(student_predictions / self.temperature, axis=1),
        )
        total_loss = student_loss + distillation_loss

        # Report progress
        self.loss_tracker.update_state(distillation_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        # for Accuracy metrics -- need Y labels
        self.compiled_metrics.update_state(y, student_predictions)
        # append the result in the metrics we use
        # 2 metrics implement
        # keras.metrics.SparseCategoricalAccuracy()
        # metric loss tracker from mean()
        results = {m.name: m.result() for m in self.metrics}
        return results


with strategy.scope():

    def main(args):

        EPOCHS = args.epochs
        BATCH_SIZE = args.batch_size
        # Dataset training
        train_ds, test_ds = mixup(BATCH_SIZE)
        # model training
        student_model = conv_student_resnet_custome(
            model_name, width_scale, num_class)
        teacher_model = conv_teacher_resnet_custome(
            model_name, width_scale, num_class)

        # Model Optimizer Configure

        # Model compiling
        add_kwargs = {
            "EPOCHS": EPOCHS,
            "monitor": "total_loss",  # Tracking Contrastive Loss
            "patience_stop": 50,  # epochs
            "reducelr_patience": 10,  # epochs
            "min_lr": 1e-7,
            "checkpoint_period": 2,
            # original "B0_Mixup_update.h5", 99.2 test--prune_check_v 99.6/ 99.7
            "checkpoint_name": "./h5",
            # result 28-3 almost the same 28-2
            "log_path": "./",
            # [checkpoint, earlystop,reducelr, lr_cosine_annealing ,schedule_lr, tensorboard, lr_cosine_annealing2]
            "callback_list": [0,  5],
        }
        # '''Note model during experiment

        callbacks_list, name_list = callback_func(add_kwargs, pruning=False,)
        print(name_list)

        opt = None
        Metrics = None
        temperature = None
        distill_loss = None
        student_loss = None

        if args.distillation == "distillation":
            print("You implement Distillation")
            distill_model = conv_distillation(student_model, teacher_model)
            distill_model.compile(optimizer=opt, metrics=Metrics,
                                  distillation_loss_fn=distill_loss, temperature=temperature)
            distill_model.fit(train_ds, validation_data=test_ds,
                              epochs=EPOCHS, callbacks=callbacks_list)
            distill_model.student.save_weights(args.path_distill)

        elif args.distillation == "distillation_consistency_semantic":
            print("You implement distillation_consistency_semantic")
            distill_model = conv_consistency_semantic_distillation(
                student_model, teacher_model)
            distill_model.compile(optimizer=opt, metrics=Metrics, student_loss_fn=student_loss,
                                  distillation_loss_fn=distill_loss, temperature=temperature)
            distill_model.fit(train_ds, validation_data=test_ds,
                              epochs=EPOCHS, callbacks=callbacks_list)
            distill_model.student.save_weights(args.path_distill_semantic)

        else:
            raise Exception("You not implement existing distillation type")

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument('--encoder', type=str, required=True, choices=[
                            'resnet18', 'resnet34', 'resnet50', 'resnet101'], help='Encoder architecture')
        parser.add_argument('--num_epochs', type=int,
                            default=2000, help='Number of epochs')
        parser.add_argument('--batch_size', type=int,
                            default=608, help='Batch size for pretraining')
        parser.add_argument('--temperature', type=float,
                            default=10, help='Temperature distillation')
        parser.add_argument('--path_distill', type=str,
                            default='./Distill_weights/student_distillation_weight.h5', help='distill_saving_weight_path')

        parser.add_argument('--path_distill_semantic', type=str,
                            default='./Distill_weights/student_distill_consistency_semantic_weight.h5', help='distill_semantic_saving_weight_path')

        args = parser.parse_args()
        main(args)

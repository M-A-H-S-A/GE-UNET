import sys
sys.path.insert(0, './')

import src.grape as grape
import src.algorithms as algorithms
from src.functions import add, sub, mul, pdiv, neg, and_, or_, not_, less_than_or_equal, greater_than_or_equal
from tensorflow import keras
#from keras import backend as K
import os
import numpy as np
import configparser
import collections
import tensorflow as tf
from deap import base, creator, tools
from keras.models import Model
from keras.layers import Input, concatenate, UpSampling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda, Conv1D, Layer, MaxPooling2D, AveragePooling2D, BatchNormalization, add, Conv2DTranspose

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
import ast
import gc
import time
import datetime
start_time = time.time()
print ("Start: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
import cv2
from skimage.io import imread
from keras.utils import io_utils
from tensorflow.python.platform import tf_logging as logging
import random



# ====================Data========================

# ====================DropBlock2D========================

class DropBlock2D(keras.layers.Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.supports_masking = True
        self.height = self.width = self.ones = self.zeros = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            self.height, self.width = input_shape[2], input_shape[3]
        else:
            self.height, self.width = input_shape[1], input_shape[2]
        self.ones = K.ones((self.height, self.width), name='ones')
        self.zeros = K.zeros((self.height, self.width), name='zeros')
        super().build(input_shape)

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self):
        """Get the number of activation units to drop"""
        height, width = K.cast(self.height, K.floatx()), K.cast(self.width, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self):
        positions = K.concatenate([
            K.expand_dims(K.tile(K.expand_dims(K.arange(self.height), axis=1), [1, self.width]), axis=-1),
            K.expand_dims(K.tile(K.expand_dims(K.arange(self.width), axis=0), [self.height, 1]), axis=-1),
        ], axis=-1)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < self.height - half_block_size,
                        positions[:, :, 1] < self.width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            self.ones,
            self.zeros,
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        mask = K.random_binomial(shape, p=self._get_gamma())
        mask *= self._compute_valid_seed_region()
        mask = keras.layers.MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask *\
                (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)



# ====================EarlyStopping========================


class EarlyStopping(tf.keras.callbacks.Callback):

    def __init__(
        self,
        threshold,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.threshold=threshold

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, " "fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    io_utils.print_msg(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{self.best_epoch + 1}."
                    )
                self.model.set_weights(self.best_weights)

        val_acc=logs["val_accuracy"]
        if val_acc < self.threshold:
            self.model.stop_training=True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)



custom_early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    min_delta=0.01,
    mode='max',
    threshold=0.85
)
# ====================SAUNet========================

def spatial_attention(input_feature):
    kernel_size=7

    if K.image_data_format() == "channels_first":
        channel=input_feature.shape[1]
        cbam_feature=Permute((2, 3, 1))(input_feature)
    else:
        channel=input_feature.shape[-1]
        cbam_feature=input_feature

    avg_pool=Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool=Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat=Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2

    cbam_feature=Conv2D(1, (7, 7),
                        strides=1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature=Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def upsample_conv(filters, kernel_size, strides, padding,drop_type):
    if (drop_type == Dropout):
        drop_type = Dropout
    else:
        drop_type = DropBlock2D
    for x in reversed(down_layers):
        filters//=2
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = drop_type()(x)
    return x




def conv2d_block(filters,kernel_size,activation,padding,pooling_type,drop_type,network_depth):
    down_layers=[]
    if (pooling_type == 1):
        pooling_2d=MaxPooling2D
    else:
        pooling_2d=AveragePooling2D
    if (drop_type == Dropout):
        drop_type = Dropout
    else:
        drop_type = DropBlock2D
    for l in range(network_depth):
        filters=filters,
        x=Conv2D(filters, kernel_size, activation, padding)(x)
        down_layers.append(x)
        x=BatchNormalization()(x)
        x=pooling_2d((2, 2), padding='same', data_format=None)(x)
        x=drop_type()(x)
        filters=filters * 2

    return x




problem = 'SA-UNET'

if problem == 'SA-UNET':

    BNF_GRAMMAR = grape.Grammar(r"grammars/UNET3.bnf")

    X_train = np.random.randint = [[random.random() for e in range(1)] for e in range(100)]
    Y_train = np.random.randint = [[random.random() for e in range(1)] for e in range(100)]
    X_validate = np.random.randint = [[random.random() for e in range(1)] for e in range(100)]
    Y_validate = np.random.randint = [[random.random() for e in range(1)] for e in range(100)]

input_size=(592,592,3)


def eval_model_loss_function(individual, points_train):
    if individual.invalid:
        return np.NAN,
    start_model_time = time.time()
    print ("Start-model: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f'GA------------Individual = {individual.phenotype}')
    #create function UNet, this is the UNet creted by GE
    exec(eval(individual.phenotype), globals())
    print(eval(individual.phenotype))
    try:
        model=UNET(input_size=(592, 592, 3))
        history=model.fit(x_train, y_train, epochs=20, batch_size=4, verbose=2, validation_data=(x_validate, y_validate),
                          shuffle=True)
    except(ValueError):
        return np.NAN,

    fitness1=max(history.history['val_accuracy'])
    print(f'Myresult')
    print(fitness1)
    print ("End-model: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("---The End %s seconds ---" % (time.time() - start_model_time))
    del history
    del model
    gc.collect()
    return fitness1,



toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("Fitness", base.Fitness, weights=(1.0,))

creator.create('Individual', grape.Individual, fitness=creator.Fitness)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
#toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
#toolbox.register("populationCreator", grape.PI_Grow, creator.Individual)

toolbox.register("evaluate", eval_model_loss_function)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=6)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 20
MAX_GENERATIONS = 20
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = round(0.01*POPULATION_SIZE)

INIT_GENOME_LENGTH = 30 #used only for random initialisation
random_initilisation = False #put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 10
MIN_INIT_TREE_DEPTH = 10
MAX_TREE_DEPTH = 90
MAX_WRAPS = 0
CODON_SIZE = 255

N_RUNS = 3

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i+1)
    print()

    # create initial population (generation 0):
    if random_initilisation:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           init_genome_length=INIT_GENOME_LENGTH,
                                           max_init_depth=MAX_TREE_DEPTH,
                                           codon_size=CODON_SIZE
                                           )
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           codon_consumption='eager'
                                            )

    # define the hall-of-fame object:
#    hof = tools.HallOfFame(ELITE_SIZE)

    # prepare the statistics object:
    hof=tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR, codon_size=CODON_SIZE,
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              points_train=[X_train, Y_train],
#                                              points_test=[X_test, Y_test],
                                              stats=stats, halloffame=hof, verbose=False)

    import textwrap
    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
#    print("Test Fitness: ", fitness_eval(hof.items[0], [X_,Y_test])[0])
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')

    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")

    selection_time = logbook.select("selection_time")
    generation_time = logbook.select("generation_time")
    gen, invalid = logbook.select("gen", "invalid")
    avg_used_codons = logbook.select("avg_used_codons")
    best_ind_used_codons = logbook.select("best_ind_used_codons")

    fitness_test = logbook.select("fitness_test")

    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    structural_diversity = logbook.select("structural_diversity")

    import csv
    import random
    r = random.randint(1,1e10)

    header = ['gen', 'invalid', 'avg', 'std', 'min', 'max', 'fitness_test',
              'best_ind_length', 'avg_length',
              'best_ind_nodes', 'avg_nodes',
              'best_ind_depth', 'avg_depth',
              'avg_used_codons', 'best_ind_used_codons',
              'structural_diversity',
              'selection_time', 'generation_time']
    with open("results/" + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value],
                             fitness_test[value],
                             best_ind_length[value],
                             avg_length[value],
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value],
                             structural_diversity[value],
                             selection_time[value],
                             generation_time[value]])

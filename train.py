import os
from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras import optimizers
from keras_lr_multiplier import LRMultiplier
from datetime import datetime
from generator import data_generator_wrapper
from model import *

class ParallelModelCheckpoint(Callback):
    def __init__(self, callback, model):
        super(ParallelModelCheckpoint, self).__init__()
        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)

def step_decay(epoch, lr):
    if (epoch+1) % 30 == 0:
        lr = lr*0.1
    return lr

if __name__ == '__main__':
    gpus = '4,5'
    save_weights_path = 'net_model/cub200'
    num_cls = 200
    batch_size = 8
    initial_learning_rate = 8e-4
    dataset_dir = '/mnt/sde/clf8113/datasets/CUB_200_2011'

    num_gpu = len(gpus.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    save_weights_path = os.path.join(save_weights_path, datetime.now().strftime('%Y%m%d_%H%M%S'))
    batch_size *= num_gpu

    train_data = open(os.path.join(dataset_dir, 'train.txt')).readlines()
    val_data = open(os.path.join(dataset_dir, 'test.txt')).readlines()

    # train_generator = data_generator_wrapper(train_data, batch_size, num_cls, is_train=True)
    # val_generator = data_generator_wrapper(val_data, batch_size, num_cls, is_train=False)

    model = create_model(num_classes=num_cls)
    model.summary()
    checkpoint = ModelCheckpoint(os.path.join(save_weights_path, 'trained_weights_{epoch:03d}.h5'), verbose=1,
                                 monitor='val_cls_loss', mode='auto', save_weights_only=True, save_best_only=True, period=1)
    tensorboard = TensorBoard(log_dir=save_weights_path)
    logs = CSVLogger(filename=os.path.join(save_weights_path, 'training.log'))
    reduce_lr = ReduceLROnPlateau(monitor='val_cls_loss', mode='auto', factor=0.1, patience=10, verbose=1)
    # reduce_lr = LearningRateScheduler(schedule=step_decay, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_cls_loss', min_delta=0, patience=5, verbose=1)

    if num_gpu > 1:
        training_model = multi_gpu_model(model, gpus=num_gpu)
        checkpoint = ParallelModelCheckpoint(checkpoint, model)
    else:
        training_model = model

    optimizer = optimizers.SGD(lr=initial_learning_rate, momentum=0.9)
    # optimizer = optimizers.RMSprop(lr=initial_learning_rate)
    # optimizer = optimizers.Adam(lr=initial_learning_rate)
    train_generator = data_generator_wrapper(train_data, batch_size, num_cls, is_train=True)
    val_generator = data_generator_wrapper(val_data, batch_size, num_cls, is_train=False)
    training_model.compile(optimizer=LRMultiplier(optimizer, multipliers={'mask':10., 'cls':10., 'adv':10.}),
                           loss={'cls': celoss,
                                 'adv': celoss,
                                 'loc': l1loss},
                           metrics={'cls': unswap_acc},
                           loss_weights={'cls': 1.,
                                         'adv': 1.,
                                         'loc': 1.})
    training_model.fit_generator(generator=train_generator, steps_per_epoch=len(train_data) // batch_size,
                                 initial_epoch=0, epochs=500, verbose=1, callbacks=[checkpoint, tensorboard, logs, reduce_lr],
                                 validation_data=val_generator, validation_steps=len(val_data) // batch_size,
                                 use_multiprocessing=True, workers=8, max_queue_size=16)
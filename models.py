import tensorflow.keras as keras
from pyts.classification import KNeighborsClassifier
import numpy as np

# Needed for LSTMFCN
import keras
from keras import backend as K
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
#from keras.layers import Input, Dense, LSTM, CuDNNLSTM, concatenate, Activation, GRU, SimpleRNN
from keras.layers import Input, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from keras.layers import Permute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier

def make_CNN_model(input_shape, num_classes=2):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def make_LSTMFCN_model(MAX_SEQUENCE_LENGTH, NB_CLASS=2, NUM_CELLS=8):
    ip = Input(shape=(MAX_SEQUENCE_LENGTH, 1))

    x = LSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model

def train_LSTMFCN_model(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=64,
						val_subset=None, cutoff=None, normalize_timeseries=False, learning_rate=1e-3):

	classes = np.unique(y_train)
	le = LabelEncoder()
	y_ind = le.fit_transform(y_train.ravel())
	recip_freq = len(y_train) / (len(le.classes_) *
	                             np.bincount(y_ind).astype(np.float64))
	class_weight = recip_freq[le.transform(classes)]

	y_train = to_categorical(y_train, len(np.unique(y_train)))
	y_test = to_categorical(y_test, len(np.unique(y_test)))

	factor = 1. / np.cbrt(2)

	"""model_checkpoint = ModelCheckpoint("./weights/%s_weights.h5" % dataset_prefix, verbose=1,
	                                   monitor='loss', save_best_only=True, save_weights_only=True)"""
	reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
	                              factor=factor, cooldown=0, min_lr=1e-4, verbose=2)

	callback_list = [reduce_lr]

	optm = Adam(lr=learning_rate)

	model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

	if val_subset is not None:
	    x_test = x_test[:val_subset]
	    y_test = y_test[:val_subset]

	classy_weight = {i : class_weight[i] for i in range(len(class_weight))}
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
	          class_weight=classy_weight, verbose=2, validation_data=(x_test, y_test))

	return model

def train_KNN_model(x_train, y_train):
	model = KNeighborsClassifier(metric='dtw')
	model.fit(x_train.reshape(x_train.shape[0],x_train.shape[1]), y_train)
	return model

def train_CNN_model(model, x_train, y_train, epochs=100, batch_size=64, num_classes=2):
	callbacks = [
	    keras.callbacks.ModelCheckpoint(
	        "best_model.h5", save_best_only=True, monitor="val_loss"
	    )
	]
	model.compile(
	    optimizer="adam",
	    loss="sparse_categorical_crossentropy",
	    metrics=["sparse_categorical_accuracy"],
	)
	history = model.fit(
	    x_train,
	    y_train,
	    batch_size=batch_size,
	    epochs=epochs,
	    callbacks=callbacks,
	    validation_split=0.2,
	    verbose=1,
	)
	return model, history

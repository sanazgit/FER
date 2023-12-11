from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from wsgiref import validate
import tensorflow as tf
tf.random.set_seed(42)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
import numpy as np
np.random.seed(42)

# Define a function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    # Compute the confusion matrix
    cm = tf.math.confusion_matrix(y_true, y_pred)
    # Convert the confusion matrix to a numpy array
    cm = cm.numpy()
    # Normalize the confusion matrix by row
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot the heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    # Add a color bar
    ax.figure.colorbar(im, ax=ax)
    # Set the tick labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over the data dimensions and create text annotations
    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    # Return the figure
    return fig
    
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, test_labels, class_names):
        super(ConfusionMatrixCallback, self).__init__()
        self.test_data = test_data
        self.test_labels = test_labels
        self.class_names = class_names
        self.best_accuracy = 0  # Initialize the best accuracy

    def on_epoch_end(self, epoch, logs=None):
        # Get the current accuracy from the logs
        current_accuracy = logs.get('val_accuracy')

        # If the current accuracy is better than the best so far, save the confusion matrix
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy  # Update the best accuracy

            # Predict the outputs
            y_pred = self.model.predict(self.test_data)
            y_pred = np.argmax(y_pred, axis=1)

            # Convert one-hot encoded targets to class indices
            y_true = np.argmax(self.test_labels, axis=1) if len(self.test_labels.shape) > 1 else self.test_labels

            # Plot and save the confusion matrix
            fig = plot_confusion_matrix(y_true, y_pred, self.class_names)
            plt.show()  # Display the figure in the Jupyter notebook
            fig.savefig(f"best_confusion_matrix.png")  # Save the figure
            plt.close(fig)  # Close the figure to free



input_shape = (128*4, 49) # 64 --> 128
num_classes = 7
learning_rate = 0.001
weight_decay = 0.01
batch_size = 16
num_epochs = 30  
num_patches = 128*4  # 64 --> 128
projection_dim = 48
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 512]  # 1024 --> 384 (32*12)

## go complexnn/init.py and change the "from keras.utils.generic_utils....." to
## "from tensorflow.keras.utils import (serialize_keras_object, deserialize_keras_object)"
from   complexnn      import *
from tensorflow.keras.layers import (
    Dense,
)


# Q-MHSA module
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = QuaternionDense(embed_dim)
        self.key_dense = QuaternionDense(embed_dim)
        self.value_dense = QuaternionDense(embed_dim)
        self.combine_heads = QuaternionDense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output
        
def QF_Net(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = QuaternionConv2D(int(units/4), 3, strides=1, padding="same")(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Activation(tf.nn.gelu)(x)
        x = QuaternionConv2D(int(units/4), 3, strides=1, padding="same")(x)
    return x

def multilayer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = QuaternionDense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        # encoded = patch + self.position_embedding(positions)
        return encoded


def create_qvit_classifier():
    inputs = layers.Input(shape=input_shape)

    # position embedding
    encoded_patches = PatchEncoder(num_patches, projection_dim)(inputs)

    for _ in range(transformer_layers):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = MultiHeadSelfAttention(projection_dim, num_heads)(x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        x4 = tf.keras.layers.Reshape((16,32,48))(x3) # 32*12=384

        x5 = QF_Net(x4, hidden_units=transformer_units, dropout_rate=0.3)

        x6 = tf.keras.layers.Reshape((128*4, 48))(x5) #64-->128

        encoded_patches = layers.Add()([x6, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.4)(representation) # 0.5>>>0.3

    features = multilayer_perceptron(representation, hidden_units=mlp_head_units, dropout_rate=0.4) # 0.5>>>0.3

    logits = layers.Dense(num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = tf.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )


    checkpoint_filepath = "./tmp/RAFDB/model_{epoch:03d}-{val_accuracy:.4f}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )



    # Create an instance of the confusion matrix callback
    cm_callback = ConfusionMatrixCallback(
        test_data=test,
        test_labels=test_label,
        class_names=['anger','disgust','fear','happy','neural','sad','surprise']
    )

#     Define the early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy', 
        patience=30, 
        verbose=1, 
        mode='max', 
        restore_best_weights=True)

    history = model.fit(
        x=train,
        y=train_label,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(test, test_label),
        callbacks=[checkpoint_callback, cm_callback],
#         callbacks=[checkpoint_callback, cm_callback, early_stopping_callback],
    )

    return history        
    
# vit_classifier = create_qvit_classifier()
# history = run_experiment(vit_classifier)    

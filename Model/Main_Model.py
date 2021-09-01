import tensorflow as tf
from Model import Encoder, Decoder


class FcnModel:
    def __init__(self, config, save_model=False, show_summary=False):
        self.x_size = config["Image"]["Size_x"]
        self.y_size = config["Image"]["Size_y"]
        self.channels = config["Image"]["Size_channel"]
        self.classes = config["Network"]["num_classes"]
        self.use_pretrained_weights = config["train"]["weight_initialization"]["use_pretrained"]
        self.train_scratch = config["Network"]["train_from_scratch"]
        self.graph_path = config["Network"]["graph_path"]
        self.decode = config["Network"]["Decoder"]
        self.save_model = save_model
        self.show_summary = show_summary
        self.model_img = config["Network"]["modelpath"]

    def model(self):

        if self.use_pretrained_weights:
            json_file = open(self.graph_path, "r")
            model_json = json_file.read()
            json_file.close()
            model = tf.keras.models.model_from_json(model_json)
            return model
        else:
            if self.train_scratch:
                input_, pool_3, pool_4, encoder_out = Encoder.random_initialized_net(self.x_size, self.y_size,
                                                                                     num_channels=self.channels,
                                                                                     num_classes=self.classes)
            else:
                input_, pool_3, pool_4, encoder_out = Encoder.model_with_vgg_weights(self.x_size, self.y_size,
                                                                                     num_channels=self.channels,
                                                                                     num_classes=self.classes)

            if self.decode == "8X":
                decoder_out = Decoder.decoder_8x(encoder_out=encoder_out, pool3=pool_3, pool4=pool_4,
                                                 num_class=self.classes)
            elif self.decode == "16X":
                decoder_out = Decoder.decoder_16x(encoder_out=encoder_out, pool4=pool_4, num_class=self.classes)
            elif self.decode == "32X":
                decoder_out = Decoder.decoder_32x(encoder_out=encoder_out, num_class=self.classes)

            else:
                raise Exception("Unknown Decoder")

            model = tf.keras.Model(inputs=input_, outputs=decoder_out)

            if self.save_model:
                tf.keras.utils.plot_model(model, show_dtype=True, show_layer_names=True, show_shapes=True,
                                          to_file=self.model_img)

            if self.show_summary:
                print(model.summary())

            return model

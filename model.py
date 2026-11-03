from src.module import *
import torch.nn as nn
class PretrainedHiCReconstruction(nn.Module):
    def __init__(self, x_dim, x_encoder_layers, x_bottleneck,
                 x_recon_layers, use_gated=False,
                 x_encoder_to_bottleneck_activation=None,
                 x_recon_output_activation="Softplus",
                 dropout=0.3):
        super().__init__()
        num_features = x_dim
        self.x_encoder = TokenDeepEncoder(
            num_features=num_features,
            stage_dims=x_encoder_layers,
            bottleneck_dim=x_bottleneck,
            activation="ReLU",
            to_bottleneck_activation=x_encoder_to_bottleneck_activation,
            feature_dim=x_dim,
            use_gated=use_gated,
            dropout=dropout
        )
        self.x_reconstructor = TokenReconstructor(
            bottleneck_dim=x_bottleneck,
            num_features=num_features,
            stage_dims=x_recon_layers,
            activation="ReLU",
            feature_dim=x_dim,
            use_gated=use_gated,
            hidden_activation=None,
            output_activation=x_recon_output_activation,
            dropout=dropout
        )
    def forward(self, x):
        u1 = self.x_encoder(x)
        reconstructed_x = self.x_reconstructor(u1)
        return reconstructed_x, u1
class UniPrediction(nn.Module):
    def __init__(self, x_dim, x_encoder_layers,
                 x_bottleneck, x_recon_layers,
                 y_length=None, y_channels=None, x_p=None,
                 y_dec_cnn_hidden=None, y_dec_cnn_in=None,
                 y_dec_transformer_stage_dims=None,
                 y_dec_kernel_size=5,
                 y_tok_dec_stage_dims=None,
                 use_gated=False,
                 x_encoder_to_bottleneck_activation=None,
                 x_recon_output_activation="Softplus",
                 dropout=0.3):
        super().__init__()
        self.x_reconstruction = PretrainedHiCReconstruction(
            x_dim=x_dim,
            x_encoder_layers=x_encoder_layers,
            x_bottleneck=x_bottleneck,
            x_recon_layers=x_recon_layers,
            use_gated=use_gated,
            x_encoder_to_bottleneck_activation=x_encoder_to_bottleneck_activation,
            x_recon_output_activation=x_recon_output_activation,
            dropout=dropout
        )
        self.use_token_path = (y_length is not None and y_length == x_p)
        if self.use_token_path:
            self.x_pred_y_up = TokenReconstructor(
                bottleneck_dim=x_bottleneck,
                num_features=y_length,
                stage_dims=y_tok_dec_stage_dims,
                activation="Linear",
                feature_dim=y_channels,
                use_gated=use_gated,
                hidden_activation="ReLU",
                output_activation="Sigmoid",
                dropout=dropout
            )
        else:
            self.x_pred_y_up = Conv1DTransformerDecoder(
                bottleneck_dim=x_bottleneck,
                y_length=y_length,
                y_channels=y_channels,
                x_p=x_p,
                cnn_hidden=y_dec_cnn_hidden,
                cnn_in=y_dec_cnn_in,
                transformer_stage_dims=y_dec_transformer_stage_dims,
                kernel_size=y_dec_kernel_size,
                activation="Linear",
                use_gated=use_gated,
                hidden_activation="ReLU",
                output_activation="Sigmoid",
                dropout=dropout
            )
    def forward(self, x):
        reconstructed_x, u1 = self.x_reconstruction(x)
        predicted_y = self.x_pred_y_up(u1)
        return predicted_y, reconstructed_x, u1
class PretrainedYReconstruction(nn.Module):
    def __init__(self, x_dim,
                 y_length=None, y_channels=None, x_p=None,
                 y_enc_cnn_hidden=None, y_enc_cnn_out=None,
                 y_enc_transformer_stage_dims=None,
                 y_bottleneck=None,
                 y_rec_cnn_hidden=None, y_rec_cnn_in=None,
                 y_rec_transformer_stage_dims=None,
                 y_cnn_kernel_size=5,
                 y_tok_enc_stage_dims=None,
                 y_tok_rec_stage_dims=None,
                 use_gated=False,
                 dropout=0.3):
        super().__init__()
        num_features = x_dim
        self.use_token_path = (y_length is not None and y_length == x_p)
        if self.use_token_path:
            self.y_encoder = TokenDeepEncoder(
                num_features=y_length,
                stage_dims=y_tok_enc_stage_dims,
                bottleneck_dim=y_bottleneck,
                activation="ReLU",
                feature_dim=y_channels,
                use_gated=use_gated,
                dropout=dropout
            )
            self.y_reconstructor = TokenReconstructor(
                bottleneck_dim=y_bottleneck,
                num_features=y_length,
                stage_dims=y_tok_rec_stage_dims,
                activation="Linear",
                feature_dim=y_channels,
                use_gated=use_gated,
                hidden_activation="ReLU",
                output_activation="Sigmoid",
                dropout=dropout
            )
        else:
            self.y_encoder = Conv1DTransformerEncoder(
                y_length=y_length,
                y_channels=y_channels,
                x_p=x_p,
                cnn_hidden=y_enc_cnn_hidden,
                cnn_out=y_enc_cnn_out,
                transformer_stage_dims=y_enc_transformer_stage_dims,
                bottleneck=y_bottleneck,
                kernel_size=y_cnn_kernel_size,
                use_gated=use_gated,
                dropout=dropout
            )
            self.y_reconstructor = Conv1DTransformerDecoder(
                bottleneck_dim=y_bottleneck,
                y_length=y_length,
                y_channels=y_channels,
                x_p=x_p,
                cnn_hidden=y_rec_cnn_hidden,
                cnn_in=y_rec_cnn_in,
                transformer_stage_dims=y_rec_transformer_stage_dims,
                kernel_size=y_cnn_kernel_size,
                activation="Linear",
                use_gated=use_gated,
                hidden_activation="ReLU",
                output_activation="Sigmoid",
                dropout=dropout
            )
    def forward(self, y):
        if self.use_token_path:
            y_2d = y.squeeze(0).permute(1, 0)
            u2 = self.y_encoder(y_2d)                                             
            reconstructed_y = self.y_reconstructor(u2)                                
        else:
            u2 = self.y_encoder(y)                                                 
            reconstructed_y = self.y_reconstructor(u2)                                
        return reconstructed_y, u2
class ReversePrediction(nn.Module):
    def __init__(self, x_dim,
                 y_length=None, y_channels=None, x_p=None,
                 y_enc_cnn_hidden=None, y_enc_cnn_out=None,
                 y_enc_transformer_stage_dims=None,
                 y_bottleneck=None,
                 y_predict_extra=None,
                 y_rec_cnn_hidden=None, y_rec_cnn_in=None,
                 y_rec_transformer_stage_dims=None,
                 y_cnn_kernel_size=5,
                 y_tok_enc_stage_dims=None,
                 y_tok_rec_stage_dims=None,
                 use_gated=False,
                 x_recon_output_activation="Softplus",
                 dropout=0.3):
        super().__init__()
        num_features = x_dim
        self.y_reconstruction = PretrainedYReconstruction(
            x_dim=x_dim,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_enc_cnn_hidden=y_enc_cnn_hidden,
            y_enc_cnn_out=y_enc_cnn_out,
            y_enc_transformer_stage_dims=y_enc_transformer_stage_dims,
            y_bottleneck=y_bottleneck,
            y_rec_cnn_hidden=y_rec_cnn_hidden,
            y_rec_cnn_in=y_rec_cnn_in,
            y_rec_transformer_stage_dims=y_rec_transformer_stage_dims,
            y_cnn_kernel_size=y_cnn_kernel_size,
            y_tok_enc_stage_dims=y_tok_enc_stage_dims,
            y_tok_rec_stage_dims=y_tok_rec_stage_dims,
            use_gated=use_gated,
            dropout=dropout
        )
        self.y_pred_x_up = TokenPredictor(
            bottleneck_dim=y_bottleneck,
            num_output_tokens=num_features,
            stage_dims=y_predict_extra,
            activation="ReLU",
            feature_dim=x_dim,
            use_gated=use_gated,
            hidden_activation="LayerNorm",
            output_activation=x_recon_output_activation,
            dropout=dropout
        )
    def forward(self, y):
        reconstructed_y, u2 = self.y_reconstruction(y)
        predicted_x = self.y_pred_x_up(u2)
        return predicted_x, reconstructed_y, u2
class RingModel(nn.Module):
    def __init__(self, x_dim, x_encoder_layers,
                 x_bottleneck,
                 x_recon_layers,
                 y_length=None, y_channels=None, x_p=None,
                 y_enc_cnn_hidden=None, y_enc_cnn_out=None,
                 y_dec_cnn_hidden=None, y_dec_cnn_in=None,
                 y_enc_transformer_stage_dims=None,
                 y_dec_transformer_stage_dims=None,
                 y_rec_transformer_stage_dims=None,
                 y_bottleneck=None,
                 y_predict_extra=None,
                 y_cnn_kernel_size=5,
                 y_tok_enc_stage_dims=None,
                 y_tok_rec_stage_dims=None,
                 y_tok_dec_stage_dims=None,
                 use_gated=False,
                 x_encoder_to_bottleneck_activation=None,
                 x_recon_output_activation="Softplus",
                 dropout=0.3):
        super().__init__()
        self.y_prediction = UniPrediction(
            x_dim=x_dim,
            x_encoder_layers=x_encoder_layers,
            x_bottleneck=x_bottleneck,
            x_recon_layers=x_recon_layers,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_dec_cnn_hidden=y_dec_cnn_hidden,
            y_dec_cnn_in=y_dec_cnn_in,
            y_dec_transformer_stage_dims=y_dec_transformer_stage_dims,
            y_dec_kernel_size=y_cnn_kernel_size,
            y_tok_dec_stage_dims=y_tok_dec_stage_dims,
            use_gated=use_gated,
            x_encoder_to_bottleneck_activation=x_encoder_to_bottleneck_activation,
            x_recon_output_activation=x_recon_output_activation,
            dropout=dropout
        )
        self.x_prediction = ReversePrediction(
            x_dim=x_dim,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_enc_cnn_hidden=y_enc_cnn_hidden,
            y_enc_cnn_out=y_enc_cnn_out,
            y_enc_transformer_stage_dims=y_enc_transformer_stage_dims,
            y_bottleneck=y_bottleneck,
            y_predict_extra=y_predict_extra,
            y_rec_cnn_hidden=y_dec_cnn_hidden,
            y_rec_cnn_in=y_dec_cnn_in,
            y_rec_transformer_stage_dims=y_rec_transformer_stage_dims,
            y_cnn_kernel_size=y_cnn_kernel_size,
            y_tok_enc_stage_dims=y_tok_enc_stage_dims,
            y_tok_rec_stage_dims=y_tok_rec_stage_dims,
            use_gated=use_gated,
            x_recon_output_activation=x_recon_output_activation,
            dropout=dropout
        )
    def forward(self, x, y):
        predicted_y, reconstructed_x, u1 = self.y_prediction(x)
        predicted_x, reconstructed_y, u2 = self.x_prediction(y)
        return (predicted_x, predicted_y,
                reconstructed_x, reconstructed_y,
                u1, u2)
class HalfRingModel(nn.Module):
    def __init__(self, x_dim, x_encoder_layers,
                 x_bottleneck,
                 x_recon_layers,
                 y_length=None, y_channels=None, x_p=None,
                 y_enc_cnn_hidden=None, y_enc_cnn_out=None,
                 y_rec_cnn_hidden=None, y_rec_cnn_in=None,
                 y_enc_transformer_stage_dims=None,
                 y_rec_transformer_stage_dims=None,
                 y_bottleneck=None,
                 y_cnn_kernel_size=5,
                 y_tok_enc_stage_dims=None,
                 y_tok_rec_stage_dims=None,
                 use_gated=False,
                 x_encoder_to_bottleneck_activation=None,
                 x_recon_output_activation="Softplus",
                 dropout=0.3):
        super().__init__()
        self.x_reconstruction = PretrainedHiCReconstruction(
            x_dim=x_dim,
            x_encoder_layers=x_encoder_layers,
            x_bottleneck=x_bottleneck,
            x_recon_layers=x_recon_layers,
            use_gated=use_gated,
            x_encoder_to_bottleneck_activation=x_encoder_to_bottleneck_activation,
            x_recon_output_activation=x_recon_output_activation,
            dropout=dropout
        )
        self.y_reconstruction = PretrainedYReconstruction(
            x_dim=x_dim,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_enc_cnn_hidden=y_enc_cnn_hidden,
            y_enc_cnn_out=y_enc_cnn_out,
            y_enc_transformer_stage_dims=y_enc_transformer_stage_dims,
            y_bottleneck=y_bottleneck,
            y_rec_cnn_hidden=y_rec_cnn_hidden,
            y_rec_cnn_in=y_rec_cnn_in,
            y_rec_transformer_stage_dims=y_rec_transformer_stage_dims,
            y_cnn_kernel_size=y_cnn_kernel_size,
            y_tok_enc_stage_dims=y_tok_enc_stage_dims,
            y_tok_rec_stage_dims=y_tok_rec_stage_dims,
            use_gated=use_gated,
            dropout=dropout
        )
    def forward(self, x, y):
        reconstructed_x, u1 = self.x_reconstruction(x)
        reconstructed_y, u2 = self.y_reconstruction(y)
        return reconstructed_x, reconstructed_y, u1, u2

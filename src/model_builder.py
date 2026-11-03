from model import (PretrainedHiCReconstruction, UniPrediction,
                   ReversePrediction, RingModel,
                   PretrainedYReconstruction, HalfRingModel)
def build_model(model_type, config, x_p, y_p, rank, y_length=None, preset="SMALL"):
    model_cfg = config["presets"][preset]
    use_gated = model_cfg.get('use_gated', False)
    dropout = model_cfg.get('dropout', 0.3)
    x_enc  = model_cfg['x_encoder']
    x_rec  = model_cfg['x_reconstructor']
    x_enc_bn_act = x_enc.get('to_bottleneck_activation', None)
    x_recon_output_act = model_cfg.get('x_reconstructor_output_activation', "Softplus")
    y_bottleneck = model_cfg['y_encoder']['bottleneck']
    cnn_k        = model_cfg.get('y_cnn_kernel_size', 5)
    y_enc_cnn_h = model_cfg['y_encoder']['cnn_hidden']
    y_enc_cnn_out = model_cfg['y_encoder']['cnn_out']
    y_dec_cnn_h = model_cfg['y_predict_up']['cnn_hidden']
    y_dec_cnn_in  = model_cfg['y_predict_up']['cnn_in']
    y_rec_cnn_h = model_cfg['y_reconstructor']['cnn_hidden']
    y_rec_cnn_in  = model_cfg['y_reconstructor']['cnn_in']
    y_enc_sd  = model_cfg['y_encoder']['transformer_stage_dims']
    y_dec_sd  = model_cfg['y_predict_up']['transformer_stage_dims']
    y_rec_sd  = model_cfg['y_reconstructor']['transformer_stage_dims']
    y_pred_sd = model_cfg['y_predict_extra']['stage_dims']
    y_tok_enc_sd = model_cfg.get('y_token_encoder', {}).get('stage_dims', None)
    y_tok_rec_sd = model_cfg.get('y_token_reconstructor', {}).get('stage_dims', None)
    y_tok_dec_sd = model_cfg.get('y_token_predict', {}).get('stage_dims', None)
    y_channels = y_p                                           
    y_length   = y_length if y_length is not None else x_p                         
    if model_type == 'x_reconstruction':
        model = PretrainedHiCReconstruction(
            x_dim=x_p,
            x_encoder_layers=x_enc['layers'],
            x_bottleneck=x_enc['bottleneck'],
            x_recon_layers=x_rec,
            use_gated=use_gated,
            x_encoder_to_bottleneck_activation=x_enc_bn_act,
            x_recon_output_activation=x_recon_output_act,
            dropout=dropout
        )
    elif model_type == 'y_prediction':
        model = UniPrediction(
            x_dim=x_p,
            x_encoder_layers=x_enc['layers'],
            x_bottleneck=x_enc['bottleneck'],
            x_recon_layers=x_rec,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_dec_cnn_hidden=y_dec_cnn_h,
            y_dec_cnn_in=y_dec_cnn_in,
            y_dec_transformer_stage_dims=y_dec_sd,
            y_dec_kernel_size=cnn_k,
            y_tok_dec_stage_dims=y_tok_dec_sd,
            use_gated=use_gated,
            x_encoder_to_bottleneck_activation=x_enc_bn_act,
            x_recon_output_activation=x_recon_output_act,
            dropout=dropout
        )
    elif model_type == 'x_prediction':
        model = ReversePrediction(
            x_dim=x_p,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_enc_cnn_hidden=y_enc_cnn_h,
            y_enc_cnn_out=y_enc_cnn_out,
            y_enc_transformer_stage_dims=y_enc_sd,
            y_bottleneck=y_bottleneck,
            y_predict_extra=y_pred_sd,
            y_rec_cnn_hidden=y_rec_cnn_h,
            y_rec_cnn_in=y_rec_cnn_in,
            y_rec_transformer_stage_dims=y_rec_sd,
            y_cnn_kernel_size=cnn_k,
            y_tok_enc_stage_dims=y_tok_enc_sd,
            y_tok_rec_stage_dims=y_tok_rec_sd,
            use_gated=use_gated,
            x_recon_output_activation=x_recon_output_act,
            dropout=dropout
        )
    elif model_type == 'ring':
        model = RingModel(
            x_dim=x_p,
            x_encoder_layers=x_enc['layers'],
            x_bottleneck=x_enc['bottleneck'],
            x_recon_layers=x_rec,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_enc_cnn_hidden=y_enc_cnn_h,
            y_enc_cnn_out=y_enc_cnn_out,
            y_dec_cnn_hidden=y_dec_cnn_h,
            y_dec_cnn_in=y_dec_cnn_in,
            y_enc_transformer_stage_dims=y_enc_sd,
            y_dec_transformer_stage_dims=y_dec_sd,
            y_rec_transformer_stage_dims=y_rec_sd,
            y_bottleneck=y_bottleneck,
            y_predict_extra=y_pred_sd,
            y_cnn_kernel_size=cnn_k,
            y_tok_enc_stage_dims=y_tok_enc_sd,
            y_tok_rec_stage_dims=y_tok_rec_sd,
            y_tok_dec_stage_dims=y_tok_dec_sd,
            use_gated=use_gated,
            x_encoder_to_bottleneck_activation=x_enc_bn_act,
            x_recon_output_activation=x_recon_output_act,
            dropout=dropout
        )
    elif model_type == 'y_reconstruction':
        model = PretrainedYReconstruction(
            x_dim=x_p,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_enc_cnn_hidden=y_enc_cnn_h,
            y_enc_cnn_out=y_enc_cnn_out,
            y_enc_transformer_stage_dims=y_enc_sd,
            y_bottleneck=y_bottleneck,
            y_rec_cnn_hidden=y_rec_cnn_h,
            y_rec_cnn_in=y_rec_cnn_in,
            y_rec_transformer_stage_dims=y_rec_sd,
            y_cnn_kernel_size=cnn_k,
            y_tok_enc_stage_dims=y_tok_enc_sd,
            y_tok_rec_stage_dims=y_tok_rec_sd,
            use_gated=use_gated,
            dropout=dropout
        )
    elif model_type == 'halfring':
        model = HalfRingModel(
            x_dim=x_p,
            x_encoder_layers=x_enc['layers'],
            x_bottleneck=x_enc['bottleneck'],
            x_recon_layers=x_rec,
            y_length=y_length,
            y_channels=y_channels,
            x_p=x_p,
            y_enc_cnn_hidden=y_enc_cnn_h,
            y_enc_cnn_out=y_enc_cnn_out,
            y_rec_cnn_hidden=y_rec_cnn_h,
            y_rec_cnn_in=y_rec_cnn_in,
            y_enc_transformer_stage_dims=y_enc_sd,
            y_rec_transformer_stage_dims=y_rec_sd,
            y_bottleneck=y_bottleneck,
            y_cnn_kernel_size=cnn_k,
            y_tok_enc_stage_dims=y_tok_enc_sd,
            y_tok_rec_stage_dims=y_tok_rec_sd,
            use_gated=use_gated,
            x_encoder_to_bottleneck_activation=x_enc_bn_act,
            x_recon_output_activation=x_recon_output_act,
            dropout=dropout
        )
    else:
        raise ValueError(f"[RANK {rank}] Model type '{model_type}' is not supported")
    return model.to(rank)

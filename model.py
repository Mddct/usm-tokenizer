import torch


class TrainModel(torch.nn.Module):

    def __init__(
        self,
        config,
        encoder: torch.nn.Module,
        vq: torch.nn.Module,
        vqdecoder: torch.nn.Module,
        ctc: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.config = config
        self.encoder = encoder
        self.vq = self.decoder

        # TODO: reconstruct loss for vq
        self.vqdecoder = vqdecoder

        self.ctc = ctc

    def forward(self, batch, device):
        feats = batch['feats'].to(device)
        feats_lens = batch['feats_lens'].to(device)
        text = batch['text'].to(device)
        text_lens = batch['text_lens'].to(device)

        encoder, encoder_out_mask, intermediate_outs = self.encoder(
            feats, feats_lens, intermediate=True)
        z = intermediate_outs[self.config.self.config.intermediate_layer]

        # we assume z_maks == encoder_mask
        outputs = self.vq(z, 1 - encoder_out_mask.squeeze(1).to(z.dtype))
        loss_ctc, _ = self.ctc(encoder,
                               encoder_out_mask.squeeze(1).sum(), text,
                               text_lens)
        return {'vq_outputs': outputs, 'loss_ctc': loss_ctc}

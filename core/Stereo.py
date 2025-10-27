import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import BasicMultiUpdateBlock
from .extractor import MultiBasicEncoder, Feature
from HLFS import hlfs
from .submodule import *
from .MGI import mgi,Geometric_Volume
import time

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class Stereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch",
                                      downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])

        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
        )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = mgi(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        self.HLFS = hlfs()
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)

        return up_disp

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp // 4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            geo_volume = self.cost_agg(gwc_volume, features_left)

            prob = F.softmax(self.classifier(geo_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.args.max_disp // 4)

            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                        zip(inp_list, self.context_zqr_convs)]

        geo_block = Geometric_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_volume.float(),
                           radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []


        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp,
                                                                      iter16=self.args.n_gru_layers == 3,
                                                                      iter08=self.args.n_gru_layers >= 2)

            disp = disp + delta_disp
            if test_mode and itr < iters - 1:
                continue


            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            if itr == iters - 1:
               refine_disp = self.HLFS(disp_up, image1, image2)
               disp_up = disp_up + refine_disp

            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import conv, rescale_flow
from .pwc_modules import upsample2d_as, initialize_msra
from .pwc_modules import WarpingLayer, FeatureExtractor
from .pwc_modules import ContextNetwork, FlowEstimatorDense
from .pwc_modules import FlowAndOccEstimatorDense, FlowAndOccContextNetwork
from .correlation_package.correlation import Correlation


class TRFlow(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(TRFlow, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.flow_estimators = nn.ModuleList()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = 2 + self.dim_corr
                #num_ch_in = self.dim_corr
            else:
                num_ch_in = 2 + (self.dim_corr + ch + 2)

            layer = FlowEstimatorDense(num_ch_in)
            self.flow_estimators.append(layer)

        self.context_networks = ContextNetwork(2 + (self.dim_corr + 32 + 2) + 448 + 2)

        initialize_msra(self.modules())

    def forward(self, input_dict):

        if 'input_images' in input_dict.keys():
            list_imgs = input_dict['input_images']
        else:
            x1_raw = input_dict['input1']
            x2_raw = input_dict['input2']
            list_imgs = [x1_raw, x2_raw]

        _, _, height_im, width_im = list_imgs[0].size()

        # on the bottom level are original images
        list_pyramids = [] #indices : [time][level]
        for im in list_imgs:
            list_pyramids.append(self.feature_pyramid_extractor(im) + [im])

        # outputs
        output_dict = {}
        flows = [] #indices : [level][time]
        flows_b = [] #indices : [level][time]
        for l in range(self.output_level + 1):
            flows.append([])
            flows_b.append([])

        # init
        b_size, _, h_x1, w_x1, = list_pyramids[0][0].size()
        init_dtype = list_pyramids[0][0].dtype
        init_device = list_pyramids[0][0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        previous_flow = []

        for i in range(len(list_imgs) - 1): #time
            x1_pyramid, x2_pyramid = list_pyramids[i:i+2]

            for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
                if i == 0:
                    bs_, _, h_, w_, = list_pyramids[0][l].size()
                    previous_flow.append(torch.zeros(bs_, 2, h_, w_, dtype=init_dtype, device=init_device).float())

                # warping
                if l == 0:
                    x2_warp = x2
                    x1_warp = x1
                else:
                    flow = upsample2d_as(flow, x1, mode="bilinear")
                    flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                    x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
                    x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                #x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)

                # correlation
                out_corr = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
                out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)

                out_corr_relu = self.leakyRELU(out_corr)
                out_corr_relu_b = self.leakyRELU(out_corr_b)

                # flow estimator
                if l == 0:
                    if i > 0: #temporal connection:
                        previous_flow[l] = self.warping_layer(previous_flow[l],
                                        flows_b[l][-1], height_im, width_im, self._div_flow)
                    features = torch.cat([previous_flow[l], out_corr_relu], 1)
                    features_b = torch.cat([torch.zeros_like(previous_flow[l]), out_corr_relu_b], 1)
                    x_intm, flow = self.flow_estimators[l](features)
                    with torch.no_grad():
                        x_intm_b, flow_b = self.flow_estimators[l](features_b)
                    previous_flow[l] = flow
                else:
                    if i > 0: #temporal connection:
                        previous_flow[l] = self.warping_layer(previous_flow[l],
                                        flows_b[l][-1], height_im, width_im, self._div_flow)
                    features = torch.cat([previous_flow[l], out_corr_relu, x1, flow], 1)
                    features_b = torch.cat([torch.zeros_like(previous_flow[l]), out_corr_relu_b, x2, flow_b], 1)
                    x_intm, flow = self.flow_estimators[l](features)
                    with torch.no_grad():
                        x_intm_b, flow_b = self.flow_estimators[l](features_b)
                    previous_flow[l] = flow

                # upsampling or post-processing
                flows_b[l].append(flow_b)
                if l != self.output_level:
                    flows[l].append(flow)
                else:
                    flow_res = self.context_networks(torch.cat([x_intm, flow], dim=1))
                    flow = flow + flow_res
                    flows[l].append(flow)
                    break
            #flow = flows[0][i] # pg! (for the next step, init flow with the coarsest flow from previous time step)
            flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        if self.training:
            if len(list_imgs) > 2:
                for l in range(len(flows)):
                    flows[l] = torch.stack(flows[l], 0)
            else:
                for l in range(len(flows)):
                    flows[l] = flows[l][0]
            output_dict['flow'] = flows
            return output_dict
        else:
            output_dict_eval = {}
            if len(list_imgs) > 2:
                out_flow = []
                for i in range(len(flows[0])):
                    out_flow.append(upsample2d_as(flows[-1][i], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow))
                out_flow = torch.stack(out_flow, 0)
            else:
                out_flow = upsample2d_as(flows[-1][0], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow)
            output_dict_eval['flow'] = out_flow
            return output_dict_eval


class TRFlow_occjoint(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(TRFlow_occjoint, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        #self.flow_estimators = nn.ModuleList()
        #self.occ_estimators = nn.ModuleList()
        self.flow_and_occ_estimators = nn.ModuleList()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = 2 + self.dim_corr
            else:
                num_ch_in = 2 + self.dim_corr + ch + 2 + 1

            layer = FlowAndOccEstimatorDense(num_ch_in)
            self.flow_and_occ_estimators.append(layer)

        self.context_networks = FlowAndOccContextNetwork(2 + (self.dim_corr + 32 + 2 + 1) + 448 + 2 + 1)

        initialize_msra(self.modules())

    def forward(self, input_dict):

        if 'input_images' in input_dict.keys():
            list_imgs = input_dict['input_images']
        else:
            x1_raw = input_dict['input1']
            x2_raw = input_dict['input2']
            list_imgs = [x1_raw, x2_raw]

        _, _, height_im, width_im = list_imgs[0].size()

        # on the bottom level are original images
        list_pyramids = [] #indices : [time][level]
        for im in list_imgs:
            list_pyramids.append(self.feature_pyramid_extractor(im) + [im])

        # outputs
        output_dict = {}
        flows = [] #indices : [level][time]
        flows_b = [] #indices : [level][time]
        occs = [] #indices : [level][time]
        for l in range(self.output_level + 1):
            flows.append([])
            flows_b.append([])
            occs.append([])

        # init
        b_size, _, h_x1, w_x1, = list_pyramids[0][0].size()
        init_dtype = list_pyramids[0][0].dtype
        init_device = list_pyramids[0][0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        occ = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        occ_b = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        previous_flow = []

        for i in range(len(list_imgs) - 1): #time
            x1_pyramid, x2_pyramid = list_pyramids[i:i+2]

            for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
                if i == 0:
                    bs_, _, h_, w_, = list_pyramids[0][l].size()
                    previous_flow.append(torch.zeros(bs_, 2, h_, w_, dtype=init_dtype, device=init_device).float())

                # warping
                if l == 0:
                    x2_warp = x2
                    x1_warp = x1
                else:
                    flow = upsample2d_as(flow, x1, mode="bilinear")
                    flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                    occ = upsample2d_as(occ, x1, mode="bilinear")
                    occ_b = upsample2d_as(occ_b, x1, mode="bilinear")
                    x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
                    x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                #x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)

                # correlation
                out_corr = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
                out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)

                out_corr_relu = self.leakyRELU(out_corr)
                out_corr_relu_b = self.leakyRELU(out_corr_b)

                # flow estimator
                if l == 0:
                    if i > 0: #temporal connection:
                        previous_flow[l] = self.warping_layer(previous_flow[l],
                                        flows_b[l][-1], height_im, width_im, self._div_flow)
                    features = torch.cat([previous_flow[l], out_corr_relu], 1)
                    features_b = torch.cat([torch.zeros_like(previous_flow[l]), out_corr_relu_b], 1)
                    x_intm, flow, occ = self.flow_and_occ_estimators[l](features)
                    with torch.no_grad():
                        x_intm_b, flow_b, occ_b = self.flow_and_occ_estimators[l](features_b)
                    previous_flow[l] = flow
                else:
                    if i > 0: #temporal connection:
                        previous_flow[l] = self.warping_layer(previous_flow[l],
                                        flows_b[l][-1], height_im, width_im, self._div_flow)
                    features = torch.cat([previous_flow[l], out_corr_relu, x1, flow, occ], 1)
                    features_b = torch.cat([torch.zeros_like(previous_flow[l]), out_corr_relu_b, x2, flow_b, occ_b], 1)
                    x_intm, flow, occ = self.flow_and_occ_estimators[l](features)
                    with torch.no_grad():
                        x_intm_b, flow_b, occ_b = self.flow_and_occ_estimators[l](features_b)
                    previous_flow[l] = flow

                # upsampling or post-processing
                flows_b[l].append(flow_b)
                if l != self.output_level:
                    flows[l].append(flow)
                    occs[l].append(occ)
                else:
                    flow_fine, occ_fine = self.context_networks(torch.cat([x_intm, flow, occ], dim=1))
                    flow = flow + flow_fine
                    occ = occ + occ_fine
                    flows[l].append(flow)
                    occs[l].append(occ)
                    break
            #flow = flows[0][i] # pg! (for the next step, init flow with the coarsest flow from previous time step)
            flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
            occ = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        if self.training:
            if len(list_imgs) > 2:
                for l in range(len(flows)):
                    flows[l] = torch.stack(flows[l], 0)
                    occs[l] = torch.stack(occs[l], 0)
            else:
                for l in range(len(flows)):
                    flows[l] = flows[l][0]
                    occs[l] = occs[l][0]
            output_dict['flow'] = flows
            output_dict['occ'] = occs
            return output_dict
        else:
            output_dict_eval = {}
            if len(list_imgs) > 2:
                out_flow = []
                out_occ = []
                for i in range(len(flows[0])):
                    out_flow.append(upsample2d_as(flows[-1][i], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow))
                    out_occ.append(upsample2d_as(occs[-1][i], list_imgs[0], mode="bilinear"))
                out_flow = torch.stack(out_flow, 0)
                out_occ = torch.stack(out_occ, 0)
            else:
                out_flow = upsample2d_as(flows[-1][0], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow)
                out_occ = upsample2d_as(occs[-1][0], list_imgs[0], mode="bilinear")
            output_dict_eval['flow'] = out_flow
            output_dict_eval['occ'] = out_occ
            return output_dict_eval



class TRFlow_irr(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(TRFlow_irr, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 2 + self.dim_corr + 32 + 2

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        initialize_msra(self.modules())

    def forward(self, input_dict):

        if 'input_images' in input_dict.keys():
            list_imgs = input_dict['input_images']
        else:
            x1_raw = input_dict['input1']
            x2_raw = input_dict['input2']
            list_imgs = [x1_raw, x2_raw]

        _, _, height_im, width_im = list_imgs[0].size()

        # on the bottom level are original images
        list_pyramids = [] #indices : [time][level]
        for im in list_imgs:
            list_pyramids.append(self.feature_pyramid_extractor(im) + [im])

        # outputs
        output_dict = {}
        flows = [] #indices : [level][time]
        flows_b = [] #indices : [level][time]
        for l in range(self.output_level + 1):
            flows.append([])
            flows_b.append([])

        # init
        b_size, _, h_x1, w_x1, = list_pyramids[0][0].size()
        init_dtype = list_pyramids[0][0].dtype
        init_device = list_pyramids[0][0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        previous_flow = []

        for i in range(len(list_imgs) - 1): #time
            x1_pyramid, x2_pyramid = list_pyramids[i:i+2]

            for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
                if i == 0:
                    bs_, _, h_, w_, = list_pyramids[0][l].size()
                    previous_flow.append(torch.zeros(bs_, 2, h_, w_, dtype=init_dtype, device=init_device).float())

                # warping
                if l == 0:
                    x2_warp = x2
                    x1_warp = x1
                else:
                    flow = upsample2d_as(flow, x1, mode="bilinear")
                    flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                    x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
                    x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                #x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)

                # correlation
                out_corr = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
                out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)

                out_corr_relu = self.leakyRELU(out_corr)
                out_corr_relu_b = self.leakyRELU(out_corr_b)

                if i > 0: #temporal connection:
                    previous_flow[l] = self.warping_layer(previous_flow[l],
                                    flows_b[l][-1], height_im, width_im, self._div_flow)

                # concat and estimate flow
                flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=True)
                flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)

                features = torch.cat([previous_flow[l], out_corr_relu, x1_1by1, flow], 1)
                features_b = torch.cat([torch.zeros_like(previous_flow[l]), out_corr_relu_b, x2_1by1, flow_b], 1)

                x_intm, flow_res = self.flow_estimators(features)
                flow = flow + flow_res
                with torch.no_grad():
                    x_intm_b, flow_b_res = self.flow_estimators(features_b)
                    flow_b = flow_b + flow_b_res

                flow_fine = self.context_networks(torch.cat([x_intm, flow], dim=1))
                flow = flow + flow_fine
                with torch.no_grad():
                    flow_b_fine = self.context_networks(torch.cat([x_intm_b, flow_b], dim=1))
                    flow_b = flow_b + flow_b_fine

                flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=False)
                flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)

                previous_flow[l] = flow
                flows[l].append(flow)
                flows_b[l].append(flow_b)

                # upsampling or post-processing
                if l == self.output_level:
                    break

            #flow = flows[0][i] # pg! (for the next step, init flow with the coarsest flow from previous time step)
            flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
            flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        if self.training:
            if len(list_imgs) > 2:
                for l in range(len(flows)):
                    flows[l] = torch.stack(flows[l], 0)
            else:
                for l in range(len(flows)):
                    flows[l] = flows[l][0]
            output_dict['flow'] = flows
            return output_dict
        else:
            output_dict_eval = {}
            if len(list_imgs) > 2:
                out_flow = []
                for i in range(len(flows[0])):
                    out_flow.append(upsample2d_as(flows[-1][i], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow))
                out_flow = torch.stack(out_flow, 0)
            else:
                out_flow = upsample2d_as(flows[-1][0], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow)
            output_dict_eval['flow'] = out_flow
            return output_dict_eval


class TRFlow_irr_occjoint(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(TRFlow_irr_occjoint, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 2 + self.dim_corr + 32 + 2 + 1

        self.flow_estimators = FlowAndOccEstimatorDense(self.num_ch_in)

        self.context_networks = FlowAndOccContextNetwork(self.num_ch_in + 448 + 2 + 1)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        initialize_msra(self.modules())

    def forward(self, input_dict):

        if 'input_images' in input_dict.keys():
            list_imgs = input_dict['input_images']
        else:
            x1_raw = input_dict['input1']
            x2_raw = input_dict['input2']
            list_imgs = [x1_raw, x2_raw]

        _, _, height_im, width_im = list_imgs[0].size()

        # on the bottom level are original images
        list_pyramids = [] #indices : [time][level]
        for im in list_imgs:
            list_pyramids.append(self.feature_pyramid_extractor(im) + [im])

        # outputs
        output_dict = {}
        flows = [] #indices : [level][time]
        flows_b = [] #indices : [level][time]
        occs = [] #indices : [level][time]
        for l in range(self.output_level + 1):
            flows.append([])
            flows_b.append([])
            occs.append([])

        # init
        b_size, _, h_x1, w_x1, = list_pyramids[0][0].size()
        init_dtype = list_pyramids[0][0].dtype
        init_device = list_pyramids[0][0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        occ = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        occ_b = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        previous_flow = []

        for i in range(len(list_imgs) - 1): #time
            x1_pyramid, x2_pyramid = list_pyramids[i:i+2]

            for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
                if i == 0:
                    bs_, _, h_, w_, = list_pyramids[0][l].size()
                    previous_flow.append(torch.zeros(bs_, 2, h_, w_, dtype=init_dtype, device=init_device).float())

                # warping
                if l == 0:
                    x2_warp = x2
                    x1_warp = x1
                else:
                    flow = upsample2d_as(flow, x1, mode="bilinear")
                    flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                    occ = upsample2d_as(occ, x1, mode="bilinear")
                    occ_b = upsample2d_as(occ_b, x1, mode="bilinear")
                    x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
                    x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                #x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)

                # correlation
                out_corr = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
                out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)

                out_corr_relu = self.leakyRELU(out_corr)
                out_corr_relu_b = self.leakyRELU(out_corr_b)

                if i > 0: #temporal connection:
                    previous_flow[l] = self.warping_layer(previous_flow[l],
                                    flows_b[l][-1], height_im, width_im, self._div_flow)

                # concat and estimate flow
                flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=True)
                flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

                x1_1by1 = self.conv_1x1[l](x1)
                x2_1by1 = self.conv_1x1[l](x2)

                features = torch.cat([previous_flow[l], out_corr_relu, x1_1by1, flow, occ], 1)
                features_b = torch.cat([torch.zeros_like(previous_flow[l]), out_corr_relu_b, x2_1by1, flow_b, occ_b], 1)

                x_intm, flow_res, occ_res = self.flow_estimators(features)
                flow = flow + flow_res
                occ = occ + occ_res
                with torch.no_grad():
                    x_intm_b, flow_b_res, occ_b_res = self.flow_estimators(features_b)
                    flow_b = flow_b + flow_b_res
                    occ_b = occ_b + occ_b_res

                flow_fine, occ_fine = self.context_networks(torch.cat([x_intm, flow, occ], dim=1))
                flow = flow + flow_fine
                occ = occ + occ_fine
                with torch.no_grad():
                    flow_b_fine, occ_b_fine = self.context_networks(torch.cat([x_intm_b, flow_b, occ_b], dim=1))
                    flow_b = flow_b + flow_b_fine
                    occ_b = occ_b + occ_b_fine

                flow = rescale_flow(flow, self._div_flow, width_im, height_im, to_local=False)
                flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)

                previous_flow[l] = flow
                flows[l].append(flow)
                flows_b[l].append(flow_b)
                occs[l].append(occ)

                # upsampling or post-processing
                if l == self.output_level:
                    break

            #flow = flows[0][i] # pg! (for the next step, init flow with the coarsest flow from previous time step)
            flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
            flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
            occ = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
            occ_b = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        if self.training:
            if len(list_imgs) > 2:
                for l in range(len(flows)):
                    flows[l] = torch.stack(flows[l], 0)
                    occs[l] = torch.stack(occs[l], 0)
            else:
                for l in range(len(flows)):
                    flows[l] = flows[l][0]
                    occs[l] = occs[l][0]
            output_dict['flow'] = flows
            output_dict['occ'] = occs
            return output_dict
        else:
            output_dict_eval = {}
            if len(list_imgs) > 2:
                out_flow = []
                out_occ = []
                for i in range(len(flows[0])):
                    out_flow.append(upsample2d_as(flows[-1][i], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow))
                    out_occ.append(upsample2d_as(occs[-1][i], list_imgs[0], mode="bilinear"))
                out_flow = torch.stack(out_flow, 0)
                out_occ = torch.stack(out_occ, 0)
            else:
                out_flow = upsample2d_as(flows[-1][0], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow)
                out_occ = upsample2d_as(occs[-1][0], list_imgs[0], mode="bilinear")
            output_dict_eval['flow'] = out_flow
            output_dict_eval['occ'] = out_occ
            return output_dict_eval

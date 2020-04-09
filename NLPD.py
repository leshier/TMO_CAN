import torch
import math
import torch.nn.functional as F


def upsample(img, odd, filt):
    img = F.pad(img, (1, 1, 1, 1), mode='replicate')
    h = 2 * img.shape[2]
    w = 2 * img.shape[3]
    if img.is_cuda:
        o = torch.zeros([img.shape[0], img.shape[1], h, w], device=img.get_device())
    else:
        o = torch.zeros([img.shape[0], img.shape[1], h, w])
    o[:, :, 0:h:2, 0:w:2] = 4 * img
    o = F.conv2d(o, filt, padding=math.floor(filt.shape[2] / 2))
    o = o[:, :, 2:h - 2 - odd[0], 2:w - 2 - odd[1]]

    return o


def downsample(img, filt):
    pad = math.floor(filt.shape[2]/2)
    img = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    o = F.conv2d(img, filt)
    o = o[:, :, :img.shape[2]:2, :img.shape[3]:2]

    return o


def laplacian_pyramid_s(img, n_lev, filt):
    pyr = [0] * n_lev
    o = img

    for i in range(0, n_lev - 1):
        g = downsample(o, filt)
        h_odd = g.shape[2] * 2 - o.shape[2]
        w_odd = g.shape[3] * 2 - o.shape[3]
        pyr[i] = o - upsample(g, [h_odd, w_odd], filt)
        o = g

    pyr[n_lev - 1] = o

    return pyr


def nlp(img, n_lev, params):  # 求得原图的拉普拉斯金字塔
        npyr = [0] * n_lev

        img = torch.pow(img, 1 / params['gamma'])
        # img = torch.log(img)
        # img = (1e3/math.pi)*torch.atan(img)
        pyr = laplacian_pyramid_s(img, n_lev, params['F1'])

        for i in range(0, n_lev-1):
            pad = math.floor(params['filts'][0].shape[2] / 2)
            apyr = F.pad(torch.abs(pyr[i]), (pad, pad, pad, pad), mode='replicate')
            den = F.conv2d(apyr, params['filts'][0]) + params['sigmas'][0]
            npyr[i] = pyr[i] / den

        pad = math.floor(params['filts'][1].shape[2] / 2)
        apyr = F.pad(torch.abs(pyr[n_lev-1]), (pad, pad, pad, pad), mode='replicate')
        den = F.conv2d(apyr, params['filts'][1]) + params['sigmas'][1]

        npyr[n_lev-1] = pyr[n_lev-1] / den

        return npyr


class NLPD_Loss(torch.nn.Module):
    def __init__(self):
        super(NLPD_Loss, self).__init__()
        self.params = dict()
        self.params['gamma'] = 2.60
        # self.params['gamma'] = 10
        self.params['filts'] = dict()
        self.params['filts'][0] = torch.tensor([[0.0400, 0.0400, 0.0500, 0.0400, 0.0400],
                                                [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                                                [0.0500, 0.0400, 0.0500, 0.0400, 0.0500],
                                                [0.0400, 0.0300, 0.0400, 0.0300, 0.0400],
                                                [0.0400, 0.0400, 0.0500, 0.0400, 0.0400]],
                                                dtype=torch.float)
        self.params['filts'][0] = self.params['filts'][0].unsqueeze(0).unsqueeze(0)

        self.params['filts'][1] = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0], [0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0]],
                                                dtype=torch.float)
        self.params['filts'][1] = self.params['filts'][1].unsqueeze(0).unsqueeze(0)

        self.params['sigmas'] = torch.tensor([0.1700, 4.8600], dtype=torch.float)
        # self.params['sigmas'] = torch.tensor([0.177, 20], dtype=torch.float)

        self.params['F1'] = torch.tensor([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                                          [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                          [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                                          [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                                          [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                                          dtype=torch.float)
        self.params['F1'] = self.params['F1'].unsqueeze(0).unsqueeze(0)

        self.exp_s = 2.00
        self.exp_f = 0.60

    def forward(self, h_img, l_img, n_lev=None):
        if n_lev is None:
            n_lev = math.floor(math.log(min(h_img.shape[2:]), 2)) - 2  # 求得金字塔的层数

            filts_0 = self.params['filts'][0]
            filts_1 = self.params['filts'][1]
            sigmas = self.params['sigmas']
            F1 = self.params['F1']

        if h_img.is_cuda:
            filts_0 = filts_0.cuda(h_img.get_device())
            filts_1 = filts_1.cuda(h_img.get_device())
            sigmas = sigmas.cuda(h_img.get_device())
            F1 = F1.cuda(h_img.get_device())

        filts_0 = filts_0.type_as(h_img)
        filts_1 = filts_1.type_as(h_img)
        sigmas = sigmas.type_as(h_img)
        F1 = F1.type_as(h_img)

        self.params['filts'][0] = filts_0
        self.params['filts'][1] = filts_1
        self.params['sigmas'] = sigmas
        self.params['F1'] = F1

        h_pyr = nlp(h_img, n_lev, self.params)
        l_pyr = nlp(l_img, n_lev, self.params)

        dis = []

        for i in range(0, n_lev):
            diff = torch.pow(torch.abs(h_pyr[i] - l_pyr[i]), self.exp_s)
            diff_pow = torch.pow(torch.mean(torch.mean(diff, dim=-1), dim=-1), self.exp_f / self.exp_s)
            dis.append(diff_pow)

        dis = torch.cat(dis, -1)
        loss = torch.pow(torch.mean(dis, dim=-1), 1. / self.exp_f)

        return loss.mean()



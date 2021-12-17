import torch as th
import torch.nn.functional as F


def cpt_mean_maximum_similarity(vid_rep,
                                txt_rep):
  # vid_rep: [b, 218, 512]
  # txt_rep: [b, 30, 512]
  txt_bs = txt_rep.shape[0]
  vid_bs = vid_rep.shape[0]
  txt_rep_norm = F.normalize(txt_rep, dim=-1)
  vid_rep_norm = F.normalize(vid_rep, dim=-1)
  txt_rep_norm = txt_rep_norm.unsqueeze(1).repeat(1, vid_bs, 1, 1)
  vid_rep_norm = vid_rep_norm.unsqueeze(0).repeat(txt_bs, 1, 1, 1)
  sim = th.matmul(txt_rep_norm, vid_rep_norm.permute(0, 1, 3, 2))  # [bs, bs, 30, 218]
  t2v = th.mean(th.max(sim, dim=-1)[0], dim=-1)  # [bs, bs, 1]
  v2t = th.mean(th.max(sim, dim=-2)[0], dim=-1)  # [bs, bs, 1]
  t2v = t2v.view(txt_bs, vid_bs)
  v2t = v2t.view(vid_bs, txt_bs)

  return {
    "t2v": t2v,
    "v2t": v2t
  }

def cpt_mean_maximum_similarity2(vid_rep,
                                txt_rep):
  # vid_rep: [b, 218, 512]
  # txt_rep: [b, 30, 512]
  txt_bs, txt_n_tokens, txt_dim = txt_rep.shape
  vid_bs, vid_n_tokens, vid_dim = vid_rep.shape
  txt_rep_norm = F.normalize(txt_rep, dim=-1)
  vid_rep_norm = F.normalize(vid_rep, dim=-1)
  txt_rep_norm = txt_rep_norm.view(txt_bs * txt_n_tokens, txt_dim)  # [bs*218, 512]
  vid_rep_norm = vid_rep_norm.view(vid_bs * vid_n_tokens, vid_dim)  # [bs*30, 512]
  sim = th.matmul(txt_rep_norm, vid_rep_norm.T)  # [bs*218, bs*30]
  sim = sim.view(txt_bs, txt_n_tokens, vid_bs, vid_n_tokens)  # [bs, 218, bs, 30]
  t2v = th.mean(th.max(sim, dim=-1)[0], dim=1)  # [bs, bs]
  v2t = th.mean(th.max(sim, dim=1)[0], dim=-1)  # [bs, bs]

  return {
    "t2v": t2v,
    "v2t": v2t
  }


if __name__ == '__main__':
  txt_rep = F.normalize(th.randn(128, 30, 1024, requires_grad=True), dim=1).cuda(1)
  vid_rep = F.normalize(th.randn(128, 40, 1024, requires_grad=True), dim=1).cuda(1)
  sim1 = cpt_mean_maximum_similarity(vid_rep=vid_rep, txt_rep=txt_rep)
  sim2 = cpt_mean_maximum_similarity2(vid_rep=vid_rep, txt_rep=txt_rep)
  pass
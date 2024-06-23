import torch


def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, last_state=None):
    frames = x.split(1, -2)
    m_frames = []

    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue

        m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, -2)
    
    pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r

    return pcen_



input1 = torch.rand([1,1,126,40])
input2 = input1.clone()
# input1 = torch.ones([1,1,126,40])
# input2 = torch.ones([1,1,126,40])

out1 = pcen(input1)

out2 = pcen(input2, training=True)

res = torch.sum(out1 - out2)

aa = 0
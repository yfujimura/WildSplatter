from __future__ import annotations

import copy
import torch
import torch.nn as nn


def _copy_output_conv2(
    output_conv2: nn.Sequential,
) -> nn.Sequential:

    new_seq: nn.Sequential = copy.deepcopy(output_conv2)
    return new_seq

def _copy_output_conv2_with_zero_init(
    output_conv2: nn.Sequential,
) -> nn.Sequential:

    new_seq: nn.Sequential = copy.deepcopy(output_conv2)
    with torch.no_grad():
        for m in new_seq.modules():
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.zero_()
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.zero_()

    return new_seq

def _copy_output_conv2_with_app_in(
    output_conv2: nn.Sequential,
    app_dim: int,
) -> nn.Sequential:
    """
    Copy pretrained output_conv2 (Sequential) and expand the first Conv2d in_channels
    from 128 -> 128 + app_dim, while keeping pretrained behavior at init by:
      - copying old weights into [:, :128, :, :]
      - zero-initializing weights for the appended channels [:, 128:, :, :]
      - copying bias 그대로
    """
    assert isinstance(output_conv2, nn.Sequential), type(output_conv2)
    assert len(output_conv2) >= 1, "output_conv2 must have at least one layer"
    assert isinstance(output_conv2[0], nn.Conv2d), "output_conv2[0] must be Conv2d"

    old_conv1: nn.Conv2d = output_conv2[0]
    old_in = old_conv1.in_channels
    assert old_in == 128, f"Expected old in_channels=128, but got {old_in}"

    new_in = old_in + int(app_dim)

    # まず全体を deepcopy しておく（ReLU/Conv 等含めて pretrained 状態を複製）
    new_seq: nn.Sequential = copy.deepcopy(output_conv2)

    # 1層目だけ差し替え
    new_conv1 = nn.Conv2d(
        in_channels=new_in,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        dilation=old_conv1.dilation,
        groups=old_conv1.groups,
        bias=(old_conv1.bias is not None),
        padding_mode=old_conv1.padding_mode,
    )

    # 重み・バイアスを “pretrained を壊さない” 形で初期化
    with torch.no_grad():
        # いったんゼロ（追加チャネル分も含めてゼロになる）
        new_conv1.weight.zero_()

        # 既存 128ch 部分は pretrained をコピー
        new_conv1.weight[:, :old_in, :, :].copy_(old_conv1.weight)

        # bias もコピー
        if old_conv1.bias is not None:
            new_conv1.bias.copy_(old_conv1.bias)

    new_seq[0] = new_conv1
    return new_seq

def _copy_output_conv2_aux_for_conf(
    output_conv2_aux: nn.ModuleList,
    conf_head_idx: int = 3,   # ← (0-3) の「3」を指定
) -> nn.Sequential:
    """
    scratch.output_conv2_aux を deepcopy し、
    conf_head_idx 番目の Sequential の最後の Conv2d を
    out_channels=1 & zero init に差し替えて confidence head を作る。

    戻り値は nn.Sequential（conf 用 head 単体）。
    """
    # ModuleList を丸ごとコピー（pretrained を壊さない）
    aux_all = copy.deepcopy(output_conv2_aux)

    # conf 用に使う Sequential
    head: nn.Sequential = aux_all[conf_head_idx]

    # 最後の層チェック
    if not isinstance(head[-1], nn.Conv2d):
        raise TypeError(
            f"Expected head[-1] to be Conv2d, but got {type(head[-1])}"
        )

    last: nn.Conv2d = head[-1]

    # in_channels は pretrained のまま引き継ぐ（=32）
    new_last = nn.Conv2d(
        in_channels=last.in_channels,
        out_channels=1,        # s を 1ch で出す
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )

    # ★重要：初期は s=0 → conf=exp(0)=1
    nn.init.zeros_(new_last.weight)
    nn.init.zeros_(new_last.bias)

    # 差し替え
    head[-1] = new_last

    return head


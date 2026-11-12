# -*- coding: utf-8 -*-
"""Быстрый потоковый Qwen3-TTS: статический KV-кэш + CUDA Graph на весь шаг кадра + потоковый декодер.

Оригинальный qwen_tts на каждый кадр (12.5 в секунду) вызывает HF generate() для 15 под-кодов,
и почти всё время уходит на накладные расходы Python. Здесь шаг кадра — это один replay() графа,
а звук отдаётся кусками по мере генерации. Работает в .venv_tts.
"""
import threading

import numpy as np
import torch
import torch.nn.functional as F
from qwen_tts.core.models import modeling_qwen3_tts as M


class _Captured(Exception):
    pass


def _sample(logits, temperature, top_k):
    """top-k + temperature сэмплинг без синхронизации с CPU (Gumbel-max), можно внутри CUDA Graph."""
    vals, idx = logits.float().topk(top_k, dim=-1)
    u = torch.rand_like(vals).clamp_(1e-10, 1.0)
    choice = (vals / temperature - torch.log(-torch.log(u))).argmax(-1, keepdim=True)
    return idx.gather(-1, choice)  # (1, 1)


class FastQwenTTS:
    def __init__(self, tts, max_len=2048, steps_per_sync=4):
        self.tts = tts
        m = tts.model
        self.talker = m.talker
        self.cp = m.talker.code_predictor
        tc = m.config.talker_config
        self.tc = tc
        self.dev = self.talker.device
        self.dtype = self.talker.dtype
        self.max_len = max_len
        self.steps_per_sync = steps_per_sync
        self.groups = tc.num_code_groups
        self.eos = tc.codec_eos_token_id
        g = m.generate_config if hasattr(m, "generate_config") else {}
        self.temp = float(g.get("temperature", 0.9))
        self.top_k = int(g.get("top_k", 50))
        self.sub_temp = float(g.get("subtalker_temperature", 0.9))
        self.sub_top_k = int(g.get("subtalker_top_k", 50))
        self.rep = float(g.get("repetition_penalty", 1.05))

        a0 = self.talker.model.layers[0].self_attn
        self.hd = a0.head_dim
        self.kvh = tc.num_key_value_heads
        self.mrope = a0.rope_scaling["mrope_section"]
        self.mrope_inter = a0.rope_scaling.get("interleaved", False)
        cpa = self.cp.model.layers[0].self_attn
        self.cp_hd, self.cp_kvh = cpa.head_dim, self.cp.config.num_key_value_heads
        self.cp_len = self.groups + 1

        dev, dt = self.dev, self.dtype
        nl = len(self.talker.model.layers)
        self.kc = torch.zeros(nl, 1, self.kvh, max_len, self.hd, device=dev, dtype=dt)
        self.vc = torch.zeros_like(self.kc)
        cnl = len(self.cp.model.layers)
        self.ckc = torch.zeros(cnl, 1, self.cp_kvh, self.cp_len, self.cp_hd, device=dev, dtype=dt)
        self.cvc = torch.zeros_like(self.ckc)

        V, D = tc.vocab_size, tc.hidden_size
        self.s_code0 = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.s_hidden = torch.zeros(1, 1, D, device=dev, dtype=dt)
        self.s_step = torch.zeros(1, dtype=torch.long, device=dev)
        self.s_pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.s_text = torch.zeros(1, max_len, D, device=dev, dtype=dt)
        self.s_seen = torch.zeros(1, V, dtype=torch.bool, device=dev)
        self.s_bias = torch.zeros(1, V, device=dev)
        self.codes = torch.zeros(max_len, self.groups, dtype=torch.long, device=dev)

        sup = torch.zeros(1, V, device=dev)
        sup[:, V - 1024:] = float("-inf")
        sup[:, self.eos] = 0.0
        self.bias_ok = sup
        self.bias_no_eos = sup.clone()
        self.bias_no_eos[:, self.eos] = float("-inf")

        self.kpos = torch.arange(max_len, device=dev)
        self.cpos = torch.arange(self.cp_len, device=dev)
        self.cp_masks = [(self.cpos[None, :] <= torch.tensor([[0], [1]], device=dev))[None, None]] + \
                        [(self.cpos <= j + 1)[None, None, None, :] for j in range(1, self.groups - 1)]

        dec = tts.model.speech_tokenizer
        self.decoder = getattr(getattr(dec, "model", dec), "decoder")
        self.upsample = int(self.decoder.total_upsample)
        self.sr = 24000
        self.lock = threading.Lock()
        self.graph = None

    # ------------------------------------------------------------------ layers
    def _attn_layer(self, layer, x, cos, sin, kc, vc, qpos, mask, rope):
        a = layer.self_attn
        B, T, _ = x.shape
        h = layer.input_layernorm(x)
        q = a.q_norm(a.q_proj(h).view(B, T, -1, a.head_dim)).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).view(B, T, -1, a.head_dim)).transpose(1, 2)
        v = a.v_proj(h).view(B, T, -1, a.head_dim).transpose(1, 2)
        q, k = rope(q, k, cos, sin)
        kc.index_copy_(2, qpos, k)
        vc.index_copy_(2, qpos, v)
        rep = a.num_key_value_groups
        L = kc.shape[2]
        kk = kc[:, :, None].expand(B, kc.shape[1], rep, L, a.head_dim).reshape(B, -1, L, a.head_dim)
        vv = vc[:, :, None].expand(B, vc.shape[1], rep, L, a.head_dim).reshape(B, -1, L, a.head_dim)
        o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, scale=a.scaling)
        x = x + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        return x + layer.mlp(layer.post_attention_layernorm(x))

    def _talker(self, x, qpos):
        """x: (1,T,D) at positions qpos (T,). Returns normed hidden (1,T,D)."""
        pid = qpos.view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.talker.model.rotary_emb(x, pid)
        mask = (self.kpos[None, :] <= qpos[:, None])[None, None]
        rope = lambda q, k, c, s: M.apply_multimodal_rotary_pos_emb(q, k, c, s, self.mrope, self.mrope_inter)  # noqa: E731
        for i, layer in enumerate(self.talker.model.layers):
            x = self._attn_layer(layer, x, cos, sin, self.kc[i], self.vc[i], qpos, mask, rope)
        return self.talker.model.norm(x)

    def _cp(self, x, qpos, mask):
        cos, sin = self.cp.model.rotary_emb(x, qpos.view(1, -1))
        for i, layer in enumerate(self.cp.model.layers):
            x = self._attn_layer(layer, x, cos, sin, self.ckc[i], self.cvc[i], qpos, mask, M.apply_rotary_pos_emb)
        return self.cp.model.norm(x)

    # ------------------------------------------------------------------ one frame
    def _step(self):
        emb = self.talker.model.codec_embedding
        cemb = self.cp.model.codec_embedding
        proj = self.cp.small_to_mtp_projection
        e0 = emb(self.s_code0)                                   # (1,1,D)
        h = self._cp(proj(torch.cat([self.s_hidden, e0], 1)), self.cpos[:2], self.cp_masks[0])
        c = _sample(self.cp.lm_head[0](h[:, -1]), self.sub_temp, self.sub_top_k)
        sub = [c]
        acc = e0 + cemb[0](c)
        for j in range(1, self.groups - 1):
            h = self._cp(proj(cemb[j - 1](c)), self.cpos[j + 1:j + 2], self.cp_masks[j])
            c = _sample(self.cp.lm_head[j](h[:, -1]), self.sub_temp, self.sub_top_k)
            sub.append(c)
            acc = acc + cemb[j](c)
        frame = torch.cat([self.s_code0] + sub, dim=1)           # (1,16)
        self.codes.index_copy_(0, self.s_step, frame)

        x = acc + self.s_text.index_select(1, self.s_step)
        h = self._talker(x, self.s_pos)
        logits = self.talker.codec_head(h[:, -1]).float()
        pen = torch.where(logits < 0, logits * self.rep, logits / self.rep)
        logits = torch.where(self.s_seen, pen, logits) + self.s_bias
        nxt = _sample(logits, self.temp, self.top_k)
        self.s_seen.scatter_(1, nxt, True)
        self.s_code0.copy_(nxt)
        self.s_hidden.copy_(h)
        self.s_step.add_(1)
        self.s_pos.add_(1)

    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self.s_step.zero_(); self.s_pos.fill_(8)
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ public
    def _prepare(self, text, speaker, language):
        def fake(**kw):
            raise _Captured(kw)
        self.talker.generate = fake
        try:
            self.tts.generate_custom_voice(text=text, speaker=speaker, language=language)
        except _Captured as e:
            return e.args[0]
        finally:
            del self.talker.generate
        raise RuntimeError("talker.generate was not called")

    @torch.inference_mode()
    def stream(self, text, speaker, language, first_frames=3, chunk_frames=4, left_ctx=25, cancel=None):
        """Генератор float32 numpy-кусков звука 24 кГц. cancel: threading.Event — прервать генерацию.

        Куски по 4 кадра (0.33 c): пока LLM генерирует на той же видеокарте, TTS замедляется ~в 3 раза,
        и с большими кусками (12 кадров) в звуке появлялись паузы до 0.36 c; с 4 кадрами пауз нет.
        """
        with self.lock:
            if self.graph is None:
                self._capture()
            kw = self._prepare(text, speaker, language)
            emb = kw["inputs_embeds"].to(self.dtype)
            L = emb.shape[1]
            limit = self.max_len - L - 1
            trailing = kw["trailing_text_hidden"].to(self.dtype)
            self.s_text[:] = kw["tts_pad_embed"].to(self.dtype).view(1, 1, -1)
            n = min(trailing.shape[1], limit)
            self.s_text[:, :n] = trailing[:, :n]

            self.s_seen.zero_()
            self.s_bias.copy_(self.bias_no_eos)
            h = self._talker(emb, self.kpos[:L])
            logits = self.talker.codec_head(h[:, -1]).float() + self.s_bias
            nxt = _sample(logits, self.temp, self.top_k)
            self.s_seen.scatter_(1, nxt, True)
            self.s_code0.copy_(nxt)
            self.s_hidden.copy_(h[:, -1:])
            self.s_step.zero_()
            self.s_pos.fill_(L)

            done, emitted, total = False, 0, 0
            first = True
            while not done:
                need = first_frames if emitted == 0 else chunk_frames
                while total - emitted < need and not done:
                    k = min(self.steps_per_sync, limit - total)
                    if k <= 0 or (cancel is not None and cancel.is_set()):
                        done = True
                        break
                    for _ in range(k):
                        self.graph.replay()
                        if first:   # min_new_tokens=2: eos запрещён и на первом шаге
                            self.s_bias.copy_(self.bias_ok)
                            first = False
                    c0 = self.codes[total:total + k, 0].tolist()
                    if self.eos in c0:
                        total += c0.index(self.eos)
                        done = True
                    else:
                        total += k
                if cancel is not None and cancel.is_set():
                    return
                if total > emitted:
                    ctx = min(left_ctx, emitted)
                    codes = self.codes[emitted - ctx:total].clamp(min=0).T.unsqueeze(0)
                    wav = self.decoder(codes)[0, 0, ctx * self.upsample:]
                    emitted = total
                    yield wav.float().cpu().numpy()

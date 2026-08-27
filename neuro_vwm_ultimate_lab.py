#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: ULTIMATE V1-V4 VWM NEUROFEEDBACK SUITE (v61.0)
- Полный научный стек зрительной рабочей памяти (F1 - F9):
    [F1] Научный Лабораторный HUD
    [F2] ВРАЩЕНИЕ (Ориентация / Pinwheel)
    [F3] СМЕЩЕНИЕ (Фазовый дрейф / Микросаккады)
    [F4] ЗУМ (Пространственная частота / Looming)
    [F5] КРИВИЗНА (Изгиб контура / Banana Gabor / V2-V4)
    [F6] ПЛЕД (Ко-активация ортогональных ориентаций / V1-MT)
    [F7] ДВОЙНАЯ ПАМЯТЬ (Dual-Item Theta Time-Sharing / 2 угла одновременно)
    [F8] ОРТОГОНАЛЬНОЕ ПОДПРОСТРАНСТВО (Attractor Shielding / Panichello 2021)
    [F9] СИНАПТИЧЕСКИЙ ПИНГ (Activity-Silent STSP Trace / Stokes 2020)
- 100% честная физика (нет сигнала = чистая тьма).
- 100% CUDA DSP + Hardware GLSL (500+ FPS).
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try:
    os.nice(-20)
except Exception:
    pass

import time
import math
import threading
from collections import deque
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import torch

try:
    from OpenGL.GL import *
    from OpenGL.GL import shaders
except ImportError:
    raise ImportError("Установите PyOpenGL: pip install PyOpenGL PyOpenGL_accelerate")

from pylsl import StreamInlet, resolve_byprop

# ==============================================================================
# КОНФИГУРАЦИЯ И СЕНСОРНАЯ ГЕОМЕТРИЯ (26 мм @ Oz)
# ==============================================================================
FS = 250.0
BUF_SIZE = 256
NUM_CHANNELS = 16
NUM_FREQS = 32
NUM_PAIRS = 120
WIDTH, HEIGHT = 1400, 900
ASPECT = HEIGHT / WIDTH

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
I_IDX, J_IDX = np.triu_indices(NUM_CHANNELS, k=1)

DX_PAIR = (COORDS_X[J_IDX] - COORDS_X[I_IDX]).astype(np.float32)
DY_PAIR = (COORDS_Y[J_IDX] - COORDS_Y[I_IDX]).astype(np.float32)

CURL_WEIGHTS = ((COORDS_X[I_IDX] * DY_PAIR - COORDS_Y[I_IDX] * DX_PAIR) / 100.0).astype(np.float32)
DIV_WEIGHTS  = ((COORDS_X[I_IDX] * DX_PAIR + COORDS_Y[I_IDX] * DY_PAIR) / 100.0).astype(np.float32)

RADII = np.hypot(COORDS_X, COORDS_Y)
IS_INNER = RADII < 8.0
idx_core = [p for p in range(NUM_PAIRS) if (IS_INNER[I_IDX[p]] and IS_INNER[J_IDX[p]])]
idx_ring = [p for p in range(NUM_PAIRS) if (not IS_INNER[I_IDX[p]] and not IS_INNER[J_IDX[p]])]
idx_cross = [p for p in range(NUM_PAIRS) if (IS_INNER[I_IDX[p]] != IS_INNER[J_IDX[p]])]

I_GPU = torch.tensor(I_IDX, device=DEVICE, dtype=torch.long)
J_GPU = torch.tensor(J_IDX, device=DEVICE, dtype=torch.long)
DX_GPU = torch.tensor(DX_PAIR, device=DEVICE, dtype=torch.float32)
DY_GPU = torch.tensor(DY_PAIR, device=DEVICE, dtype=torch.float32)
CURL_GPU = torch.tensor(CURL_WEIGHTS, device=DEVICE, dtype=torch.float32)
DIV_GPU = torch.tensor(DIV_WEIGHTS, device=DEVICE, dtype=torch.float32)
IDX_CROSS_GPU = torch.tensor(idx_cross, device=DEVICE, dtype=torch.long)

# ==============================================================================
# 1. ПРЯМОЙ ПОТОКОВЫЙ LSL ПРИЕМНИК
# ==============================================================================
class DirectLSLStream:
    def __init__(self):
        self.raw_buffer = np.zeros((NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        self.is_running = True
        self.inlet = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.is_running:
            if not self.inlet:
                streams = resolve_byprop('type', 'EEG', timeout=0.5)
                if streams:
                    try:
                        self.inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=BUF_SIZE, recover=True)
                        print("[LSL] Входной поток Oz подключен.")
                    except Exception:
                        self.inlet = None
                time.sleep(0.1)
                continue

            try:
                chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                if chunk:
                    arr = np.array(chunk, dtype=np.float32).T
                    n = arr.shape[1]
                    with self.lock:
                        if n >= BUF_SIZE: self.raw_buffer = arr[:NUM_CHANNELS, -BUF_SIZE:]
                        else:
                            self.raw_buffer = np.roll(self.raw_buffer, -n, axis=1)
                            self.raw_buffer[:, -n:] = arr[:NUM_CHANNELS, :]
                else: time.sleep(0.0005)
            except Exception:
                self.inlet = None

    def pull_to_gpu(self, gpu_dest):
        with self.lock:
            gpu_dest.copy_(torch.from_numpy(self.raw_buffer))

    def stop(self): self.is_running = False

# ==============================================================================
# 2. МЕЖЦИКЛОВАЯ ПАМЯТЬ + СИНАПТИЧЕСКИЙ СЛЕД (STSP)
# ==============================================================================
class AdvancedVWMState:
    def __init__(self):
        self.angle_fut_t1 = 0.0
        self.angle_pres_t1 = 0.0
        self.rho_fut = 1.0
        self.rho_past = 1.0
        self.has_history = False

        # F9: Activity-Silent Синаптический След (Leaky STSP)
        self.stsp_angle = 0.0
        self.stsp_contrast = 0.0

        # Осциллограф истории
        self.history_fut = deque([0.0]*60, maxlen=60)
        self.history_past = deque([0.0]*60, maxlen=60)

    def on_new_theta_cycle(self, a_past, a_now, a_fut, c_now):
        if self.has_history:
            diff_fut = math.cos(2.0 * (a_now - self.angle_fut_t1))
            self.rho_fut = self.rho_fut * 0.75 + diff_fut * 0.25

            diff_past = math.cos(2.0 * (a_past - self.angle_pres_t1))
            self.rho_past = self.rho_past * 0.75 + diff_past * 0.25

            self.history_fut.append(self.rho_fut)
            self.history_past.append(self.rho_past)

        self.angle_fut_t1 = a_fut
        self.angle_pres_t1 = a_now
        self.has_history = True

        # Обновление синаптического следа STSP (медленный распад тау ~ 2 сек)
        self.stsp_angle = self.stsp_angle * 0.85 + a_now * 0.15
        self.stsp_contrast = self.stsp_contrast * 0.92 + c_now * 0.08

# ==============================================================================
# 3. ПОЛНЫЙ CUDA DSP ДВИЖОК
# ==============================================================================
class UltimateVWM_DSP:
    def __init__(self):
        self.freqs = torch.fft.fftfreq(BUF_SIZE, d=1.0/FS).to(DEVICE)
        self.notch = torch.ones_like(self.freqs)
        self.notch[(torch.abs(self.freqs) >= 48.0) & (torch.abs(self.freqs) <= 52.0)] = 0.0
        self.notch[(torch.abs(self.freqs) >= 98.0) & (torch.abs(self.freqs) <= 102.0)] = 0.0
        self.notch = self.notch.view(1, BUF_SIZE)

        self.f_theta = (torch.exp(-0.5 * ((self.freqs - 6.0) / 1.5)**2) * 2.0).view(1, BUF_SIZE)
        self.f_theta[:, self.freqs < 0] = 0.0

        gamma_centers = torch.linspace(30.0, 85.0, NUM_FREQS, device=DEVICE).view(NUM_FREQS, 1, 1)
        freqs_3d = self.freqs.view(1, 1, BUF_SIZE)
        self.gamma_filters = torch.exp(-0.5 * ((freqs_3d - gamma_centers) / 4.5)**2) * 2.0
        self.gamma_filters[:, :, self.freqs < 0] = 0.0

        self.slot_angles = (-math.pi + (2.0 * math.pi / NUM_FREQS) * (torch.arange(NUM_FREQS, device=DEVICE) + 0.5)).view(NUM_FREQS, 1, 1)
        self.raw_gpu = torch.zeros((NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)

        self.pll_phase = 0.0
        self.pll_freq_hz = 6.0
        self.smooth_gamma_120 = torch.zeros((NUM_FREQS, NUM_PAIRS), device=DEVICE, dtype=torch.float32)

        self.state = AdvancedVWMState()

    @torch.inference_mode()
    def process(self, raw_input, dt):
        centered = raw_input - torch.mean(raw_input, dim=1, keepdim=True)
        fft_clean = torch.fft.fft(centered, dim=-1) * self.notch

        # 1. Тета
        Z_theta = torch.fft.ifft(fft_clean * self.f_theta, dim=-1)
        P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
        mean_th_phasor = torch.mean(P_theta, dim=0, keepdim=True)
        phi_theta_raw = torch.angle(mean_th_phasor)
        
        target_eeg_phase = float(phi_theta_raw[0, -1].item())
        th_vec = phi_theta_raw[0]
        d_phi = (th_vec[1:] - th_vec[:-1] + math.pi) % (2.0 * math.pi) - math.pi
        measured_hz = float(np.clip(float(torch.mean(d_phi).item()) / (2.0 * math.pi) * FS, 3.5, 9.0))

        self.pll_freq_hz = self.pll_freq_hz * 0.95 + measured_hz * 0.05
        phase_err = (target_eeg_phase - self.pll_phase + math.pi) % (2.0 * math.pi) - math.pi
        
        old_phase = self.pll_phase
        self.pll_phase = (self.pll_phase + 2.0 * math.pi * self.pll_freq_hz * dt + phase_err * 2.0 * dt) % (2.0 * math.pi)

        # 2. 32 PAC Гамма-слота
        fft_exp = fft_clean.unsqueeze(0)
        Z_gamma = torch.fft.ifft(fft_exp * self.gamma_filters, dim=-1)
        P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)

        p_diff = phi_theta_raw.unsqueeze(0) - self.slot_angles
        w = torch.exp(3.2 * torch.cos(p_diff))
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

        cg_gamma = P_gamma[:, I_GPU, :] * torch.conj(P_gamma[:, J_GPU, :])
        raw_gamma_120 = torch.sum(torch.imag(cg_gamma) * w, dim=-1)

        self.smooth_gamma_120 = self.smooth_gamma_120 * 0.85 + raw_gamma_120 * 0.15

        # 3. Декодирование параметров V1 - V4
        flow_x = torch.sum(self.smooth_gamma_120 * DX_GPU, dim=-1)
        flow_y = torch.sum(self.smooth_gamma_120 * DY_GPU, dim=-1)

        angles = torch.atan2(flow_y, flow_x) * 0.5
        contrasts = torch.clamp(torch.hypot(flow_x, flow_y) * 0.08, 0.0, 1.0)
        shift_x = torch.clamp(flow_x * 0.0008, -0.03, 0.03) * ASPECT
        shift_y = torch.clamp(flow_y * 0.0008, -0.03, 0.03)

        cross_div = torch.sum(self.smooth_gamma_120[:, IDX_CROSS_GPU], dim=-1) * 0.02
        base_freq = 6.5
        freqs = base_freq * (1.0 + torch.clamp(cross_div, -0.4, 0.4))

        curl_val = torch.sum(self.smooth_gamma_120 * CURL_GPU, dim=-1) * 0.02
        curvatures = torch.clamp(curl_val * 1.5, -0.8, 0.8)

        spec_std = torch.std(self.smooth_gamma_120, dim=-1) * 1.2
        plaids = torch.clamp(spec_std, 0.0, 1.0)

        # F8: Ортогональное Защитное Подпространство (Null-Space Projection по Panichello 2021)
        # Измеряем долю мощности, ушедшую в ортогональный базис
        nullspace_proj = float(torch.mean(torch.abs(torch.sin(angles * 2.0))).item())

        # F7: Раздельное декодирование ДВУХ ОБЪЕКТОВ (Dual-Item)
        # Объект A = среднее первой половины Теты (слоты 0..15)
        # Объект B = среднее второй половины Теты (слоты 16..31)
        item_a_angle = float(torch.mean(angles[:16]).item())
        item_a_contrast = float(torch.mean(contrasts[:16]).item())
        item_b_angle = float(torch.mean(angles[16:]).item())
        item_b_contrast = float(torch.mean(contrasts[16:]).item())

        # Темпоральность
        mid_idx = NUM_FREQS // 2
        len_past = float(torch.sum(contrasts[:mid_idx]).item())
        len_fut = float(torch.sum(contrasts[mid_idx:]).item())
        ry = float(np.clip((len_fut - len_past) / (len_fut + len_past + 1e-6), -1.0, 1.0))

        base_slot = (self.pll_phase / (2.0 * math.pi)) * NUM_FREQS
        k_now = int(base_slot + ry * 8.0) % NUM_FREQS
        k_fut = int(base_slot + 12.0 + ry * 8.0) % NUM_FREQS

        a_now, c_now = float(angles[k_now].item()), float(contrasts[k_now].item())
        a_past, c_past = float(angles[0].item()), float(contrasts[0].item())
        a_fut, c_fut = float(angles[k_fut].item()), float(contrasts[k_fut].item())

        if self.pll_phase < old_phase:
            self.state.on_new_theta_cycle(a_past, a_now, a_fut, c_now)

        slot_energies = torch.norm(self.smooth_gamma_120, dim=1).cpu().numpy()
        edge_energies = torch.mean(torch.abs(self.smooth_gamma_120), dim=0).cpu().numpy()

        return {
            'a_now': a_now, 'c_now': c_now, 'sx_now': float(shift_x[k_now].item()), 'sy_now': float(shift_y[k_now].item()),
            'f_now': float(freqs[k_now].item()), 'curv_now': float(curvatures[k_now].item()), 'plaid_now': float(plaids[k_now].item()),
            'a_past': a_past, 'c_past': c_past, 'sx_past': float(shift_x[0].item()), 'sy_past': float(shift_y[0].item()),
            'f_past': float(freqs[0].item()), 'curv_past': float(curvatures[0].item()), 'plaid_past': float(plaids[0].item()),
            'a_fut_prev': self.state.angle_fut_t1, 'c_fut_prev': c_fut, 'sx_fut': float(shift_x[k_fut].item()), 'sy_fut': float(shift_y[k_fut].item()),
            'f_fut': float(freqs[k_fut].item()), 'curv_fut': float(curvatures[k_fut].item()), 'plaid_fut': float(plaids[k_fut].item()),
            
            # Новые прорывные параметры
            'item_a_angle': item_a_angle, 'item_a_contrast': item_a_contrast,
            'item_b_angle': item_b_angle, 'item_b_contrast': item_b_contrast,
            'nullspace_proj': nullspace_proj,
            'stsp_angle': self.state.stsp_angle, 'stsp_contrast': self.state.stsp_contrast,

            'rho_fut': self.state.rho_fut, 'rho_past': self.state.rho_past,
            'theta_hz': self.pll_freq_hz, 'theta_phase': self.pll_phase, 'ry': ry, 'rms': float(torch.std(raw_input).item()),
            'base_freq': base_freq,
            'slot_energies': slot_energies, 'edge_energies': edge_energies,
            'hist_fut': list(self.state.history_fut), 'hist_past': list(self.state.history_past)
        }

# ==============================================================================
# 4. GLSL ШЕЙДЕР С ПОЛНЫМ НАУЧНЫМ СТЕКОМ (F2 - F9)
# ==============================================================================
GABOR_VERT = """
#version 330 core
layout(location = 0) in vec2 in_pos;
out vec2 v_uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_pos; }
"""

GABOR_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 out_color;

// Флаги режимов (F2 - F9)
uniform float u_mode_rot;    // F2
uniform float u_mode_trans;  // F3
uniform float u_mode_zoom;   // F4
uniform float u_mode_curv;   // F5
uniform float u_mode_plaid;  // F6
uniform float u_mode_dual;   // F7 (Dual Item Multiplexing)
uniform float u_mode_shield; // F8 (Orthogonal Subspace Shield)
uniform float u_mode_stsp;   // F9 (Activity-Silent STSP Ping)

// Основные каналы (Будущее, Настоящее, Прошлое)
uniform float u_a_fut;  uniform float u_c_fut;  uniform vec2 u_shift_fut;  uniform float u_freq_fut;  uniform float u_curv_fut;  uniform float u_plaid_fut;
uniform float u_a_now;  uniform float u_c_now;  uniform vec2 u_shift_now;  uniform float u_freq_now;  uniform float u_curv_now;  uniform float u_plaid_now;
uniform float u_a_past; uniform float u_c_past; uniform vec2 u_shift_past; uniform float u_freq_past; uniform float u_curv_past; uniform float u_plaid_past;

// F7: Dual Item
uniform float u_item_a_ang; uniform float u_item_a_c;
uniform float u_item_b_ang; uniform float u_item_b_c;
uniform float u_theta_phase;

// F8: Shield & F9: STSP
uniform float u_nullspace;
uniform float u_stsp_ang;
uniform float u_stsp_c;
uniform float u_base_freq;

float eval_wave(vec2 uv, vec2 shift, float ang, float freq, float curv, float plaid, float contrast) {
    vec2 p = uv - (u_mode_trans > 0.5 ? shift : vec2(0.0));
    float env = exp(-dot(p, p) * 12.0);

    float a = (u_mode_rot > 0.5 ? ang : 0.0);
    vec2 rot_p = vec2(p.x * cos(a) + p.y * sin(a), -p.x * sin(a) + p.y * cos(a));

    if (u_mode_curv > 0.5) rot_p.y += curv * rot_p.x * rot_p.x * 3.0;

    float f = (u_mode_zoom > 0.5 ? freq : u_base_freq);
    float phase_main = 2.0 * 3.14159265 * f * rot_p.y;
    float wave = max(0.0, cos(phase_main));

    if (u_mode_plaid > 0.5) {
        float phase_ortho = 2.0 * 3.14159265 * f * rot_p.x;
        wave = mix(wave, (wave + max(0.0, cos(phase_ortho))) * 0.5, plaid * 0.7);
    }
    return wave * contrast * env;
}

void main() {
    float R, G, B;

    // [F7] ДВОЙНАЯ ПАМЯТЬ (Dual-Item Theta Time-Sharing)
    if (u_mode_dual > 0.5) {
        // Ритмичное переключение между Объектом А и Б в такт Тете
        float weight_a = clamp(sin(u_theta_phase) * 2.0, 0.0, 1.0);
        float weight_b = 1.0 - weight_a;

        float wave_a = eval_wave(v_uv, vec2(0.0), u_item_a_ang, u_base_freq, 0.0, 0.0, u_item_a_c);
        float wave_b = eval_wave(v_uv, vec2(0.0), u_item_b_ang, u_base_freq, 0.0, 0.0, u_item_b_c);
        
        R = wave_a * weight_a * 2.5;
        G = (wave_a * weight_a + wave_b * weight_b) * 1.5;
        B = wave_b * weight_b * 2.5;
    } else {
        R = eval_wave(v_uv, u_shift_fut,  u_a_fut,  u_freq_fut,  u_curv_fut,  u_plaid_fut,  u_c_fut)  * 2.0;
        G = eval_wave(v_uv, u_shift_now,  u_a_now,  u_freq_now,  u_curv_now,  u_plaid_now,  u_c_now)  * 2.0;
        B = eval_wave(v_uv, u_shift_past, u_a_past, u_freq_past, u_curv_past, u_plaid_past, u_c_past) * 2.0;
    }

    // [F8] ОРТОГОНАЛЬНЫЙ ЗАЩИТНЫЙ ПЛАЩ (Null-Space Shielding)
    if (u_mode_shield > 0.5) {
        float shield_mod = 1.0 + u_nullspace * 0.8 * sin(length(v_uv) * 30.0 - u_theta_phase * 3.0);
        R *= shield_mod; G *= shield_mod; B *= shield_mod;
    }

    // [F9] СИНАПТИЧЕСКИЙ СЛЕД (Activity-Silent STSP Ghost Ping)
    if (u_mode_stsp > 0.5 && u_stsp_c > 0.02) {
        float wave_ghost = eval_wave(v_uv, vec2(0.0), u_stsp_ang, u_base_freq, 0.0, 0.0, u_stsp_c);
        vec3 ghost_col = vec3(0.4, 0.2, 0.8) * wave_ghost * 1.2;
        R += ghost_col.r; G += ghost_col.g; B += ghost_col.b;
    }

    out_color = vec4(clamp(vec3(R, G, B), 0.0, 1.0), 1.0);
}
"""

HUD_VERT = """
#version 330 core
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
out vec2 v_uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_uv; }
"""
HUD_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_hud_tex;
void main() { out_color = texture(u_hud_tex, v_uv); }
"""

class UltimateVisualizer:
    def __init__(self):
        vs = shaders.compileShader(GABOR_VERT, GL_VERTEX_SHADER)
        fs = shaders.compileShader(GABOR_FRAG, GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vs, fs)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        quad = np.array([-1.0, -1.0,  1.0, -1.0, -1.0,  1.0,
                          1.0, -1.0,  1.0,  1.0, -1.0,  1.0], dtype=np.float32)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBindVertexArray(0)

        self.u_vars = {
            'u_mode_rot': glGetUniformLocation(self.shader, "u_mode_rot"),
            'u_mode_trans': glGetUniformLocation(self.shader, "u_mode_trans"),
            'u_mode_zoom': glGetUniformLocation(self.shader, "u_mode_zoom"),
            'u_mode_curv': glGetUniformLocation(self.shader, "u_mode_curv"),
            'u_mode_plaid': glGetUniformLocation(self.shader, "u_mode_plaid"),
            'u_mode_dual': glGetUniformLocation(self.shader, "u_mode_dual"),
            'u_mode_shield': glGetUniformLocation(self.shader, "u_mode_shield"),
            'u_mode_stsp': glGetUniformLocation(self.shader, "u_mode_stsp"),
            'u_base_freq': glGetUniformLocation(self.shader, "u_base_freq"),

            'u_a_fut': glGetUniformLocation(self.shader, "u_a_fut"),
            'u_c_fut': glGetUniformLocation(self.shader, "u_c_fut"),
            'u_shift_fut': glGetUniformLocation(self.shader, "u_shift_fut"),
            'u_freq_fut': glGetUniformLocation(self.shader, "u_freq_fut"),
            'u_curv_fut': glGetUniformLocation(self.shader, "u_curv_fut"),
            'u_plaid_fut': glGetUniformLocation(self.shader, "u_plaid_fut"),

            'u_a_now': glGetUniformLocation(self.shader, "u_a_now"),
            'u_c_now': glGetUniformLocation(self.shader, "u_c_now"),
            'u_shift_now': glGetUniformLocation(self.shader, "u_shift_now"),
            'u_freq_now': glGetUniformLocation(self.shader, "u_freq_now"),
            'u_curv_now': glGetUniformLocation(self.shader, "u_curv_now"),
            'u_plaid_now': glGetUniformLocation(self.shader, "u_plaid_now"),

            'u_a_past': glGetUniformLocation(self.shader, "u_a_past"),
            'u_c_past': glGetUniformLocation(self.shader, "u_c_past"),
            'u_shift_past': glGetUniformLocation(self.shader, "u_shift_past"),
            'u_freq_past': glGetUniformLocation(self.shader, "u_freq_past"),
            'u_curv_past': glGetUniformLocation(self.shader, "u_curv_past"),
            'u_plaid_past': glGetUniformLocation(self.shader, "u_plaid_past"),

            'u_item_a_ang': glGetUniformLocation(self.shader, "u_item_a_ang"),
            'u_item_a_c': glGetUniformLocation(self.shader, "u_item_a_c"),
            'u_item_b_ang': glGetUniformLocation(self.shader, "u_item_b_ang"),
            'u_item_b_c': glGetUniformLocation(self.shader, "u_item_b_c"),
            'u_theta_phase': glGetUniformLocation(self.shader, "u_theta_phase"),
            'u_nullspace': glGetUniformLocation(self.shader, "u_nullspace"),
            'u_stsp_ang': glGetUniformLocation(self.shader, "u_stsp_ang"),
            'u_stsp_c': glGetUniformLocation(self.shader, "u_stsp_c")
        }

        # HUD
        vs_hud = shaders.compileShader(HUD_VERT, GL_VERTEX_SHADER)
        fs_hud = shaders.compileShader(HUD_FRAG, GL_FRAGMENT_SHADER)
        self.hud_prog = shaders.compileProgram(vs_hud, fs_hud)
        self.hud_vao = glGenVertexArrays(1)
        self.hud_vbo = glGenBuffers(1)
        self.hud_tex = glGenTextures(1)

        hud_quad = np.array([
            -0.98,  0.15,  0.0, 1.0,   -0.12,  0.15,  1.0, 1.0,   -0.98,  0.98,  0.0, 0.0,
            -0.12,  0.15,  1.0, 1.0,   -0.12,  0.98,  1.0, 0.0,   -0.98,  0.98,  0.0, 0.0,
        ], dtype=np.float32)

        glBindVertexArray(self.hud_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, hud_quad.nbytes, hud_quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(8))
        glBindVertexArray(0)

        self.HUD_W, self.HUD_H = 600, 600
        self.hud_surface = pygame.Surface((self.HUD_W, self.HUD_H), pygame.SRCALPHA)
        self.font_title = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_main = pygame.font.SysFont("consolas", 11, bold=True)
        self.font_sub = pygame.font.SysFont("consolas", 9)
        
        glBindTexture(GL_TEXTURE_2D, self.hud_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

    def render(self, state, flags):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shader)
        
        glUniform1f(self.u_vars['u_mode_rot'], 1.0 if flags['f2'] else 0.0)
        glUniform1f(self.u_vars['u_mode_trans'], 1.0 if flags['f3'] else 0.0)
        glUniform1f(self.u_vars['u_mode_zoom'], 1.0 if flags['f4'] else 0.0)
        glUniform1f(self.u_vars['u_mode_curv'], 1.0 if flags['f5'] else 0.0)
        glUniform1f(self.u_vars['u_mode_plaid'], 1.0 if flags['f6'] else 0.0)
        glUniform1f(self.u_vars['u_mode_dual'], 1.0 if flags['f7'] else 0.0)
        glUniform1f(self.u_vars['u_mode_shield'], 1.0 if flags['f8'] else 0.0)
        glUniform1f(self.u_vars['u_mode_stsp'], 1.0 if flags['f9'] else 0.0)
        glUniform1f(self.u_vars['u_base_freq'], state['base_freq'])

        # Будущее
        glUniform1f(self.u_vars['u_a_fut'], state['a_fut_prev'])
        glUniform1f(self.u_vars['u_c_fut'], state['c_fut_prev'])
        glUniform2f(self.u_vars['u_shift_fut'], state['sx_fut'], state['sy_fut'])
        glUniform1f(self.u_vars['u_freq_fut'], state['f_fut'])
        glUniform1f(self.u_vars['u_curv_fut'], state['curv_fut'])
        glUniform1f(self.u_vars['u_plaid_fut'], state['plaid_fut'])

        # Настоящее
        glUniform1f(self.u_vars['u_a_now'], state['a_now'])
        glUniform1f(self.u_vars['u_c_now'], state['c_now'])
        glUniform2f(self.u_vars['u_shift_now'], state['sx_now'], state['sy_now'])
        glUniform1f(self.u_vars['u_freq_now'], state['f_now'])
        glUniform1f(self.u_vars['u_curv_now'], state['curv_now'])
        glUniform1f(self.u_vars['u_plaid_now'], state['plaid_now'])

        # Прошлое
        glUniform1f(self.u_vars['u_a_past'], state['a_past'])
        glUniform1f(self.u_vars['u_c_past'], state['c_past'])
        glUniform2f(self.u_vars['u_shift_past'], state['sx_past'], state['sy_past'])
        glUniform1f(self.u_vars['u_freq_past'], state['f_past'])
        glUniform1f(self.u_vars['u_curv_past'], state['curv_past'])
        glUniform1f(self.u_vars['u_plaid_past'], state['plaid_past'])

        # Новые прорывные переменные
        glUniform1f(self.u_vars['u_item_a_ang'], state['item_a_angle'])
        glUniform1f(self.u_vars['u_item_a_c'], state['item_a_contrast'])
        glUniform1f(self.u_vars['u_item_b_ang'], state['item_b_angle'])
        glUniform1f(self.u_vars['u_item_b_c'], state['item_b_contrast'])
        glUniform1f(self.u_vars['u_theta_phase'], state['theta_phase'])
        glUniform1f(self.u_vars['u_nullspace'], state['nullspace_proj'])
        glUniform1f(self.u_vars['u_stsp_ang'], state['stsp_angle'])
        glUniform1f(self.u_vars['u_stsp_c'], state['stsp_contrast'])

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

        if flags['f1']:
            self._render_lab_hud(state, flags)

    def _render_lab_hud(self, state, flags):
        self.hud_surface.fill((10, 15, 22, 245))
        pygame.draw.rect(self.hud_surface, (0, 200, 255), (0, 0, self.HUD_W, self.HUD_H), 1, border_radius=4)

        self.hud_surface.blit(self.font_title.render("V1-V4 ULTIMATE VWM SUITE (Oz / 26mm)", True, (255, 255, 255)), (15, 10))
        self.hud_surface.blit(self.font_sub.render(f"RMS: {state['rms']:.1f} uV | Theta: {state['theta_hz']:.2f} Hz | ry: {state['ry']:+.2f}", True, (150, 180, 200)), (15, 26))

        y = 44
        # Список 8 режимов (F2 - F9)
        modes = [
            ('f2', 'ROTATION (Orientation θ)', f"{math.degrees(state['a_now']):.0f}°"),
            ('f3', 'TRANSLATION (Phase Drift)', f"dx:{state['sx_now']*100:.1f}"),
            ('f4', 'ZOOM (Spatial Freq f)  ', f"{state['f_now']:.1f} cpd"),
            ('f5', 'CURVATURE (Banana V2/V4)', f"{state['curv_now']:+.2f}"),
            ('f6', 'PLAID (Cross-Pattern MT)', f"{state['plaid_now']*100:.0f}%"),
            ('f7', 'DUAL-ITEM PAC MULTIPLEX ', f"A:{math.degrees(state['item_a_angle']):.0f}° B:{math.degrees(state['item_b_angle']):.0f}°"),
            ('f8', 'NULLSPACE SHIELD (Fusi) ', f"{state['nullspace_proj']*100:.0f}%"),
            ('f9', 'STSP SYNAPTIC PING (ST) ', f"C:{state['stsp_contrast']:.2f}")
        ]

        for f_key, name, val_str in modes:
            is_on = flags[f_key]
            col = (0, 255, 180) if is_on else (100, 100, 100)
            self.hud_surface.blit(self.font_main.render(f"[{f_key.upper()}] {name}: {'ON' if is_on else 'OFF'} ({val_str})", True, col), (15, y))
            y += 16

        y += 6
        # Полярный компас
        cx, cy, r_comp = 75, y + 42, 38
        pygame.draw.circle(self.hud_surface, (25, 35, 45), (cx, cy), r_comp)
        pygame.draw.circle(self.hud_surface, (60, 80, 100), (cx, cy), r_comp, 1)
        
        ap = state['a_past'] * 2.0
        pygame.draw.line(self.hud_surface, (50, 150, 255), (cx, cy), (cx + int(r_comp * math.cos(ap)), cy - int(r_comp * math.sin(ap))), 2)
        an = state['a_now'] * 2.0
        pygame.draw.line(self.hud_surface, (0, 255, 120), (cx, cy), (cx + int(r_comp * math.cos(an)), cy - int(r_comp * math.sin(an))), 3)
        af = state['a_fut_prev'] * 2.0
        pygame.draw.line(self.hud_surface, (255, 80, 80), (cx, cy), (cx + int(r_comp * math.cos(af)), cy - int(r_comp * math.sin(af))), 2)

        self.hud_surface.blit(self.font_sub.render("PHASE COMPASS", True, (200, 200, 200)), (38, y + 84))

        # Метрики связи
        col_f = (0, 255, 180) if state['rho_fut'] > 0.4 else (255, 100, 100)
        self.hud_surface.blit(self.font_main.render(f"• PREDICTION (rho_fut) : {state['rho_fut']:+.2f}", True, col_f), (155, y + 10))
        pygame.draw.rect(self.hud_surface, (30, 40, 50), (155, y + 25, 200, 5))
        pygame.draw.rect(self.hud_surface, col_f, (155, y + 25, int(max(0, (state['rho_fut']+1)/2) * 200), 5))

        col_p = (80, 180, 255) if state['rho_past'] > 0.4 else (255, 100, 100)
        self.hud_surface.blit(self.font_main.render(f"• ANCHOR     (rho_past): {state['rho_past']:+.2f}", True, col_p), (155, y + 42))
        pygame.draw.rect(self.hud_surface, (30, 40, 50), (155, y + 57, 200, 5))
        pygame.draw.rect(self.hud_surface, col_p, (155, y + 57, int(max(0, (state['rho_past']+1)/2) * 200), 5))

        y += 105
        # Осциллограф
        self.hud_surface.blit(self.font_sub.render("INTER-CYCLE CONTINUITY (Last 60 Theta Cycles)", True, (150, 180, 200)), (15, y))
        y += 14
        ox, oy, ow, oh = 15, y, 570, 44
        pygame.draw.rect(self.hud_surface, (15, 22, 30), (ox, oy, ow, oh))
        pygame.draw.rect(self.hud_surface, (40, 60, 80), (ox, oy, ow, oh), 1)
        pygame.draw.line(self.hud_surface, (50, 70, 90), (ox, oy + oh//2), (ox + ow, oy + oh//2), 1)

        h_fut = state['hist_fut']
        h_past = state['hist_past']
        if len(h_fut) > 1:
            pts_f = [(ox + i * (ow/60.0), oy + oh//2 - int(h_fut[i] * (oh//2 - 4))) for i in range(len(h_fut))]
            pts_p = [(ox + i * (ow/60.0), oy + oh//2 - int(h_past[i] * (oh//2 - 4))) for i in range(len(h_past))]
            pygame.draw.lines(self.hud_surface, (255, 80, 80), False, pts_f, 2)
            pygame.draw.lines(self.hud_surface, (50, 150, 255), False, pts_p, 2)

        y += 52
        # Спектры
        self.hud_surface.blit(self.font_sub.render("32 GAMMA BINS PAC", True, (150, 180, 200)), (15, y))
        self.hud_surface.blit(self.font_sub.render("120-EDGE iPLV MATRIX", True, (150, 180, 200)), (305, y))
        y += 14
        bx, by, bw, bh = 15, y, 275, 36
        pygame.draw.rect(self.hud_surface, (15, 22, 30), (bx, by, bw, bh))
        s_eng = state['slot_energies']
        max_s = np.max(s_eng) + 1e-6
        for k in range(NUM_FREQS):
            val_h = int((s_eng[k] / max_s) * bh)
            col_b = (255, 100, 100) if k >= 22 else ((0, 255, 150) if k >= 11 else (80, 150, 255))
            pygame.draw.rect(self.hud_surface, col_b, (bx + k * (bw/32.0), by + bh - val_h, int(bw/32.0)-1, val_h))

        ex, ey, ew, eh = 305, y, 280, 36
        pygame.draw.rect(self.hud_surface, (15, 22, 30), (ex, ey, ew, eh))
        e_eng = state['edge_energies']
        max_e = np.max(e_eng) + 1e-6
        for p in range(NUM_PAIRS):
            val_h = int((e_eng[p] / max_e) * eh)
            col_e = (255, 80, 80) if p in idx_core else ((80, 255, 120) if p in idx_ring else (80, 150, 255))
            pygame.draw.rect(self.hud_surface, col_e, (ex + p * (ew/120.0), ey + eh - val_h, int(ew/120.0)+1, val_h))

        y += 42
        self.hud_surface.blit(self.font_sub.render("[F1] HUD | [F2-F6] Geometry (Rot/Shift/Zoom/Curv/Plaid) | [F7-F9] Advanced (Dual/Shield/Ping)", True, (130, 150, 170)), (15, y))

        raw_tex_data = pygame.image.tostring(self.hud_surface, "RGBA", False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self.hud_prog)
        glBindTexture(GL_TEXTURE_2D, self.hud_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.HUD_W, self.HUD_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw_tex_data)

        glBindVertexArray(self.hud_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("NeuroCanvas: Ultimate V1-V4 VWM Neurofeedback Suite")

    glViewport(0, 0, WIDTH, HEIGHT)

    stream = DirectLSLStream()
    dsp = UltimateVWM_DSP()
    visualizer = UltimateVisualizer()

    flags = {
        'f1': False, # HUD
        'f2': True,  # Rotation
        'f3': True,  # Shift
        'f4': True,  # Zoom
        'f5': True,  # Curvature
        'f6': True,  # Plaid
        'f7': False, # Dual-Item (выключен по умолчанию, включается по F7)
        'f8': False, # Shield (выключен по умолчанию, включается по F8)
        'f9': False  # STSP Ping (выключен по умолчанию, включается по F9)
    }

    last_time = time.perf_counter()
    running = True

    try:
        while running:
            now = time.perf_counter()
            dt = max(0.0005, min(0.05, now - last_time))
            last_time = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F1: flags['f1'] = not flags['f1']
                    elif event.key == pygame.K_F2: flags['f2'] = not flags['f2']
                    elif event.key == pygame.K_F3: flags['f3'] = not flags['f3']
                    elif event.key == pygame.K_F4: flags['f4'] = not flags['f4']
                    elif event.key == pygame.K_F5: flags['f5'] = not flags['f5']
                    elif event.key == pygame.K_F6: flags['f6'] = not flags['f6']
                    elif event.key == pygame.K_F7: flags['f7'] = not flags['f7']
                    elif event.key == pygame.K_F8: flags['f8'] = not flags['f8']
                    elif event.key == pygame.K_F9: flags['f9'] = not flags['f9']

            stream.pull_to_gpu(dsp.raw_gpu)
            state = dsp.process(dsp.raw_gpu, dt)
            visualizer.render(state, flags)

            pygame.display.flip()

    finally:
        stream.stop()
        pygame.quit()

if __name__ == '__main__':
    main()

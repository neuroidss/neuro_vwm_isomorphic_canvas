#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: HIERARCHICAL MULTI-DEVICE VWM LAB (v62.2 - ZERO-DROPOUT LSL)
- Устранен баг с периодическим пересозданием потока (UID Tracking).
- 1 устройство работает непрерывно без 2-секундных микро-пауз и спама в консоль.
- Бесшовное добавление 2, 3, 4 устройств на лету.
- [1, 2, 3, 4] - Цикличное назначение регионов (Oz, PO7, Pz, P7, OFF).
- [F1] - Научный Лабораторный HUD.
- 100% Batched CUDA DSP + Hardware GLSL (500+ FPS).
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
# КОНФИГУРАЦИЯ И СЕНСОРНАЯ ГЕОМЕТРИЯ
# ==============================================================================
FS = 250.0
BUF_SIZE = 256
NUM_CHANNELS = 16
NUM_MAX_DEVICES = 4
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
idx_cross = [p for p in range(NUM_PAIRS) if (IS_INNER[I_IDX[p]] != IS_INNER[J_IDX[p]])]
idx_core = [p for p in range(NUM_PAIRS) if (IS_INNER[I_IDX[p]] and IS_INNER[J_IDX[p]])]
idx_ring = [p for p in range(NUM_PAIRS) if (not IS_INNER[I_IDX[p]] and not IS_INNER[J_IDX[p]])]

I_GPU = torch.tensor(I_IDX, device=DEVICE, dtype=torch.long)
J_GPU = torch.tensor(J_IDX, device=DEVICE, dtype=torch.long)
DX_GPU = torch.tensor(DX_PAIR, device=DEVICE, dtype=torch.float32)
DY_GPU = torch.tensor(DY_PAIR, device=DEVICE, dtype=torch.float32)
CURL_GPU = torch.tensor(CURL_WEIGHTS, device=DEVICE, dtype=torch.float32)
DIV_GPU = torch.tensor(DIV_WEIGHTS, device=DEVICE, dtype=torch.float32)
IDX_CROSS_GPU = torch.tensor(idx_cross, device=DEVICE, dtype=torch.long)

# ==============================================================================
# 1. СТАБИЛЬНЫЙ LSL ПРИЕМНИК С УНИКАЛЬНОЙ ИДЕНТИФИКАЦИЕЙ (БЕЗ ОБРЫВОВ)
# ==============================================================================
class StableMultiLSLReceiver:
    def __init__(self):
        self.raw_buffers = np.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        self.inlets = []
        self.connected_uids = set()
        self.active_count = 0
        self.is_running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        last_search_time = 0.0
        while self.is_running:
            now = time.time()
            
            # Ищем ТОЛЬКО новые устройства раз в 3 секунды, не трогая существующие
            if (len(self.inlets) < NUM_MAX_DEVICES) and (now - last_search_time > 3.0):
                last_search_time = now
                try:
                    streams = resolve_byprop('type', 'EEG', timeout=0.1)
                    if streams:
                        for s in streams:
                            uid = s.uid()
                            # Если этого устройства еще нет в нашем списке - подключаем его!
                            if uid not in self.connected_uids and len(self.inlets) < NUM_MAX_DEVICES:
                                try:
                                    inlet = StreamInlet(s, max_buflen=1, max_chunklen=BUF_SIZE, recover=True)
                                    with self.lock:
                                        self.inlets.append(inlet)
                                        self.connected_uids.add(uid)
                                        self.active_count = len(self.inlets)
                                        slot_id = self.active_count - 1
                                    print(f"[LSL] Подключено новое устройство: {s.name()} -> Device [{slot_id}] (Всего: {self.active_count})")
                                except Exception:
                                    pass
                except Exception:
                    pass

            # Непрерывный сбор данных без пересоздания сокетов
            pulled_any = False
            with self.lock:
                current_inlets = list(self.inlets)

            for i, inlet in enumerate(current_inlets):
                try:
                    chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                    if chunk:
                        arr = np.array(chunk, dtype=np.float32).T
                        n = arr.shape[1]
                        with self.lock:
                            if n >= BUF_SIZE:
                                self.raw_buffers[i] = arr[:NUM_CHANNELS, -BUF_SIZE:]
                            else:
                                self.raw_buffers[i] = np.roll(self.raw_buffers[i], -n, axis=1)
                                self.raw_buffers[i, :, -n:] = arr[:NUM_CHANNELS, :]
                        pulled_any = True
                except Exception:
                    pass
            
            if not pulled_any:
                time.sleep(0.0005)

    def pull_to_gpu(self, gpu_dest):
        with self.lock:
            gpu_dest.copy_(torch.from_numpy(self.raw_buffers))

    def stop(self):
        self.is_running = False

# ==============================================================================
# 2. BATCHED CUDA DSP ДВИЖОК
# ==============================================================================
class BatchedHierarchicalDSP:
    def __init__(self):
        self.freqs = torch.fft.fftfreq(BUF_SIZE, d=1.0/FS).to(DEVICE)
        self.notch = torch.ones_like(self.freqs)
        self.notch[(torch.abs(self.freqs) >= 48.0) & (torch.abs(self.freqs) <= 52.0)] = 0.0
        self.notch[(torch.abs(self.freqs) >= 98.0) & (torch.abs(self.freqs) <= 102.0)] = 0.0
        self.notch = self.notch.view(1, 1, BUF_SIZE)

        self.f_theta = (torch.exp(-0.5 * ((self.freqs - 6.0) / 1.5)**2) * 2.0).view(1, 1, BUF_SIZE)
        self.f_theta[:, :, self.freqs < 0] = 0.0

        gamma_centers = torch.linspace(30.0, 85.0, NUM_FREQS, device=DEVICE).view(1, NUM_FREQS, 1, 1)
        freqs_4d = self.freqs.view(1, 1, 1, BUF_SIZE)
        self.gamma_filters = torch.exp(-0.5 * ((freqs_4d - gamma_centers) / 4.5)**2) * 2.0
        self.gamma_filters[:, :, :, self.freqs < 0] = 0.0

        self.slot_angles = (-math.pi + (2.0 * math.pi / NUM_FREQS) * (torch.arange(NUM_FREQS, device=DEVICE) + 0.5)).view(1, NUM_FREQS, 1, 1)
        self.raw_gpu = torch.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)

        self.pll_phase = 0.0
        self.pll_freq_hz = 6.0
        self.smooth_gamma_devices = torch.zeros((NUM_MAX_DEVICES, NUM_FREQS, NUM_PAIRS), device=DEVICE, dtype=torch.float32)

        self.hist_v1_fut = torch.zeros(2, device=DEVICE, dtype=torch.float32)
        self.hist_v1_pres = torch.zeros(2, device=DEVICE, dtype=torch.float32)
        self.last_phase = 0.0
        self.rho_fut = 1.0
        self.rho_past = 1.0
        self.hist_fut_queue = deque([0.0]*60, maxlen=60)
        self.hist_past_queue = deque([0.0]*60, maxlen=60)

    @torch.inference_mode()
    def process(self, raw_input, dt, routing_table, active_lsl_count):
        centered = raw_input - torch.mean(raw_input, dim=2, keepdim=True)
        fft_clean = torch.fft.fft(centered, dim=-1) * self.notch

        # 1. Мастер-Тета (по ведущему активному прибору)
        Z_theta = torch.fft.ifft(fft_clean * self.f_theta, dim=-1)
        P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
        mean_th_phasor = torch.mean(P_theta[0], dim=0, keepdim=True)
        phi_theta_raw = torch.angle(mean_th_phasor)
        
        target_eeg_phase = float(phi_theta_raw[0, -1].item())
        th_vec = phi_theta_raw[0]
        d_phi = (th_vec[1:] - th_vec[:-1] + math.pi) % (2.0 * math.pi) - math.pi
        measured_hz = float(np.clip(float(torch.mean(d_phi).item()) / (2.0 * math.pi) * FS, 3.5, 9.0))

        self.pll_freq_hz = self.pll_freq_hz * 0.95 + measured_hz * 0.05
        phase_err = (target_eeg_phase - self.pll_phase + math.pi) % (2.0 * math.pi) - math.pi
        old_phase = self.pll_phase
        self.pll_phase = (self.pll_phase + 2.0 * math.pi * self.pll_freq_hz * dt + phase_err * 2.0 * dt) % (2.0 * math.pi)

        # 2. 32 PAC слота на CUDA [4, 32, 120]
        fft_exp = fft_clean.unsqueeze(1)
        Z_gamma = torch.fft.ifft(fft_exp * self.gamma_filters, dim=-1)
        P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)

        p_diff = phi_theta_raw.view(1, 1, 1, BUF_SIZE) - self.slot_angles
        w = torch.exp(3.2 * torch.cos(p_diff))
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

        cg_gamma = P_gamma[:, :, I_GPU, :] * torch.conj(P_gamma[:, :, J_GPU, :])
        raw_gamma_4d = torch.sum(torch.imag(cg_gamma) * w, dim=-1)

        self.smooth_gamma_devices = self.smooth_gamma_devices * 0.85 + raw_gamma_4d * 0.15

        # 3. АНСАМБЛЕВЫЙ ПУЛИНГ РЕГИОНОВ
        region_tensors = {'Oz': [], 'PO7': [], 'Pz': [], 'P7': []}
        for dev_id, reg in routing_table.items():
            if reg in region_tensors and dev_id < active_lsl_count:
                if torch.std(raw_input[dev_id]) > 0.01:
                    region_tensors[reg].append(self.smooth_gamma_devices[dev_id])

        pooled = {}
        for reg, t_list in region_tensors.items():
            if len(t_list) > 0:
                pooled[reg] = torch.mean(torch.stack(t_list, dim=0), dim=0)
            else:
                pooled[reg] = None

        base_slot = (self.pll_phase / (2.0 * math.pi)) * NUM_FREQS
        k_now = int(base_slot) % NUM_FREQS
        k_fut = int(base_slot + 12.0) % NUM_FREQS

        # ДЕКОДИРОВАНИЕ 1: Oz (V1 Текстура Габора)
        if pooled['Oz'] is not None:
            flow_x = torch.sum(pooled['Oz'] * DX_GPU, dim=-1)
            flow_y = torch.sum(pooled['Oz'] * DY_GPU, dim=-1)
            v1_angles = torch.atan2(flow_y, flow_x) * 0.5
            v1_contrasts = torch.clamp(torch.hypot(flow_x, flow_y) * 0.08, 0.0, 1.0)
            a_now, c_now = float(v1_angles[k_now].item()), float(v1_contrasts[k_now].item())
            a_past, c_past = float(v1_angles[0].item()), float(v1_contrasts[0].item())
            a_fut, c_fut = float(v1_angles[k_fut].item()), float(v1_contrasts[k_fut].item())
        else:
            a_now, c_now = 0.0, 0.75
            a_past, c_past = 0.0, 0.75
            a_fut, c_fut = 0.0, 0.75

        # ДЕКОДИРОВАНИЕ 2: PO7 (V4 Форма Пасупати-Коннора)
        if pooled['PO7'] is not None:
            cross_pwr = torch.sum(pooled['PO7'][:, IDX_CROSS_GPU], dim=-1) * 0.03
            ring_pwr = torch.sum(pooled['PO7'][:, idx_ring], dim=-1) * 0.03
            core_pwr = torch.sum(pooled['PO7'][:, idx_core], dim=-1) * 0.03

            shape_k2 = float(torch.clamp(ring_pwr[k_now] * 1.5, 0.0, 0.45).item())
            shape_k3 = float(torch.clamp(cross_pwr[k_now] * 1.5, 0.0, 0.40).item())
            shape_k4 = float(torch.clamp(core_pwr[k_now] * 1.5, 0.0, 0.35).item())
            shape_rot = float(torch.sum(pooled['PO7'][k_now] * CURL_GPU).item()) * 0.02
        else:
            shape_k2, shape_k3, shape_k4, shape_rot = 0.0, 0.0, 0.0, 0.0

        # ДЕКОДИРОВАНИЕ 3: Pz (PPC Пространственный Центр X, Y)
        if pooled['Pz'] is not None:
            pz_fx = torch.sum(pooled['Pz'] * DX_GPU, dim=-1)
            pz_fy = torch.sum(pooled['Pz'] * DY_GPU, dim=-1)
            pos_x = float(torch.clamp(pz_fx[k_now] * 0.0008, -0.25, 0.25).item()) * ASPECT
            pos_y = float(torch.clamp(pz_fy[k_now] * 0.0008, -0.25, 0.25).item())
        else:
            pos_x, pos_y = 0.0, 0.0

        # ДЕКОДИРОВАНИЕ 4: P7 (MT Скорость Потока)
        if pooled['P7'] is not None:
            mt_fx = torch.sum(pooled['P7'] * DX_GPU, dim=-1)
            mt_fy = torch.sum(pooled['P7'] * DY_GPU, dim=-1)
            flow_vx = float(mt_fx[k_now].item()) * 0.005
            flow_vy = float(mt_fy[k_now].item()) * 0.005
        else:
            flow_vx, flow_vy = 0.0, 0.0

        # Проверка памяти V1
        if self.pll_phase < old_phase and pooled['Oz'] is not None:
            diff_fut = math.cos(2.0 * (a_now - self.hist_v1_fut[0]))
            self.rho_fut = self.rho_fut * 0.75 + diff_fut * 0.25
            diff_past = math.cos(2.0 * (a_past - self.hist_v1_pres[0]))
            self.rho_past = self.rho_past * 0.75 + diff_past * 0.25
            self.hist_v1_fut[0] = a_fut
            self.hist_v1_pres[0] = a_now
            self.hist_fut_queue.append(self.rho_fut)
            self.hist_past_queue.append(self.rho_past)

        all_empty = (active_lsl_count == 0)

        return {
            'all_empty': all_empty,
            'a_now': a_now, 'c_now': c_now, 'a_past': a_past, 'c_past': c_past,
            'a_fut_prev': self.hist_v1_fut[0].item(), 'c_fut_prev': c_fut,
            'shape_k2': shape_k2, 'shape_k3': shape_k3, 'shape_k4': shape_k4, 'shape_rot': shape_rot,
            'pos_x': pos_x, 'pos_y': pos_y,
            'flow_vx': flow_vx, 'flow_vy': flow_vy,
            'rho_fut': self.rho_fut, 'rho_past': self.rho_past,
            'theta_hz': self.pll_freq_hz, 'rms': float(torch.std(raw_input[:active_lsl_count]).item()) if active_lsl_count > 0 else 0.0,
            'pooled_status': {k: (v is not None) for k, v in pooled.items()},
            'hist_fut': list(self.hist_fut_queue), 'hist_past': list(self.hist_past_queue)
        }

# ==============================================================================
# 3. GLSL ИЕРАРХИЧЕСКИЙ ШЕЙДЕР
# ==============================================================================
HIERARCHY_VERT = """
#version 330 core
layout(location = 0) in vec2 in_pos;
out vec2 v_uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_pos; }
"""

HIERARCHY_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 out_color;

uniform float u_all_empty;

// 1. Pz (PPC Центр)
uniform vec2 u_center_pos;

// 2. PO7 (V4 Форма Пасупати-Коннора)
uniform float u_shape_k2;
uniform float u_shape_k3;
uniform float u_shape_k4;
uniform float u_shape_rot;

// 3. Oz (V1 Текстура Габора)
uniform float u_a_now;  uniform float u_c_now;
uniform float u_a_fut;  uniform float u_c_fut;
uniform float u_a_past; uniform float u_c_past;

// 4. P7 (MT Оптический сдвиг фазы)
uniform float u_motion_phase;

void main() {
    if (u_all_empty > 0.5) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 p = v_uv - u_center_pos;
    float r = length(p);
    float alpha = atan(p.y, p.x) - u_shape_rot;

    // Радиальный сплайн V4
    float r_base = 0.32;
    float r_boundary = r_base * (1.0 + u_shape_k2 * cos(2.0 * alpha) 
                                      + u_shape_k3 * cos(3.0 * alpha) 
                                      + u_shape_k4 * cos(4.0 * alpha));

    float shape_mask = smoothstep(r_boundary + 0.015, r_boundary - 0.015, r);
    if (shape_mask <= 0.001) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float freq = 7.0;
    
    // RED (Будущее)
    float phase_r = 2.0 * 3.14159 * freq * (p.x * cos(u_a_fut) + p.y * sin(u_a_fut)) + u_motion_phase;
    float wave_r = max(0.0, cos(phase_r)) * u_c_fut;

    // GREEN (Настоящее)
    float phase_g = 2.0 * 3.14159 * freq * (p.x * cos(u_a_now) + p.y * sin(u_a_now)) + u_motion_phase;
    float wave_g = max(0.0, cos(phase_g)) * u_c_now;

    // BLUE (Прошлое)
    float phase_b = 2.0 * 3.14159 * freq * (p.x * cos(u_a_past) + p.y * sin(u_a_past)) + u_motion_phase;
    float wave_b = max(0.0, cos(phase_b)) * u_c_past;

    float border_glow = exp(-pow((r - r_boundary) * 35.0, 2.0)) * 0.6;

    float R = (wave_r * shape_mask + border_glow) * 2.0;
    float G = (wave_g * shape_mask + border_glow) * 2.0;
    float B = (wave_b * shape_mask + border_glow) * 2.0;

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

class HierarchicalVisualizer:
    def __init__(self):
        vs = shaders.compileShader(HIERARCHY_VERT, GL_VERTEX_SHADER)
        fs = shaders.compileShader(HIERARCHY_FRAG, GL_FRAGMENT_SHADER)
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
            'u_all_empty': glGetUniformLocation(self.shader, "u_all_empty"),
            'u_center_pos': glGetUniformLocation(self.shader, "u_center_pos"),
            'u_shape_k2': glGetUniformLocation(self.shader, "u_shape_k2"),
            'u_shape_k3': glGetUniformLocation(self.shader, "u_shape_k3"),
            'u_shape_k4': glGetUniformLocation(self.shader, "u_shape_k4"),
            'u_shape_rot': glGetUniformLocation(self.shader, "u_shape_rot"),
            'u_a_now': glGetUniformLocation(self.shader, "u_a_now"),
            'u_c_now': glGetUniformLocation(self.shader, "u_c_now"),
            'u_a_fut': glGetUniformLocation(self.shader, "u_a_fut"),
            'u_c_fut': glGetUniformLocation(self.shader, "u_c_fut"),
            'u_a_past': glGetUniformLocation(self.shader, "u_a_past"),
            'u_c_past': glGetUniformLocation(self.shader, "u_c_past"),
            'u_motion_phase': glGetUniformLocation(self.shader, "u_motion_phase"),
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

        self.HUD_W, self.HUD_H = 600, 580
        self.hud_surface = pygame.Surface((self.HUD_W, self.HUD_H), pygame.SRCALPHA)
        self.font_title = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_main = pygame.font.SysFont("consolas", 11, bold=True)
        self.font_sub = pygame.font.SysFont("consolas", 9)
        
        glBindTexture(GL_TEXTURE_2D, self.hud_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.motion_phase_acc = 0.0

    def render(self, state, routing_table, active_lsl_count, show_hud):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self.motion_phase_acc += math.hypot(state['flow_vx'], state['flow_vy']) * 5.0

        glUseProgram(self.shader)
        
        glUniform1f(self.u_vars['u_all_empty'], 1.0 if state['all_empty'] else 0.0)
        glUniform2f(self.u_vars['u_center_pos'], state['pos_x'], state['pos_y'])

        glUniform1f(self.u_vars['u_shape_k2'], state['shape_k2'])
        glUniform1f(self.u_vars['u_shape_k3'], state['shape_k3'])
        glUniform1f(self.u_vars['u_shape_k4'], state['shape_k4'])
        glUniform1f(self.u_vars['u_shape_rot'], state['shape_rot'])

        glUniform1f(self.u_vars['u_a_now'], state['a_now'])
        glUniform1f(self.u_vars['u_c_now'], state['c_now'])
        glUniform1f(self.u_vars['u_a_fut'], state['a_fut_prev'])
        glUniform1f(self.u_vars['u_c_fut'], state['c_fut_prev'])
        glUniform1f(self.u_vars['u_a_past'], state['a_past'])
        glUniform1f(self.u_vars['u_c_past'], state['c_past'])

        glUniform1f(self.u_vars['u_motion_phase'], self.motion_phase_acc)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

        if show_hud:
            self._render_lab_hud(state, routing_table, active_lsl_count)

    def _render_lab_hud(self, state, routing_table, active_lsl_count):
        self.hud_surface.fill((10, 15, 22, 245))
        pygame.draw.rect(self.hud_surface, (0, 200, 255), (0, 0, self.HUD_W, self.HUD_H), 1, border_radius=4)

        self.hud_surface.blit(self.font_title.render("HIERARCHICAL MULTI-DEVICE VWM LAB", True, (255, 255, 255)), (15, 10))
        self.hud_surface.blit(self.font_sub.render(f"LSL Inlets Active: {active_lsl_count} / {NUM_MAX_DEVICES} | RMS: {state['rms']:.1f} uV | Theta: {state['theta_hz']:.2f} Hz", True, (150, 180, 200)), (15, 26))

        y = 46
        self.hud_surface.blit(self.font_main.render("DEVICE ASSIGNMENTS [Keys 1, 2, 3, 4 to cycle]:", True, (255, 220, 100)), (15, y)); y+=18
        
        region_colors = {'Oz': (0, 255, 180), 'PO7': (255, 180, 50), 'Pz': (100, 200, 255), 'P7': (255, 100, 200), 'OFF': (100, 100, 100)}
        for dev_id in range(NUM_MAX_DEVICES):
            reg = routing_table[dev_id]
            col = region_colors[reg]
            self.hud_surface.blit(self.font_main.render(f"• Dev {dev_id}: [{reg}]", True, col), (20 + dev_id * 130, y))
        y += 24

        self.hud_surface.blit(self.font_main.render("HIERARCHICAL CORTICAL STACK:", True, (255, 255, 255)), (15, y)); y+=16
        
        stack_info = [
            ("Pz  (PPC)   : SPATIAL POINTER (x, y)", state['pooled_status']['Pz'], f"dx:{state['pos_x']*100:.1f} dy:{state['pos_y']*100:.1f}"),
            ("PO7 (V4/LOC): SHAPE SPLINE (k2, k3, k4)", state['pooled_status']['PO7'], f"k2:{state['shape_k2']:.2f} k3:{state['shape_k3']:.2f}"),
            ("Oz  (V1/V2) : GABOR TEXTURE (θ, Contrast)", state['pooled_status']['Oz'], f"θ:{math.degrees(state['a_now']):.0f}° C:{state['c_now']:.2f}"),
            ("P7  (MT/V5) : OPTIC FLOW VELOCITY (v)", state['pooled_status']['P7'], f"v:{math.hypot(state['flow_vx'], state['flow_vy'])*100:.1f}")
        ]

        for name, is_live, val_str in stack_info:
            col = (0, 255, 150) if is_live else (140, 150, 160)
            status = "LIVE ENSEMBLE" if is_live else "DETERMINISTIC STUB"
            self.hud_surface.blit(self.font_main.render(f"• {name} -> [{status}] ({val_str})", True, col), (15, y))
            y += 16

        y += 8
        col_f = (0, 255, 180) if state['rho_fut'] > 0.4 else (255, 100, 100)
        self.hud_surface.blit(self.font_main.render(f"• V1 PREDICTION (rho_fut) : {state['rho_fut']:+.2f}", True, col_f), (15, y))
        pygame.draw.rect(self.hud_surface, (30, 40, 50), (220, y + 2, 200, 6))
        pygame.draw.rect(self.hud_surface, col_f, (220, y + 2, int(max(0, (state['rho_fut']+1)/2) * 200), 6))
        y += 18

        col_p = (80, 180, 255) if state['rho_past'] > 0.4 else (255, 100, 100)
        self.hud_surface.blit(self.font_main.render(f"• V1 ANCHOR     (rho_past): {state['rho_past']:+.2f}", True, col_p), (15, y))
        pygame.draw.rect(self.hud_surface, (30, 40, 50), (220, y + 2, 200, 6))
        pygame.draw.rect(self.hud_surface, col_p, (220, y + 2, int(max(0, (state['rho_past']+1)/2) * 200), 6))
        y += 24

        self.hud_surface.blit(self.font_sub.render("V1 INTER-CYCLE CONTINUITY OSCILLOSCOPE (Last 60 Cycles)", True, (150, 180, 200)), (15, y))
        y += 14
        ox, oy, ow, oh = 15, y, 570, 50
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

        y += 65
        self.hud_surface.blit(self.font_sub.render("[1-4] Cycle Region for Dev 0-3 | [F1] Toggle HUD", True, (130, 150, 170)), (15, y))

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
    pygame.display.set_caption("NeuroCanvas: Hierarchical Multi-Device VWM Lab")

    glViewport(0, 0, WIDTH, HEIGHT)

    receiver = StableMultiLSLReceiver()
    dsp = BatchedHierarchicalDSP()
    visualizer = HierarchicalVisualizer()

    REGION_LIST = ['Oz', 'PO7', 'Pz', 'P7', 'OFF']
    routing_table = {0: 'Oz', 1: 'PO7', 2: 'Pz', 3: 'P7'}

    show_hud = False
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
                    if event.key == pygame.K_F1:
                        show_hud = not show_hud
                    elif event.key == pygame.K_1:
                        idx = REGION_LIST.index(routing_table[0])
                        routing_table[0] = REGION_LIST[(idx + 1) % len(REGION_LIST)]
                    elif event.key == pygame.K_2:
                        idx = REGION_LIST.index(routing_table[1])
                        routing_table[1] = REGION_LIST[(idx + 1) % len(REGION_LIST)]
                    elif event.key == pygame.K_3:
                        idx = REGION_LIST.index(routing_table[2])
                        routing_table[2] = REGION_LIST[(idx + 1) % len(REGION_LIST)]
                    elif event.key == pygame.K_4:
                        idx = REGION_LIST.index(routing_table[3])
                        routing_table[3] = REGION_LIST[(idx + 1) % len(REGION_LIST)]

            receiver.pull_to_gpu(dsp.raw_gpu)
            state = dsp.process(dsp.raw_gpu, dt, routing_table, receiver.active_count)
            visualizer.render(state, routing_table, receiver.active_count, show_hud)

            pygame.display.flip()

    finally:
        receiver.stop()
        pygame.quit()

if __name__ == '__main__':
    main()

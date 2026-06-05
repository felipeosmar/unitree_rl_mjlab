# Relatório de Tuning — Treinamento LiDAR (Unitree-Go2-Flat)

## 1. Hardware

| Componente | Modelo |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell (97.9 GiB VRAM) |
| CPU | Intel Xeon w5-3435X (16C/32T) |
| RAM | 125 GiB DDR5 |
| Workstation | Dell Precision 7960 Tower |

## 2. Configuração Atual

### 2.1. Parâmetros do Script (`train_lidar.sh`)

| Parâmetro | Valor |
|---|---|
| `NUM_ENVS` | 16384 |
| `num_steps_per_env` | 48 |
| `num_mini_batches` | 16 |
| `num_learning_epochs` | 8 |
| `max_iterations` | 30000 |

### 2.2. Simulação

| Parâmetro | Valor | Impacto |
|---|---|---|
| `decimation` | 4 | 4 passos de simulação por passo de política |
| `physics_dt` | 0.005 s (5 ms) | Passo de integração MuJoCo |
| `policy_dt` | 0.02 s (20 ms = 50 Hz) | Frequência de controle |
| `solver` | `newton` | Mais preciso, mais caro |
| `iterations` | **10** (vs default 100) | Já reduzido |
| `ls_iterations` | **20** (vs default 50) | Já reduzido |
| `ls_parallel` | `True` | Paralelismo GPU no solver |
| `integrator` | `implicitfast` | Equilíbrio velocidade/precisão |
| `tolerance` | 1e-8 | Precisão alta do solver |
| `ccd_iterations` | 50 | Detecção de colisão contínua |
| CUDA Graphs | **Ativado** | Kernel replay sem overhead CPU |
| `nconmax` | `None` (automático) | Contatos por mundo |
| `njmax` | 1500 | Constraints por mundo |

### 2.3. Rede Neural

| Componente | Arquitetura |
|---|---|
| Actor | MLP (1024 → 512 → 256), ativação ELU |
| Critic | MLP (1024 → 512 → 256), ativação ELU |
| Distribuição | Gaussian (std treinável escalar) |
| Normalização | `EmpiricalNormalization` (running mean/std) |

### 2.4. Observações

| Grupo | Termos | Dimensão estimada |
|---|---|---|
| Actor | base_ang_vel (3), projected_gravity (3), command (3), phase (1), joint_pos (12), joint_vel (12), actions (12), lidar_scan (2048) | **~2094** |
| Critic | Tudo do actor + base_lin_vel (3), foot_height (4), foot_air_time (4), foot_contact (4), foot_contact_forces (12) | **~2121** |

- **LiDAR**: 32 canais × 64 resolução horizontal = **2048 raios por ambiente**
- **16384 ambientes × 2048 raios = 33.5 milhões de raios** por passo de sensing
- Todo o pipeline de observações roda **na GPU** (Warp + PyTorch)

### 2.5. Pipeline PPO

1. **Coleta** (`collect_time`): `num_steps_per_env` (48) passos de:
   - `actor.act(obs)` → ações
   - `env.step(actions)` → 4 `mjwarp.step()` + sense + observações
   - `process_env_step()` → armazena transição
2. **GAE** (`compute_returns`): cálculo de vantagens (loop sequencial)
3. **Update** (`learn_time`): `num_learning_epochs` (8) × `num_mini_batches` (16) = 128 mini-batches:
   - forward actor/critic → loss → backward → optimizer.step()

### 2.6. Performance Atual

| Métrica | Valor |
|---|---|
| **Tempo por iteração** | **~19.72 s** |
| **FPS (env steps/s)** | ~39.882 (16384 envs × 48 steps / 19.72s) |
| **ETA para 30k iterações** | ~22.5 horas |

---

## 3. Oportunidades de Otimização

Prioridade: 🔥 Alta | ⚡ Média | 💡 Baixa

### 3.1. 🔥 `torch.compile` na Rede

**Problema**: Nenhuma parte da pipeline usa `torch.compile`. As redes Actor (MLP 1024→512→256) e Critic rodam com interpretação PyTorch padrão (eager mode).

**Ganho estimado**: 10-30% de speedup no `learn_time`.

**Implementação**: Em `.venv/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py`, no `__init__` do `PPO`:

```python
self.actor = torch.compile(self.actor, mode="reduce-overhead")
self.critic = torch.compile(self.critic, mode="reduce-overhead")
```

**Riscos**: Primeira iteração mais lenta (compilação). Pode aumentar consumo de VRAM ~10-20%.

---

### 3.2. 🔥 Mixed Precision (AMP)

**Problema**: Todo o treino usa FP32. As GPUs Blackwell têm aceleração FP16/BF16 significativa.

**Ganho estimado**: 20-40% de speedup no `learn_time` + economia de VRAM (~30%).

**Implementação**: Envolver o forward + loss do PPO update com `torch.amp.autocast`:

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    # actor critic forward + loss computation
    ...
scaler = torch.amp.GradScaler()
scaler.scale(loss).backward()
scaler.step(self.optimizer)
scaler.update()
```

**Riscos**: Pode haver instabilidade numérica. Usar `bfloat16` (nativo Blackwell) em vez de `float16` reduz risco.

---

### 3.3. 🔥 Trocas de Solver MuJoCo

**Problema**: `solver="newton"` é o mais caro. Para terreno plano (`Flat`), precisão excessiva.

**Opções** (em ordem de velocidade):

| Solver | Velocidade | Precisão | Recomendação |
|---|---|---|---|
| `newton` (atual) | 1.0x (base) | Excelente | — |
| `cg` | ~1.5-2x | Boa | ✅ **Melhor custo-benefício** |
| `pgs` | ~2-3x | Razoável | ⚠️ Pode instabilizar |

**Ganho estimado**: 30-50% de redução no `collect_time` com `solver="cg"`.

**Implementação**: Em `src/tasks/velocity/config/go2/env_cfgs.py`, função `unitree_go2_flat_env_cfg()`:

```python
cfg.sim.mujoco.solver = "cg"
```

**Riscos**: `cg` pode precisar de mais iterações. Aumentar `cfg.sim.mujoco.iterations` para 20-30.

---

### 3.4. ⚡ Aumentar `tolerance` do Solver

**Problema**: `tolerance=1e-8` é precisão desnecessária para simulação de locomoção.

**Solução**: Aumentar para `1e-6` ou `1e-5`:

```python
cfg.sim.mujoco.tolerance = 1e-6
```

**Ganho**: O solver converge em menos iterações. Pode reduzir `collect_time` em 5-15%.

---

### 3.5. ⚡ Reduzir `ccd_iterations`

**Problema**: `ccd_iterations=50` é o padrão, mas terreno plano tem menos colisões.

**Solução**: Reduzir para 10-20:

```python
cfg.sim.mujoco.ccd_iterations = 10
```

**Ganho**: Moderado. Economia em detecção de colisão contínua.

---

### 3.6. ⚡ `sense()` — Otimizar Frequência de Raycasting

**Problema**: O LiDAR de 2048 raios executa BVH rebuild + raycast **a cada passo de ambiente** (50 Hz). Com 16384 ambientes, são 33.5M raios/s.

**Solução**: Amostragem temporal — sense a cada 2 ou 3 env steps:

No `manager_based_rl_env.py`, modificar `step()`:

```python
if self.common_step_counter % 2 == 0:  # sense a cada 2 steps
    self.sim.sense()
```

**Ganho**: Reduz custo de raycasting em 50%. Cuidado: observações ficam "atrasadas".

---

### 3.7. ⚡ `compute_returns` — Vetorização

**Problema**: O método `compute_returns()` em `ppo.py` (linhas ~187-209) usa um loop Python sequencial sobre os timesteps:

```python
for step in reversed(range(st.num_transitions_per_env)):
    ...
```

**Solução**: Implementar versão vetorizada com operações cumulativas.

**Ganho**: Pequeno no contexto geral (< 1% do tempo total), mas é eliminação de overhead desnecessário.

---

### 3.8. 💡 Cache de `nn.MSELoss()`

**Problema**: No PPO update, `nn.MSELoss()` é recriado a cada mini-batch.

**Solução**: Instanciar uma vez no `__init__`.

**Ganho**: Marginal. Mas é zero-risk.

---

### 3.9. 💡 Gradient Accumulation

**Problema**: Com `num_mini_batches=16` e `num_learning_epochs=8`, temos 128 updates por iteração. Cada um faz `optimizer.zero_grad()` + `backward()` + `optimizer.step()`.

**Solução**: Acumular gradientes por múltiplos mini-batches antes de `step()`.

**Ganho**: Reduz sincronizações CUDA. Pode ser combinado com AMP para maior throughput.

---

## 4. Recomendações Prioritárias

### Curto Prazo (implementação imediata, sem risco):
1. ⚡ Tolerância: `1e-8` → `1e-6`
2. ⚡ CCD iterations: `50` → `10`
3. 💡 Cache de MSELoss

### Médio Prazo (testar e validar):
4. 🔥 `solver="newton"` → `"cg"` (+ aumentar iterations para 20)
5. ⚡ Sense a cada 2 steps (se a política tolerar)

### Longo Prazo (requer modificações no código-fonte das libs):
6. 🔥 `torch.compile` na rede (actor + critic)
7. 🔥 Mixed Precision (AMP) com `bfloat16`
8. ⚡ Vetorizar `compute_returns`

---

## 5. Estimativa de Ganho Acumulado

| Otimização | Ganho colet_time | Ganho learn_time | Ganho total |
|---|---|---|---|
| Solver `cg` | 30-50% | — | ~20-30% |
| Tolerância `1e-6` | 5-15% | — | ~3-10% |
| `torch.compile` | — | 10-30% | ~3-15% |
| AMP bf16 | — | 20-40% | ~6-20% |
| Sense redução | 10-20% | — | ~5-15% |
| **Total acumulado** | **~40-70%** | **~30-60%** | **~35-65%** |

**Potencial**: Reduzir tempo de iteração de ~19.7s para **~7-12s**, e ETA de 22h para **~8-14h**.

---

## 6. Observações

- **GPU utiliza apenas ~10-15% dos 98 GB VRAM** com a config atual (~16 GiB usados dos 98 GiB). Há margem enorme para aumentar `num_envs` ou complexidade da rede.
- **CPU fica ocioso** (~1 core ativo) porque a simulação é batchada na GPU via Warp + CUDA Graphs, e o PyTorch também roda na GPU. Isto é esperado e desejável.
- **CUDA Graphs já estão ativados** e funcionando (driver NVIDIA ≥ 535 detectado, memory pool ativo).
- **Warp backward pass desativado** (`wp.config.enable_backward = False`) — correto, economiza memória e tempo de compilação.
- **`ls_parallel=True`** ativado — acelera o solver Newton.

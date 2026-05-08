# Modo Galope (Bound) para Go2

Política separada de alta velocidade. O Go2 tem o checkpoint de **trote** (já
treinado, `model_10000.pt`) e a partir de agora também pode ter um checkpoint de
**galope/bound** para correr mais rápido com menos energia.

No simulador, ambos rodam ao mesmo tempo no `play.py` e você troca com o **botão Y**
do controle Xbox.

## Por que bound e não galope rotativo (estilo guepardo)?

O guepardo galopa eficiente porque tem **coluna flexível** que armazena energia
elástica. O Go2 tem chassi rígido — sem mola natural, o "ganho" do galope rotativo
sobre bound desaparece. Bound é também muito mais fácil de convergir no RL.

## Diferenças vs. trote

| Parâmetro | Trote | Bound (galope) |
|---|---|---|
| Gait offset | `[0.0, 0.5, 0.5, 0.0]` (FR+RL / FL+RR) | `[0.0, 0.0, 0.5, 0.5]` (frente / trás) |
| Período | 0.6 s | 0.35 s |
| Stance fraction | 0.56 | 0.45 (mais fase aérea) |
| `lin_vel_x` máximo | 2.0 m/s | 3.5 m/s |
| `target_height` (clearance) | 0.10 m | 0.14 m |
| `running_threshold` | 1.5 m/s | 0.8 m/s |

## Como treinar

Política separada, em terreno plano, ~50k iterações:

```bash
bash scripts/train_gallop.sh                       # 4096 envs, 50000 iters
NUM_ENVS=2048 ITERS=30000 bash scripts/train_gallop.sh
```

Logs em `logs/rsl_rl/go2_gallop/<TIMESTAMP>/`.

Monitorar:
```bash
tail -F logs/pipeline/train_gallop_<TS>.log | grep CHECKPOINT
```

## Como rodar com chaveamento trote ↔ galope

Após treinar, rode os dois policies no mesmo viewer:

```bash
.venv/bin/python scripts/play.py Unitree-Go2-Flat \
  --checkpoint_file logs/rsl_rl/go2_velocity/2026-04-15_16-10-05/model_10000.pt \
  --gallop_checkpoint logs/rsl_rl/go2_gallop/<TIMESTAMP>/model_50000.pt \
  --viewer=native
```

**Controles do Xbox:**

| Controle | Função |
|---|---|
| Stick esquerdo | Andar (forward / strafe) |
| Stick direito (X) | Girar |
| **Botão Y** | **Alternar trote ↔ galope** |

Ao alternar, o terminal imprime:
```
[Modo] trote (period=0.60s, max_vx=2.0 m/s)
[Modo] galope (period=0.35s, max_vx=3.5 m/s)
```

A `phase` observation muda de período automaticamente para casar com o policy
ativo, e o limite máximo do stick é reescalado.

## Como funciona internamente

1. `play.py` carrega dois `OnPolicyRunner` no mesmo `env`. Cada runner tem
   seu próprio actor MLP e normalizador de observação.
2. `PolicySwitcher` mantém um índice (0=trote, 1=galope) e expõe um `__call__`
   que despacha para o policy ativo.
3. Quando o botão Y é pressionado (transição de não-pressionado para
   pressionado), `toggle()`:
   - Muta `observation_manager.get_term_cfg("actor"/"critic", "phase").params["period"]`
     para o período do novo policy.
   - Atualiza o `max_speed` usado pelo escalonamento do stick analógico.
4. O `feet_gait` reward usa o período como param também, mas durante o `play`
   rewards não afetam comportamento, só estatística.

## Arquivos modificados

- `src/tasks/velocity/config/go2/env_cfgs.py` — `unitree_go2_gallop_env_cfg()`.
- `src/tasks/velocity/config/go2/rl_cfg.py` — `unitree_go2_gallop_ppo_runner_cfg()`.
- `src/tasks/velocity/config/go2/__init__.py` — registra `Unitree-Go2-Gallop`.
- `scripts/train_gallop.sh` — script de treino.
- `scripts/play.py` — `PolicySwitcher` + flags `--gallop_checkpoint` e
  `--gallop_task_id`.

## Limitações / cuidados

- **Bound exige potência**: o Go2 vai esquentar mais e bateria dura menos.
  Não use bound contínuo no robô real por longos períodos sem monitorar
  temperatura dos motores.
- **Treinado em flat**: não testar bound em rough/obstacles sem retreinar
  com terreno irregular.
- **Sim-to-real**: a transferência do bound para hardware real é mais sensível
  que o trote. Espere ajustar `action_scale` ou domain randomization se for
  fazer deploy.
- **Velocidade efetiva**: a curriculum vai até 3.5 m/s mas o Go2 real raramente
  alcança isso de forma estável. Espere ~2.5–3.0 m/s na prática.

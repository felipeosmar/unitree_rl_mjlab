# Como fazer o Go2 andar mais rápido

A velocidade máxima que o robô atinge é definida em **3 lugares diferentes**, todos no treino.
Em runtime (gamepad/play) você só consegue o que foi treinado.

## 1. O que limita a velocidade hoje

No `src/tasks/velocity/velocity_env_cfg.py` deste projeto:

```
lin_vel_x: (-1.0, 2.0)    → max 2 m/s pra frente
ang_vel_z: (-1.0, 1.0)    → max 1 rad/s de rotação
Curriculum:
  iter 0     → (-0.5, 1.0)
  iter 5000  → (-1.0, 2.0)
```

O gamepad multiplica o stick analógico pelo *upper bound* do range. Com o stick a fundo
no eixo Y do esquerdo, você manda `1.0 × 2.0 = 2.0 m/s`. O robô **nunca foi treinado**
a fazer mais que isso, então pedir além disso causa comportamento ruim ou queda.

## 2. Os três pontos de ajuste no treino

### a) Range de comando (`lin_vel_x` superior)
Aumenta o teto. Ex: `(-2.0, 3.0)`. Define o que a política vê durante o treino.

### b) Curriculum
Não pode pular direto pra 3 m/s — a política não converge. Curriculum sobe gradualmente:

| Iteração | lin_vel_x        | ang_vel_z      |
|----------|------------------|----------------|
| 0        | (-0.5, 1.0)      | (-1.0, 1.0)    |
| 5 000    | (-1.0, 2.0)      | (-1.0, 1.0)    |
| 10 000   | (-1.5, 2.5)      | (-1.0, 1.0)    |
| 20 000   | (-2.0, 3.0)      | (-1.0, 1.0)    |

### c) Recompensas e penalidades
Definem quanto esforço o robô faz pra acompanhar o comando:

- **`tracking_lin_vel`**: pontos por bater a velocidade pedida — quanto maior peso, mais ele se esforça.
- **`action_rate` / `joint_acc` / `energy`**: penalizam movimento brusco — reduzir se quiser gait mais agressivo.
- **`survival`**: incentiva não cair, mas se for alto demais o robô "trava" em pose segura e lenta.

Se o robô consistentemente não atinge a velocidade pedida, geralmente é porque
**penalty > tracking reward** naquela faixa de velocidade.

## 3. Limites físicos do Go2

- Spec real Unitree: ~3.7 m/s (modo *sport*).
- Limite absoluto: torque e velocidade máxima dos motores no MJCF. Não dá pra ir além
  sem editar o modelo XML do robô.
- Na prática, treino estável até ~3 m/s é viável; acima disso o gait fica instável.

## 4. Processo prático pra implementar

1. Editar `src/tasks/velocity/velocity_env_cfg.py`:
   - Range final do `lin_vel_x` aumentado.
   - Curriculum estendido com etapas adicionais (mais iterações pra estabilizar).
2. Aumentar `max_iterations` no treino (ex.: 30k–50k pra dar tempo do curriculum completar).
3. Treinar **do zero** ou fazer **fine-tune** a partir do checkpoint atual com resume +
   nova range.
4. Avaliar no viser: comandar velocidade alta com gamepad e observar se o gait permanece estável.

## 5. Cuidados / sintomas comuns

| Sintoma                                              | Causa provável                                    | Ajuste                                                  |
|------------------------------------------------------|---------------------------------------------------|---------------------------------------------------------|
| Cai muito durante curriculum                         | Curriculum subiu rápido demais                    | Adicionar mais etapas intermediárias                    |
| Robô anda devagar mesmo com comando alto             | Penalties dominando (action_rate, energy)         | Reduzir pesos das penalties OU aumentar peso do tracking |
| Tremor / oscilação em alta velocidade                | `action_scale` grande, ou falta penalty `joint_velocity` | Reduzir action_scale, adicionar penalty                  |
| Termination muito severa em alta velocidade          | Limites de roll/pitch ou base height muito apertados | Suavizar termination thresholds                          |

## Resumindo

**Mais velocidade = retreinar com novo range + curriculum + balanceamento de rewards.**
Não há atalho em runtime.

---

# Sensores usados no treino do Go2 (velocity task)

A arquitetura é **actor-critic assimétrica** (PPO): o actor (política que vai pro robô real)
e o critic (só usado no treino) recebem conjuntos diferentes de observações.

Configuração em `src/tasks/velocity/velocity_env_cfg.py`.

## Actor — sensores que o robô real terá

São os 8 inputs que a política consome tanto em sim quanto em hardware:

| # | Observação           | Sensor real correspondente                                  | Ruído simulado (Unoise) |
|---|----------------------|-------------------------------------------------------------|--------------------------|
| 1 | `base_ang_vel`       | **IMU — giroscópio** (3 eixos)                              | ±0.2 rad/s              |
| 2 | `projected_gravity`  | **IMU — acelerômetro** (vetor gravidade no frame do corpo)  | ±0.05                    |
| 3 | `command`            | Gamepad / comando externo (vx, vy, ωz)                      | —                        |
| 4 | `phase`              | Clock interno (sinal periódico de gait, T=0.6s)             | —                        |
| 5 | `joint_pos`          | **Encoders de posição** das 12 juntas                       | ±0.01 rad               |
| 6 | `joint_vel`          | **Encoders de velocidade** das 12 juntas                    | ±1.5 rad/s              |
| 7 | `actions`            | Última ação enviada (feedback interno)                       | —                        |
| 8 | `height_scan`        | Grid 1.6×1.0 m raycast (16×10 = 160 pontos, raio 5 m). No real seria LiDAR ou depth camera | ±0.1 m |

Os ruídos `Unoise` simulam imprecisão dos sensores reais — sem ruído no treino, a política
**não generaliza pro hardware** (sim-to-real gap).

## Critic — informação privilegiada (só durante treino)

O critic vê tudo do actor **sem ruído** + dados que o robô real não consegue medir bem:

| Observação            | Por que é só do critic                                                |
|-----------------------|------------------------------------------------------------------------|
| `base_lin_vel`        | Velocidade linear é difícil estimar com IMU sozinha (drift)            |
| `foot_height`         | Altura absoluta dos pés (mocap-like, não tem no robô)                  |
| `foot_air_time`       | Tempo no ar de cada pé                                                 |
| `foot_contact`        | Booleano de contato dos pés                                            |
| `foot_contact_forces` | Forças de contato (Newton) — exige sensor caro                          |

Esse é o ponto-chave do *asymmetric actor-critic*: o critic ajuda a treinar com info
ground-truth, mas a **política deployada usa só o actor**.

## Detalhes importantes

- **Histórico**: `history_length: 1` — apenas o estado atual, sem janela temporal.
- **Concatenação**: `concatenate_terms: True` — todas as observações viram um único vetor flat passado ao MLP.
- **Corruption**: actor tem `enable_corruption: True` (aplica ruído); critic tem `False` (vê verdade).

## O que NÃO está sendo usado

- **Câmeras** (RGB ou depth) — não há observação visual.
- **LiDAR** — substituído por `height_scan` raycast simplificado.
- **Sensor de pressão/torque nas juntas** — só posição/velocidade.
- **GPS/odometria global** — política é puramente egocêntrica.

## Resumo da arquitetura sensorial

O Go2 está sendo treinado com um **stack de sensores proprioceptivos** (IMU + encoders +
contato dos pés) **+ percepção local de terreno** (`height_scan` que simula um sensor depth
voltado pra baixo). É a configuração padrão para *blind locomotion + terrain perception*
da literatura de quadrúpedes.

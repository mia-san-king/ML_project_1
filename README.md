# EPFL CS-433 — Project 2

Branch: `project_2_Yijun_Liu` (empty start; add code and report here as the project progresses).

Notes from the project kickoff meeting: datasets, literature, and DHLAB cluster usage. Step-by-step guide (Chinese): [`tutorial.md`](tutorial.md).

## Training datasets

| Source | Link | Notes |
| --- | --- | --- |
| **SMAPshot** (Swiss landscapes) | [Development documentation](https://smapshot.heig-vd.ch/development_documentation) | Historical / georeferenced imagery for Switzerland. |
| **4Dbrowser** (European cities) | [API](https://4dbrowser.urbanhistory4d.org/api) | Urban history 4D browser API for European cities. |

When calling either API, space requests with a **friendly delay** between calls so you do not overload the server.

## Literature (starting points)

| Paper | Link | Local copy |
| --- | --- | --- |
| Preprint (Feb 2023) | [arXiv:2302.00275](https://arxiv.org/pdf/2302.00275) | `papers/arxiv-2302.00275.pdf` |
| NeurIPS 2023 | [Conference paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html) | `papers/neurips-2023-1b57aadd.pdf` |

Also check **recent papers that cite** these two for updated methods and benchmarks.

## DHLAB GPU（Gaspar yijliu）

```bash
ssh dhlab    # 已写入 ~/.ssh/config
```

首次登录后见 [`scripts/dhlab_connect.md`](scripts/dhlab_connect.md) 与 [`scripts/dhlab_once_setup.sh`](scripts/dhlab_once_setup.sh)。

## DHLAB cluster (RCP)

Student how-to (login, jobs, storage):

[EPFL-DHLAB-student-projects — README-Student-HowTo.md](https://github.com/dh-epfl-students/EPFL-DHLAB-student-projects/blob/main/README-Student-HowTo.md)

### Environment

Install Conda envs **in your own scratch path**, not at the cluster root (shared root installs can block others):

```bash
conda create --prefix "/rcp_scratch/students/<your_gaspar>/conda/myenv" python=3.11
conda activate "/rcp_scratch/students/<your_gaspar>/conda/myenv"
```

Replace `<your_gaspar>` with your EPFL username.

### GPUs

- Use **one GPU at a time** per job unless the how-to says otherwise.
- Check availability before launching training: `nvidia-smi`.

## Local layout (to be filled in)

```text
README.md          # this file
papers/            # downloaded reference PDFs
# add project code, configs, report/, data/ (gitignored), etc.
```

Project 1 remains on branch `main` in this repository.

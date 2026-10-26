# APIMS — AI-Powered Predictive Infrastructure Maintenance System

**Author / project concept:** Mahesh Balaraman  
**Repository owner:** [mahesh0606-git](https://github.com/mahesh0606-git)

An **educational, untrained Python reference implementation** based on the uploaded *APIMS Revised Pseudocode Appendix* technical monograph. This is newly written code based on the document's algorithms, **not recovered original Python source code**.

## Included
- `apims/telemetry.py`: ten telemetry features, domain-aware normalization, 128-step windows.
- `apims/model.py`: 10→64 projection, dilated temporal convolution blocks (1,2,4,8), four-head temporal attention, 30/90-day failure probabilities, RUL mean/log-variance, Monte Carlo dropout intervals.
- `apims/scheduler.py`: simplified weighted-fitness evolutionary scheduler, **not full NSGA-II**.
- `demo.py`: synthetic-input demonstration.
- `tests/`: basic unit tests.

## Install and run
Requires Python 3.10+ and PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests
```

## Scientific and implementation limitations
- **Model is randomly initialized and untrained.** Demo outputs are **not meaningful predictions**; never use for operational maintenance decisions.
- No real enterprise telemetry, verified failures, training loop, calibration, field validation, production connectors, or work-order dispatch is included.
- This is **temporal attention**, not the proposed graph-based cross-node topology architecture (GATv2/ST-GNN).
- Causal left padding and monotonic 30/90-day probability parameterization are explicit **improvements** over limitations noted in the document.
- Scheduler uses scalarized evolutionary search, not NSGA-II Pareto ranking/crowding distance, and does not enforce production constraints such as skills, parts, or HA quorum.
- Empirical impact claims in the monograph are **not independently validated** by this repository.
- No forensic/detection modules are included.

## Before real-world use
Obtain authorized, de-identified telemetry and verified failure labels; split chronologically without leakage; train and validate probability calibration, false positives, RUL error, and interval coverage; enforce hard scheduling constraints and human approval.

## Publication and licensing
Review copyright, confidentiality, employer IP ownership, credentials, and publication rights. The source Word document and confidential datasets are not included. **No license is granted in this repository until the project owner explicitly chooses one.**

# Documentation index

## Reading order

1. [Architecture](architecture.md) and [system design diagrams](design.md): implemented foundations and planned system.
2. [HPC runbook](runbooks/hpc_setup.md): source transfer, environment, and execution.
3. [Reproducibility](reproducibility.md): model identity and artifact contracts.
4. [Evaluation](evaluation.md): baselines, metrics, and claim boundaries.
5. [Validation](validation.md): checks actually executed and remaining work.
6. [Git workflow](runbooks/git_workflow.md) and [contribution guide](../CONTRIBUTING.md).
7. [Initial architecture decision](decisions/0001_initial_design.md) and [source layout decision](decisions/0002_source_package.md).

## Maintenance rules

Update code, public usage instructions, and validation evidence together. Keep implementation status current. Distinguish local CPU tests, HPC GPU checks, and GitHub CI execution. Add numbered decision records for consequential changes; preserve the rationale when superseding an earlier decision.

Public documentation must be self-contained. Exclude local notes, authentication codes, and access tokens.

Publish experiment results with their configurations and source artifacts. Add dataset and model documentation alongside the corresponding pipelines.

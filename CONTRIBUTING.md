# Contributing

Thank you for considering a contribution to PhosCrosstalk.

This project is a scientific Python package for modeling phosphorylation crosstalk networks using transcriptomics,
proteomics, phosphoproteomics, kinase-site priors, and mechanistic ODE-based optimization. Contributions should preserve
scientific clarity, reproducibility, and numerical reliability.

## Ways to Contribute

Useful contributions include:

- bug reports with reproducible examples;
- fixes for data-loading, configuration, simulation, optimization, or plotting issues;
- tests for model behavior and edge cases;
- documentation improvements;
- small example datasets or workflows;
- improvements to performance, logging, validation, or error messages;
- scientific discussion of model assumptions.

## Before Opening an Issue

Before opening an issue, check whether:

- the latest version of the code is being used;
- the configuration file is valid;
- required input files exist and use the expected format;
- the issue can be reproduced with a small example;
- existing issues already describe the problem.

For numerical or modeling issues, include the relevant configuration, data dimensions, mechanism used, solver settings,
and the exact error or non-finite output behavior.

## Development Setup

Clone the repository:

```bash
git clone git@github.com:bibymaths/phoscrosstalk.git
cd phoscrosstalk
```
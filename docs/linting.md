# Linting Configuration

## Selected Linter

For static code analysis, **flake8** was selected.

## Why flake8?

Flake8 was chosen because:

* it is simple and easy to use
* it detects style issues and common errors
* it follows PEP8 standards
* it integrates well into development workflow

## Rules

The following rules were configured:

* **max-line-length = 88**
  Limits the maximum length of a line to improve readability

* **exclude = .venv, **pycache****
  Excludes system and temporary folders from analysis

* **ignore = E203, W503**
  Ignores specific warnings that are not critical for this project

## How to run

To run the linter, use the following command:

```
flake8 .
```

## Fixing Issues

After running the linter, several issues were detected and fixed, including:

* incorrect spacing
* missing blank lines
* long lines of code

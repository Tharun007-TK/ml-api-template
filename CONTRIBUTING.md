# Contributing

Thanks for contributing.

## How To Contribute

1. Fork the repository.
2. Create a feature branch.
3. Make focused changes with clear commit messages.
4. Run local checks before opening a pull request.
5. Open a PR with a short summary and testing notes.

## Local Setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Development Notes

- Keep API payload keys consistent with `Banknote.py`.
- If model behavior changes, retrain and regenerate `classifier.pkl`.
- Update README for any endpoint or workflow changes.

## Pull Request Checklist

- [ ] Code runs locally
- [ ] README updated if needed
- [ ] No unrelated files changed
- [ ] Changes are explained clearly

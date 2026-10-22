CONTRADIX landing-page update

This ZIP contains the replacement app.py only. Keep your existing nlp/, utils/, sample_data/, tests/, and requirements files unchanged.

Changes:
- Landing page first: CONTRADIX, tagline, description, three mode cards.
- Removed 'Runs locally on CPU' homepage chip.
- Start analyzing button opens the workspace with all three modes and uploads.
- Back to home button returns to landing page.
- About & methodology remains accessible on landing page and in workspace.
- Retrieval configuration stays fixed internally: 0.43 / 25 / 2 / 0.29 / reverse=True.
- Sidebar starts collapsed.

Install: replace C:\Users\dell\Documents\Contradix_Final\app.py with the app.py from this ZIP.
Run: python -m streamlit run app.py
Test: python -m pytest -q
Push: git add app.py && git commit -m "Add CONTRADIX landing page and gated analysis workspace" && git push origin main

Python syntax compilation passed. Full integration tests must be run in your project folder.

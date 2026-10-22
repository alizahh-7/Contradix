CONTRADIX minimal sumi-e landing page

ZIP contains replacement app.py. Extract it into your existing Contradix_Final project, overwriting app.py. Keep nlp/, utils/, sample_data/ and other files.

Homepage: static embedded SVG mountains, red sun, minimal typography, Start analyzing button, About & methodology expander. Workspace: original three modes, uploads, exports and About tab. No React or extra dependencies.

PowerShell (run from project folder):
Copy-Item .\app.py .\app_before_sumi_e.py
Expand-Archive -Path "$HOME\Downloads\CONTRADIX_Minimal_SumiE_Homepage.zip" -DestinationPath . -Force
python -m streamlit run app.py

Tests: python -m pytest -q
Git: git add app.py; git commit -m "Add minimal sumi-e landing page"; git push origin main

Python syntax compiled successfully; full app tests not run here.
